import torch
import numpy as np
from typing import Optional, List, Tuple, Union, Sequence

from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.config import NetworkConfig
from mixed_order.topology import Topology
from mixed_order.theory import compute_q_from_budget
from mixed_order.data.iid import generate_iid_patterns, generate_noise_inits
from mixed_order.data.structured import generate_structured_patterns

from mixed_order.utils import seed_all, make_generator, pin_and_move

import concurrent.futures
import os
from functools import partial

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# `seed_all` and `make_generator` are provided by `mixed_order.utils`


def _load_or_generate_topology(
    N: int,
    p: float,
    beta: float,
    lam: float,
    seed: int,
    topology_device: str,
    topology_cache: Optional[dict] = None,
):
    cache_key = (int(N), float(p), float(beta), int(seed), bool(abs(lam) > 0.0), str(topology_device))
    if topology_cache is not None:
        cached = topology_cache.get(cache_key)
        if cached is not None:
            return cached

    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=topology_device)
    topology = Topology(config)
    topology.generate_masks(
        generator=make_generator(seed, topology_device),
        device_override=topology_device,
    )

    cached = (
        topology.c.detach().cpu(),
        None if topology.tri_i is None else topology.tri_i.detach().cpu(),
        None if topology.tri_j is None else topology.tri_j.detach().cpu(),
        None if topology.tri_k is None else topology.tri_k.detach().cpu(),
        topology.n_tri,
    )
    if topology_cache is not None:
        topology_cache[cache_key] = cached
    return cached


def _generate_seed_patterns(
    P_max: int,
    N: int,
    seed: int,
    device: str,
    F: Optional[torch.Tensor] = None,
):
    pattern_generator = make_generator(seed ^ 0xABCDEF, device)
    if F is not None:
        return generate_structured_patterns(P_max, F, device=device, generator=pattern_generator)
    return generate_iid_patterns(P_max, N, device=device, generator=pattern_generator).to(dtype=torch.float32)


def _generate_seed_initial_states(
    patterns: torch.Tensor,
    p_list: Sequence[int],
    n_trials: int,
    noise_level: float,
    seed: int,
    device: str,
):
    N = patterns.shape[-1]
    n_flip = min(N, max(1, int(noise_level * N)))
    B = len(p_list)
    targets = torch.empty(B, n_trials, N, device=device)
    inits = torch.empty(B, n_trials, N, device=device)

    for b, P in enumerate(p_list):
        mu_indices = torch.arange(n_trials, device=device) % P
        target = patterns[mu_indices].to(dtype=torch.float32)
        targets[b] = target
        inits[b] = generate_noise_inits(
            target,
            n_flip,
            generator=make_generator(seed * 10007 + 7919 * (b + 1), device),
        )

    return inits, targets


def _seed_worker(
    seed: int,
    P_max: int,
    N: int,
    p: float,
    beta: float,
    lam: float,
    n_trials: int,
    p_list: Sequence[int],
    noise_level: float,
    F_cpu: Optional[torch.Tensor],
    topology_cache: Optional[dict],
):
    """
    Thread worker that generates topology, patterns, initial states and targets on CPU.
    Returns CPU tensors pinned in host memory to allow async transfers to GPU.
    """
    # Generate or load topology on CPU (uses topology_cache if provided)
    c_cpu, tri_i_cpu, tri_j_cpu, tri_k_cpu, n_tri = _load_or_generate_topology(
        N=N, p=p, beta=beta, lam=lam, seed=seed, topology_device='cpu', topology_cache=topology_cache
    )

    # Generate patterns on CPU
    patterns = _generate_seed_patterns(P_max, N, seed, device='cpu', F=F_cpu)

    # Generate inits and targets on CPU
    inits, targets = _generate_seed_initial_states(
        patterns=patterns, p_list=p_list, n_trials=n_trials, noise_level=noise_level, seed=seed, device='cpu'
    )

    # Pin CPU memory to enable async host->device copies
    c_cpu = pin_and_move(c_cpu, 'cpu')
    if tri_i_cpu is not None:
        tri_i_cpu = pin_and_move(tri_i_cpu, 'cpu')
        tri_j_cpu = pin_and_move(tri_j_cpu, 'cpu')
        tri_k_cpu = pin_and_move(tri_k_cpu, 'cpu')

    patterns = pin_and_move(patterns, 'cpu')
    inits = pin_and_move(inits, 'cpu')
    targets = pin_and_move(targets, 'cpu')

    return c_cpu, tri_i_cpu, tri_j_cpu, tri_k_cpu, n_tri, patterns, inits, targets


def _pad_seed_triangles(
    tri_triplets: Sequence[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]],
    device: str,
):
    max_tri = max((n_tri for _, _, _, n_tri in tri_triplets), default=0)
    if max_tri == 0:
        return None, None, None, None

    batch_size = len(tri_triplets)

    # Build the padded triangle arrays on CPU first to avoid many small
    # device moves / pin operations inside the loop. Move once at the end.
    cpu = 'cpu'
    tri_i_cpu = torch.full((batch_size, max_tri), -1, dtype=torch.long, device=cpu)
    tri_j_cpu = torch.full((batch_size, max_tri), -1, dtype=torch.long, device=cpu)
    tri_k_cpu = torch.full((batch_size, max_tri), -1, dtype=torch.long, device=cpu)
    tri_mask_cpu = torch.zeros((batch_size, max_tri), dtype=torch.bool, device=cpu)

    for seed_idx, (tri_i, tri_j, tri_k, n_tri) in enumerate(tri_triplets):
        if n_tri == 0:
            continue
        if tri_i is not None:
            # tri_i/tri_j/tri_k should already be CPU tensors from the worker.
            tri_i_cpu[seed_idx, :n_tri] = tri_i.to(dtype=torch.long, device=cpu)
            tri_j_cpu[seed_idx, :n_tri] = tri_j.to(dtype=torch.long, device=cpu)
            tri_k_cpu[seed_idx, :n_tri] = tri_k.to(dtype=torch.long, device=cpu)
        tri_mask_cpu[seed_idx, :n_tri] = True

    # If the caller wants CPU tensors, return pinned CPU buffers where possible.
    if device == cpu:
        try:
            if not tri_i_cpu.is_pinned():
                tri_i_cpu = tri_i_cpu.pin_memory()
            if not tri_j_cpu.is_pinned():
                tri_j_cpu = tri_j_cpu.pin_memory()
            if not tri_k_cpu.is_pinned():
                tri_k_cpu = tri_k_cpu.pin_memory()
            if not tri_mask_cpu.is_pinned():
                tri_mask_cpu = tri_mask_cpu.pin_memory()
        except Exception:
            pass
        return tri_i_cpu, tri_j_cpu, tri_k_cpu, tri_mask_cpu

    # Move the full batches to the target device in one operation.
    tri_i_batch = pin_and_move(tri_i_cpu, device)
    tri_j_batch = pin_and_move(tri_j_cpu, device)
    tri_k_batch = pin_and_move(tri_k_cpu, device)
    tri_mask_batch = pin_and_move(tri_mask_cpu, device, dtype=torch.bool)
    return tri_i_batch, tri_j_batch, tri_k_batch, tri_mask_batch


def _store_batched_multiple_p(
    patterns_batch: torch.Tensor,
    p_list: Sequence[int],
    c_batch: torch.Tensor,
    tri_i_batch: Optional[torch.Tensor],
    tri_j_batch: Optional[torch.Tensor],
    tri_k_batch: Optional[torch.Tensor],
    tri_mask_batch: Optional[torch.Tensor],
    N: int,
    p: float,
    beta: float,
    lam: float,
    centered: bool = False,
    C: Optional[torch.Tensor] = None,
    triangle_chunk_size: int = 32768,
):
    seed_count = patterns_batch.shape[0]
    batch_count = len(p_list)
    device = patterns_batch.device
    p_list = sorted(p_list)

    # Decide whether to construct dense pairwise matrices or a sparse edge-based representation.
    config = NetworkConfig(N=N, p=p, beta=beta)
    use_sparse = False
    edge_i_batch = edge_j_batch = edge_mask_batch = None
    J_vals_batch = None

    if p > 0:
        # Estimate sparsity from provided topology masks `c_batch` if available.
        try:
            c_batch = pin_and_move(c_batch, device)
            upper = c_batch.triu(diagonal=1)
            n_edges_per_seed = (upper != 0).sum(dim=(1, 2))
            total_pairs = N * (N - 1) // 2
            frac_mean = float(n_edges_per_seed.float().mean().item()) / float(total_pairs) if total_pairs > 0 else 0.0
            use_sparse = (frac_mean <= config.sparse_threshold) and (int(n_edges_per_seed.max().item()) > 0)
        except Exception:
            use_sparse = (p <= config.sparse_threshold)

    if not use_sparse:
        # Dense accumulation (original path), but per-seed.
        J_batch = torch.zeros(seed_count, batch_count, N, N, device=device)
        if p > 0:
            J_acc = torch.zeros(seed_count, N, N, device=device)
            current_P = 0
            C_device = None if C is None else pin_and_move(C, device)
            c_batch = pin_and_move(c_batch, device)

            for b, P in enumerate(p_list):
                if P > current_P:
                    p_chunk = patterns_batch[:, current_P:P, :]
                    if p_chunk.shape[1] > 0:
                        J_acc = J_acc + torch.bmm(p_chunk.transpose(1, 2), p_chunk)
                    current_P = P

                J_b = J_acc
                if centered and C_device is not None:
                    J_b = J_b - P * C_device.unsqueeze(0)
                J_batch[:, b] = (J_b / (p * N)) * c_batch
        else:
            J_batch = torch.zeros(seed_count, batch_count, N, N, device=device)

    else:
        # Sparse edge-based accumulation (pad per-seed edge lists)
        # Build upper-triangular edge lists per seed
        upper = c_batch.triu(diagonal=1)
        edge_idx_lists = [torch.nonzero(upper[s], as_tuple=False) for s in range(seed_count)]
        n_edges_per_seed = [idx.shape[0] for idx in edge_idx_lists]
        max_edges = int(max(n_edges_per_seed)) if len(n_edges_per_seed) > 0 else 0

        if max_edges == 0:
            # No edges found for any seed -> fall back to zero dense J
            J_batch = torch.zeros(seed_count, batch_count, N, N, device=device)
        else:
            edge_i_batch = torch.full((seed_count, max_edges), -1, dtype=torch.long, device=device)
            edge_j_batch = torch.full((seed_count, max_edges), -1, dtype=torch.long, device=device)
            edge_mask_batch = torch.zeros((seed_count, max_edges), dtype=torch.bool, device=device)

            for s, idx in enumerate(edge_idx_lists):
                if idx.numel() == 0:
                    continue
                idx = pin_and_move(idx, device)
                edge_i_batch[s, : idx.shape[0]] = idx[:, 0]
                edge_j_batch[s, : idx.shape[0]] = idx[:, 1]
                edge_mask_batch[s, : idx.shape[0]] = True

            # Accumulate per-edge sums across patterns similar to dense logic
            J_vals_batch = torch.zeros(seed_count, batch_count, max_edges, device=device, dtype=patterns_batch.dtype)
            J_acc_edges = torch.zeros(seed_count, max_edges, device=device, dtype=patterns_batch.dtype)
            current_P = 0

            edge_i_safe = edge_i_batch.clamp(min=0)
            edge_j_safe = edge_j_batch.clamp(min=0)

            C_edge = None
            if centered and C is not None:
                C_device = pin_and_move(C, device)
                # Advanced-index: (seed_count, max_edges)
                C_edge = C_device[edge_i_safe, edge_j_safe]

            for b, P in enumerate(p_list):
                if P > current_P:
                    p_chunk = patterns_batch[:, current_P:P, :]
                    chunk_size = p_chunk.shape[1]
                    if chunk_size > 0:
                        gather_i = edge_i_safe[:, None, :].expand(seed_count, chunk_size, max_edges)
                        gather_j = edge_j_safe[:, None, :].expand(seed_count, chunk_size, max_edges)

                        p_i = torch.gather(p_chunk, 2, gather_i)
                        p_j = torch.gather(p_chunk, 2, gather_j)

                        prod = (p_i * p_j).sum(dim=1) * edge_mask_batch.to(dtype=patterns_batch.dtype)
                        J_acc_edges += prod
                    current_P = P

                if centered and C_edge is not None:
                    J_vals_b = (J_acc_edges - P * C_edge) / (p * N)
                else:
                    J_vals_b = J_acc_edges / (p * N) if p > 0 else torch.zeros_like(J_acc_edges)

                J_vals_batch[:, b, :] = J_vals_b
            # No dense J_batch allocated for sparse path
            J_batch = None

    q, _ = compute_q_from_budget(p, N, beta)
    if lam <= 0.0 or q <= 0.0 or tri_i_batch is None or tri_i_batch.numel() == 0:
        W_batch = None
    else:
        tri_count = tri_i_batch.shape[1]
        W_batch = torch.zeros(seed_count, batch_count, tri_count, device=device)
        W_acc = torch.zeros(seed_count, tri_count, device=device)
        tri_i_safe = tri_i_batch.clamp(min=0)
        tri_j_safe = tri_j_batch.clamp(min=0)
        tri_k_safe = tri_k_batch.clamp(min=0)
        tri_mask = pin_and_move(tri_mask_batch, device, dtype=patterns_batch.dtype)
        current_P = 0
        norm = lam / (q * N**2)

        for b, P in enumerate(p_list):
            if P > current_P:
                p_chunk = patterns_batch[:, current_P:P, :]
                chunk_size = p_chunk.shape[1]
                if chunk_size > 0:
                    for tri_start in range(0, tri_count, triangle_chunk_size):
                        tri_end = min(tri_start + triangle_chunk_size, tri_count)
                        tri_len = tri_end - tri_start

                        idx_i = tri_i_safe[:, tri_start:tri_end]
                        idx_j = tri_j_safe[:, tri_start:tri_end]
                        idx_k = tri_k_safe[:, tri_start:tri_end]
                        tri_valid = tri_mask[:, tri_start:tri_end]

                        gather_i = idx_i[:, None, :].expand(seed_count, chunk_size, tri_len)
                        gather_j = idx_j[:, None, :].expand(seed_count, chunk_size, tri_len)
                        gather_k = idx_k[:, None, :].expand(seed_count, chunk_size, tri_len)

                        p_i = torch.gather(p_chunk, 2, gather_i)
                        p_j = torch.gather(p_chunk, 2, gather_j)
                        p_k = torch.gather(p_chunk, 2, gather_k)

                        W_acc[:, tri_start:tri_end] += (
                            (p_i * p_j * p_k).sum(dim=1) * tri_valid
                        )

                current_P = P

            W_batch[:, b] = W_acc * norm

    return J_batch, W_batch, edge_i_batch, edge_j_batch, J_vals_batch, edge_mask_batch



def _run_seed_batch_dynamics(
    state_batch: torch.Tensor,
    J_batch: Optional[torch.Tensor],
    W_batch: Optional[torch.Tensor],
    tri_i_batch: Optional[torch.Tensor],
    tri_j_batch: Optional[torch.Tensor],
    tri_k_batch: Optional[torch.Tensor],
    tri_mask_batch: Optional[torch.Tensor],
    J_edge_i_batch: Optional[torch.Tensor] = None,
    J_edge_j_batch: Optional[torch.Tensor] = None,
    J_vals_batch: Optional[torch.Tensor] = None,
    J_edge_mask_batch: Optional[torch.Tensor] = None,
    max_steps: int = 100,
    triangle_chunk_size: int = 32768,
    convergence_check_interval: int = 8,
):
    seed_count, batch_count, n_trials, N = state_batch.shape
    flat_count = seed_count * batch_count
    state_flat = state_batch.reshape(flat_count, n_trials, N)
    ones = torch.ones(1, device=state_batch.device, dtype=state_batch.dtype)
    neg = -ones

    # Prepare triangle data if present
    has_triangles = (
        W_batch is not None
        and tri_i_batch is not None
        and tri_i_batch.numel() > 0
        and tri_mask_batch is not None
    )

    if has_triangles:
        tri_count = tri_i_batch.shape[1]
        tri_i_flat = tri_i_batch[:, None, :].expand(seed_count, batch_count, tri_count).reshape(flat_count, tri_count)
        tri_j_flat = tri_j_batch[:, None, :].expand(seed_count, batch_count, tri_count).reshape(flat_count, tri_count)
        tri_k_flat = tri_k_batch[:, None, :].expand(seed_count, batch_count, tri_count).reshape(flat_count, tri_count)
        tri_mask_flat = tri_mask_batch[:, None, :].expand(seed_count, batch_count, tri_count).reshape(flat_count, tri_count)
        W_flat = W_batch.reshape(flat_count, tri_count)
    else:
        tri_count = 0
        tri_i_flat = tri_j_flat = tri_k_flat = tri_mask_flat = W_flat = None

    # Prepare pairwise (2-body) representations
    # Note: for sparse-J support, the caller supplies padded edge indices and per-edge values.
    is_dense_J = J_batch is not None
    has_sparse_J = (
        J_vals_batch is not None and J_edge_i_batch is not None and J_edge_j_batch is not None and J_edge_mask_batch is not None
    )

    if has_sparse_J:
        edge_count = J_edge_i_batch.shape[1]
        edge_i_flat = J_edge_i_batch[:, None, :].expand(seed_count, batch_count, edge_count).reshape(flat_count, edge_count)
        edge_j_flat = J_edge_j_batch[:, None, :].expand(seed_count, batch_count, edge_count).reshape(flat_count, edge_count)
        edge_mask_flat = J_edge_mask_batch[:, None, :].expand(seed_count, batch_count, edge_count).reshape(flat_count, edge_count)
        J_vals_flat = J_vals_batch.reshape(flat_count, edge_count)

    # Reduce host<->device synchronizations by checking convergence only periodically
    for step in range(max_steps):
        if is_dense_J:
            J_flat = J_batch.reshape(flat_count, N, N)
            h = torch.bmm(state_flat, J_flat.transpose(1, 2))
        elif has_sparse_J:
            # Build pairwise field via gathers/scatter_add in edge chunks
            h = torch.zeros(flat_count, n_trials, N, device=state_flat.device, dtype=state_flat.dtype)
            if edge_count > 0:
                for e_start in range(0, edge_count, triangle_chunk_size):
                    e_end = min(e_start + triangle_chunk_size, edge_count)
                    e_len = e_end - e_start

                    idx_i = edge_i_flat[:, e_start:e_end].clamp(min=0)
                    idx_j = edge_j_flat[:, e_start:e_end].clamp(min=0)
                    mask_e = edge_mask_flat[:, e_start:e_end].to(state_flat.dtype)
                    Jvals_e = J_vals_flat[:, e_start:e_end]

                    gather_i = idx_i[:, None, :].expand(flat_count, n_trials, e_len)
                    gather_j = idx_j[:, None, :].expand(flat_count, n_trials, e_len)

                    si = torch.gather(state_flat, 2, gather_i)
                    sj = torch.gather(state_flat, 2, gather_j)

                    weights = Jvals_e[:, None, :] * mask_e[:, None, :]

                    h.scatter_add_(2, gather_i, weights * sj)
                    h.scatter_add_(2, gather_j, weights * si)
        else:
            h = torch.zeros(flat_count, n_trials, N, device=state_flat.device, dtype=state_flat.dtype)

        if tri_count > 0:
            for tri_start in range(0, tri_count, triangle_chunk_size):
                tri_end = min(tri_start + triangle_chunk_size, tri_count)
                tri_len = tri_end - tri_start

                idx_i = tri_i_flat[:, tri_start:tri_end].clamp(min=0)
                idx_j = tri_j_flat[:, tri_start:tri_end].clamp(min=0)
                idx_k = tri_k_flat[:, tri_start:tri_end].clamp(min=0)
                tri_valid = tri_mask_flat[:, tri_start:tri_end].to(state_flat.dtype)

                gather_i = idx_i[:, None, :].expand(flat_count, n_trials, tri_len)
                gather_j = idx_j[:, None, :].expand(flat_count, n_trials, tri_len)
                gather_k = idx_k[:, None, :].expand(flat_count, n_trials, tri_len)

                si = torch.gather(state_flat, 2, gather_i)
                sj = torch.gather(state_flat, 2, gather_j)
                sk = torch.gather(state_flat, 2, gather_k)

                weights = W_flat[:, tri_start:tri_end] * tri_valid
                weights = weights[:, None, :]

                h.scatter_add_(2, gather_i, weights * sj * sk)
                h.scatter_add_(2, gather_j, weights * si * sk)
                h.scatter_add_(2, gather_k, weights * si * sj)

        new_state = torch.where(h >= 0.0, ones, neg)
        # Check convergence less frequently to avoid repeated device->host syncs.
        if (step % convergence_check_interval) == 0:
            # Use torch.equal (C-implemented) to detect exact equality.
            if torch.equal(new_state, state_flat):
                state_flat = new_state
                break
        state_flat = new_state

    return state_flat.reshape(seed_count, batch_count, n_trials, N)


@torch.inference_mode()
def _run_seed_batch_retrieval(
    seed_batch: Sequence[int],
    N: int,
    p_list: List[int],
    p: float,
    beta: float = 0.5,
    n_trials: int = 5,
    max_steps: int = 100,
    lam: float = 2.0,
    device: str = _DEVICE,
    topology_cache: Optional[dict] = None,
    topology_device: Optional[str] = None,
    F: Optional[torch.Tensor] = None,
    C: Optional[torch.Tensor] = None,
    centered: bool = False,
    noise_level: float = 0.10,
    triangle_chunk_size: int = 8192,
):
    p_list = sorted(int(p_value) for p_value in p_list)
    topology_device = topology_device if topology_device is not None else device
    seed_batch = [int(seed) for seed in seed_batch]
    seed_count = len(seed_batch)
    P_max = max(p_list)

    F_device = None if F is None else pin_and_move(F, device)
    C_device = None if C is None else pin_and_move(C, device)

    c_list = []
    tri_triplets = []
    patterns_list = []
    inits_list = []
    targets_list = []

    # Parallelize CPU-side generation of topology and patterns across seeds
    # Prepare CPU copy of structured factors if provided
    F_cpu = None if F is None else F.to('cpu')

    worker = partial(
        _seed_worker,
        P_max=P_max,
        N=N,
        p=p,
        beta=beta,
        lam=lam,
        n_trials=n_trials,
        p_list=p_list,
        noise_level=noise_level,
        F_cpu=F_cpu,
        topology_cache=topology_cache,
    )

    max_workers = min(seed_count, max(1, (os.cpu_count() or 2)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker, seed_batch))

    for c_cpu, tri_i_cpu, tri_j_cpu, tri_k_cpu, n_tri, patterns, inits, targets in results:
        c_list.append(c_cpu)
        tri_triplets.append((tri_i_cpu, tri_j_cpu, tri_k_cpu, n_tri))
        patterns_list.append(patterns)
        inits_list.append(inits)
        targets_list.append(targets)

    tri_i_batch, tri_j_batch, tri_k_batch, tri_mask_batch = _pad_seed_triangles(tri_triplets, device)

    # Stack CPU-pinned tensors then move to device in larger batches (async if possible)
    c_batch = pin_and_move(torch.stack(c_list, dim=0), device)
    patterns_batch = pin_and_move(torch.stack(patterns_list, dim=0), device)
    inits_batch = pin_and_move(torch.stack(inits_list, dim=0), device)
    targets_batch = pin_and_move(torch.stack(targets_list, dim=0), device)
    J_batch, W_batch, edge_i_batch, edge_j_batch, J_vals_batch, edge_mask_batch = _store_batched_multiple_p(
        patterns_batch=patterns_batch,
        p_list=p_list,
        c_batch=c_batch,
        tri_i_batch=tri_i_batch,
        tri_j_batch=tri_j_batch,
        tri_k_batch=tri_k_batch,
        tri_mask_batch=tri_mask_batch,
        N=N,
        p=p,
        beta=beta,
        lam=lam,
        centered=centered,
        C=C_device,
        triangle_chunk_size=triangle_chunk_size,
    )
    finals = _run_seed_batch_dynamics(
        state_batch=inits_batch,
        J_batch=J_batch,
        W_batch=W_batch,
        tri_i_batch=tri_i_batch,
        tri_j_batch=tri_j_batch,
        tri_k_batch=tri_k_batch,
        tri_mask_batch=tri_mask_batch,
        J_edge_i_batch=edge_i_batch,
        J_edge_j_batch=edge_j_batch,
        J_vals_batch=J_vals_batch,
        J_edge_mask_batch=edge_mask_batch,
        max_steps=max_steps,
        triangle_chunk_size=triangle_chunk_size,
    )
    overlaps = compute_overlap(finals, targets_batch)
    return overlaps

def compute_overlap(finals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute overlap m = (1/N) * sum(x_i * xi_i).
    Shape: (..., N) -> (...)
    """
    N = finals.shape[-1]
    return (finals * targets).sum(dim=-1) / N

def compute_success(overlaps: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """
    Return boolean tensor of success based on overlap threshold.
    """
    return overlaps >= threshold

def aggregate_success(success: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute success rate along specified dimension.
    """
    return success.to(dtype=torch.float32).mean(dim=dim)

def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_boot: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    """
    if len(values) < 2:
        return np.mean(values), np.mean(values)
    resamples = np.random.choice(values, size=(n_boot, len(values)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower
    return float(np.quantile(means, lower)), float(np.quantile(means, upper))

def run_batched_retrieval(N: int, p_list: List[int], p: float, beta: float = 0.5, 
                          n_trials: int = 5, max_steps: int = 100, seed: int = 42, 
                          lam: float = 2.0, device: str = _DEVICE, 
                          topology_cache: Optional[dict] = None,
                          topology_device: Optional[str] = None,
                          F: Optional[torch.Tensor] = None, 
                          C: Optional[torch.Tensor] = None, 
                          centered: bool = False,
                          noise_level: float = 0.10,
                          triangle_chunk_size: int = 8192):
    """
    Run retrieval for a batch of capacities (P values) simultaneously.
    Returns: Tensor of shape (len(p_list), n_trials) containing overlaps.
    """
    overlaps = _run_seed_batch_retrieval(
        seed_batch=[seed],
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        max_steps=max_steps,
        lam=lam,
        device=device,
        topology_cache=topology_cache,
        topology_device=topology_device,
        F=F,
        C=C,
        centered=centered,
        noise_level=noise_level,
        triangle_chunk_size=triangle_chunk_size,
    )
    return overlaps[0]


@torch.inference_mode()
def run_batched_retrieval_many_seeds(
    seed_batch: Sequence[int],
    N: int,
    p_list: List[int],
    p: float,
    beta: float = 0.5,
    n_trials: int = 5,
    max_steps: int = 100,
    lam: float = 2.0,
    device: str = _DEVICE,
    topology_cache: Optional[dict] = None,
    topology_device: Optional[str] = None,
    F: Optional[torch.Tensor] = None,
    C: Optional[torch.Tensor] = None,
    centered: bool = False,
    noise_level: float = 0.10,
    triangle_chunk_size: int = 8192,
):
    """
    Run retrieval for multiple seeds and capacities in one call.
    Returns: Tensor of shape (len(seed_batch), len(p_list), n_trials).
    """
    return _run_seed_batch_retrieval(
        seed_batch=seed_batch,
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        max_steps=max_steps,
        lam=lam,
        device=device,
        topology_cache=topology_cache,
        topology_device=topology_device,
        F=F,
        C=C,
        centered=centered,
        noise_level=noise_level,
        triangle_chunk_size=triangle_chunk_size,
    )

def find_empirical_pc_by_success(N: int, p: float, beta: float, lam: float, 
                                 n_trials: int, n_seeds: int,
                                 success_threshold: float = 0.99,
                                 overlap_threshold: float = 0.99,
                                 noise_level: float = 0.10,
                                 pbar=None, device: str = _DEVICE,
                                 F: Optional[torch.Tensor] = None, 
                                 C: Optional[torch.Tensor] = None, 
                                 centered: bool = False,
                                 topology_cache: Optional[dict] = None,
                                 topology_device: Optional[str] = None,
                                 seed_batch_size: Optional[int] = None,
                                 triangle_chunk_size: int = 8192):
    """
    Find critical capacity Pc based on success rate.
    Returns: Pc (float) and optionally full sweep data if requested.
    """
    # Define search candidates
    max_p = int(min(N, max(64, int(0.8 * N))))
    step  = max(1, N // 20)
    p_candidates = np.unique(np.concatenate([
        np.arange(1, 15),
        np.arange(15, max_p + 1, step),
    ])).astype(int)

    # Allow callers to provide a shared topology cache to avoid
    # recomputing expensive masks (triangles / edges) across many
    # repeated calls (e.g. heatmap sweeps). If no cache is provided
    # create a local one.
    if topology_cache is None:
        topology_cache = {}
    
    if seed_batch_size is None:
        seed_batch_size = 4 if device == "cuda" else 1
    seed_batch_size = max(1, int(seed_batch_size))

    all_overlaps = []
    for batch_start in range(0, n_seeds, seed_batch_size):
        batch_seeds = list(range(batch_start, min(n_seeds, batch_start + seed_batch_size)))
        overlaps = _run_seed_batch_retrieval(
            seed_batch=batch_seeds,
            N=N,
            p_list=p_candidates.tolist(),
            p=p,
            beta=beta,
            n_trials=n_trials,
            max_steps=100,
            lam=lam,
            device=device,
            topology_cache=topology_cache,
            topology_device=topology_device,
            F=F,
            C=C,
            centered=centered,
            noise_level=noise_level,
            triangle_chunk_size=triangle_chunk_size,
        )
        all_overlaps.append(overlaps.cpu().numpy())
        if pbar is not None:
            pbar.set_postfix(p=f"{p:.2f}", seed=f"{batch_seeds[-1] + 1}/{n_seeds}")

    # (n_seeds, B, n_trials)
    all_overlaps = np.concatenate(all_overlaps, axis=0)
    # success: (n_seeds, B, n_trials)
    success = all_overlaps >= overlap_threshold
    # success_rate_per_seed: (n_seeds, B)
    success_rate_per_seed = success.mean(axis=2)
    # mean_success_rate: (B,)
    mean_success_rate = success_rate_per_seed.mean(axis=0)
    
    # Find boundary in coarse sweep
    best_idx = -1
    for i, rate in enumerate(mean_success_rate):
        if rate >= success_threshold:
            best_idx = i
        else:
            break
            
    if best_idx == -1: return 0.0
    if best_idx == len(p_candidates) - 1: return float(p_candidates[-1])
    
    # Fine sweep
    p_fine = np.arange(p_candidates[best_idx] + 1, p_candidates[best_idx+1]).astype(int)
    if len(p_fine) == 0:
        return float(p_candidates[best_idx])

    all_overlaps_fine = []
    for batch_start in range(0, n_seeds, seed_batch_size):
        batch_seeds = list(range(batch_start, min(n_seeds, batch_start + seed_batch_size)))
        overlaps = _run_seed_batch_retrieval(
            seed_batch=batch_seeds,
            N=N,
            p_list=p_fine.tolist(),
            p=p,
            beta=beta,
            n_trials=n_trials,
            max_steps=100,
            lam=lam,
            device=device,
            topology_cache=topology_cache,
            topology_device=topology_device,
            F=F,
            C=C,
            centered=centered,
            noise_level=noise_level,
            triangle_chunk_size=triangle_chunk_size,
        )
        all_overlaps_fine.append(overlaps.cpu().numpy())

    # (n_seeds, B_fine, n_trials)
    all_overlaps_fine = np.concatenate(all_overlaps_fine, axis=0)
    mean_success_fine = (all_overlaps_fine >= overlap_threshold).mean(axis=(0, 2))
    
    all_p = np.concatenate([[p_candidates[best_idx]], p_fine, [p_candidates[best_idx+1]]])
    all_rates = np.concatenate([[mean_success_rate[best_idx]], mean_success_fine, [mean_success_rate[best_idx+1]]])
    
    for i in range(len(all_rates) - 1):
        if all_rates[i] >= success_threshold and all_rates[i+1] < success_threshold:
            # Linear interpolation for fractional Pc
            denom = all_rates[i] - all_rates[i+1]
            frac = (all_rates[i] - success_threshold) / denom
            return all_p[i] + frac * (all_p[i+1] - all_p[i])
            
    return float(all_p[best_idx])

# Backward-compatible alias for legacy imports.
find_empirical_pc = find_empirical_pc_by_success
