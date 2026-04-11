from __future__ import annotations

# Ensure repository root is on sys.path when running this script directly
import sys
import pathlib
_file = pathlib.Path(__file__).resolve()
_repo_root = None
for _ancestor in _file.parents:
    if _ancestor.name == "experiments":
        _repo_root = _ancestor.parent
        break
if _repo_root is None:
    _repo_root = _file.parents[1] if len(_file.parents) >= 2 else _file.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from experiments.common import ensure_dir, save_npz
from mixed_order.data.iid import generate_iid_patterns
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success
from mixed_order.theory import compute_q_from_budget, optimal_lambda
from mixed_order.utils import make_generator
from mixed_order_model import MixedOrderHopfieldNetwork
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _capacity_grid(
    N: int,
    beta: float,
    p_vals: np.ndarray,
    lam_vals: np.ndarray,
    n_trials: int,
    n_seeds: int,
    device: str,
    topology_cache: dict,
) -> np.ndarray:
    out = np.zeros((lam_vals.size, p_vals.size), dtype=float)
    for i, lam in enumerate(tqdm(lam_vals, desc="lambda sweep")):
        for j, p in enumerate(p_vals):
            _, q_tilde = compute_q_from_budget(float(p), N, beta)
            if q_tilde <= 1e-9:
                continue
            out[i, j] = find_empirical_pc_by_success(
                N=N,
                p=float(p),
                beta=beta,
                lam=float(lam),
                n_trials=n_trials,
                n_seeds=n_seeds,
                success_threshold=0.99,
                overlap_threshold=0.99,
                noise_level=0.1,
                device=device,
                topology_cache=topology_cache,
            )
    return out


def _measure_signal_noise(
    N: int,
    beta: float,
    p_lam_pairs: List[Tuple[float, float]],
    p_grid: np.ndarray,
    n_seeds: int,
    device: str,
) -> Dict[str, np.ndarray]:
    sig_emp = np.zeros((len(p_lam_pairs), p_grid.size), dtype=float)
    tau2_emp = np.zeros_like(sig_emp)
    sig_th = np.zeros_like(sig_emp)
    tau2_th = np.zeros_like(sig_emp)

    for k, (p, lam) in enumerate(tqdm(p_lam_pairs, desc="signal-noise pairs")):
        _, q_tilde = compute_q_from_budget(p, N, beta)
        p_list = p_grid.astype(int).tolist()
        p_max = int(max(p_list))
        s_seed = []
        n_seed = []
        for seed in tqdm(range(n_seeds), desc="seeds", leave=False):
            model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=lam, device=device)
            model.generate_masks(generator=make_generator(seed, device))
            patterns = generate_iid_patterns(p_max, N, device=device, generator=make_generator(seed + 2718, device)).to(torch.float32)
            model.store_multiple_p(patterns, p_list)

            mu_idx = torch.tensor([seed % P for P in p_list], dtype=torch.long, device=device)
            xi = patterns[mu_idx]  # (B, N)
            h = model.local_field(xi.unsqueeze(1)).squeeze(1)  # (B, N)
            aligned = h * xi
            m = aligned.mean(dim=1)
            s_seed.append(m)
            n_seed.append(torch.var(aligned, dim=1, unbiased=False))

        sig_emp[k] = torch.stack(s_seed, dim=0).mean(dim=0).detach().cpu().numpy()
        tau2_emp[k] = torch.stack(n_seed, dim=0).mean(dim=0).detach().cpu().numpy()

        alpha = p_grid.astype(float) / float(N)
        sig_th[k] = 1.0 + lam / 2.0
        tau2_th[k] = alpha * (1.0 / p + (lam * lam) / (2.0 * q_tilde))

    return {
        "signal_emp": sig_emp,
        "tau2_emp": tau2_emp,
        "signal_theory": sig_th,
        "tau2_theory": tau2_th,
    }


def _mixed_feature_gram(
    N: int,
    p: float,
    beta: float,
    lam: float,
    p_grid: np.ndarray,
    seed: int,
    device: str,
) -> Dict[str, np.ndarray]:
    model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=lam, device=device)
    model.generate_masks(generator=make_generator(seed, device))

    edge_idx = torch.nonzero(torch.triu(model.c, diagonal=1), as_tuple=False)
    ei = edge_idx[:, 0] if edge_idx.numel() else torch.empty(0, dtype=torch.long, device=device)
    ej = edge_idx[:, 1] if edge_idx.numel() else torch.empty(0, dtype=torch.long, device=device)
    tri_i = model.tri_i if model.n_tri > 0 else torch.empty(0, dtype=torch.long, device=device)
    tri_j = model.tri_j if model.n_tri > 0 else torch.empty(0, dtype=torch.long, device=device)
    tri_k = model.tri_k if model.n_tri > 0 else torch.empty(0, dtype=torch.long, device=device)

    diag_mass = []
    offdiag_var = []
    ratio = []

    for P in tqdm(p_grid.astype(int).tolist(), desc="Gram P", leave=False):
        x = generate_iid_patterns(P, N, device=device, generator=make_generator(seed + 100 + P, device)).to(torch.float32)

        pair_feat = x[:, ei] * x[:, ej] if ei.numel() else torch.zeros((P, 0), device=device, dtype=torch.float32)
        tri_feat = x[:, tri_i] * x[:, tri_j] * x[:, tri_k] if tri_i.numel() else torch.zeros((P, 0), device=device, dtype=torch.float32)
        if tri_feat.shape[1] > 0:
            tri_feat = float(np.sqrt(max(lam, 1e-12))) * tri_feat
        phi = torch.cat([pair_feat, tri_feat], dim=1)
        if phi.shape[1] == 0:
            K = torch.zeros((P, P), device=device, dtype=torch.float32)
        else:
            K = (phi @ phi.T) / float(phi.shape[1])
        d = torch.diag(K)
        off = K[~torch.eye(P, dtype=torch.bool, device=device)]
        dm = float(d.mean().item())
        ov = float(off.var(unbiased=False).item()) if off.numel() else 0.0
        diag_mass.append(dm)
        offdiag_var.append(ov)
        ratio.append(dm / (ov + 1e-12))

    return {
        "diag_mass": np.asarray(diag_mass),
        "offdiag_var": np.asarray(offdiag_var),
        "diag_to_offdiag_ratio": np.asarray(ratio),
    }


def run_ch02(
    N: int = 500,
    beta: float = 0.5,
    n_p: int = 18,
    n_lam: int = 16,
    n_trials: int = 12,
    n_seeds: int = 5,
    gram_N: int = 1000,
    p_min: float = 0.02,
    p_max: float = 0.8,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    topology_cache: dict = {}

    p_vals = np.linspace(p_min, p_max, n_p)
    lam_vals = np.logspace(np.log10(0.2), np.log10(6.0), n_lam)
    pc_grid = _capacity_grid(N, beta, p_vals, lam_vals, n_trials=n_trials, n_seeds=n_seeds, device=device, topology_cache=topology_cache)

    lambda_pick_idx = np.array([0, n_lam // 2, n_lam - 1], dtype=int)
    pc_lines = pc_grid[lambda_pick_idx]
    connectivity_cutoff = np.log(N) / N
    lam_opt_curve = np.array([optimal_lambda(float(p), beta) for p in p_vals])

    sn_pairs = [(0.2, 1.0), (0.35, 2.0), (0.55, 4.0)]
    sn_p_grid = np.unique(np.round(np.linspace(20, min(0.65 * N, 360), 16)).astype(int))
    sn_data = _measure_signal_noise(N, beta, sn_pairs, sn_p_grid, n_seeds=n_seeds, device=device)

    gram_p_grid = np.unique(np.round(np.linspace(16, min(0.65 * gram_N, 280), 16)).astype(int))
    gram_data = _mixed_feature_gram(gram_N, p=0.35, beta=beta, lam=2.0, p_grid=gram_p_grid, seed=0, device=device)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "beta": np.array([beta], dtype=float),
        "p_vals": p_vals,
        "lam_vals": lam_vals,
        "pc_grid": pc_grid,
        "pc_lines": pc_lines,
        "lambda_pick_idx": lambda_pick_idx,
        "connectivity_cutoff": np.array([connectivity_cutoff], dtype=float),
        "lam_opt_curve": lam_opt_curve,
        "sn_pairs_p": np.asarray([pp for pp, _ in sn_pairs], dtype=float),
        "sn_pairs_lam": np.asarray([ll for _, ll in sn_pairs], dtype=float),
        "sn_p_grid": sn_p_grid.astype(np.int32),
        "sn_signal_emp": sn_data["signal_emp"],
        "sn_tau2_emp": sn_data["tau2_emp"],
        "sn_signal_theory": sn_data["signal_theory"],
        "sn_tau2_theory": sn_data["tau2_theory"],
        "gram_N": np.array([gram_N], dtype=np.int32),
        "gram_p_grid": gram_p_grid.astype(np.int32),
        "gram_diag_mass": gram_data["diag_mass"],
        "gram_offdiag_var": gram_data["offdiag_var"],
        "gram_ratio": gram_data["diag_to_offdiag_ratio"],
    }

    save_npz(RESULT_DIR / "ch02_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch02()
