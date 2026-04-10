from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixed_order.plotting.style import apply_pub_style, get_color_palette


Mode = Literal["pairwise", "cubic", "mixed"]


@dataclass(frozen=True)
class HamSuiteConfig:
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps: int = 8
    beta: float = 8.0
    noise_levels: Tuple[float, ...] = tuple(np.linspace(0.0, 0.30, 7))
    basin_noise_levels: Tuple[float, ...] = tuple(np.linspace(0.0, 0.45, 10))
    report_path: Path = Path(__file__).with_name("ham_report.png")
    results_path: Path = Path(__file__).with_name("ham_suite_results.npz")


def seed_all(seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed) & 0xFFFF_FFFF)


def make_generator(seed: int, device: str) -> torch.Generator:
    gen = torch.Generator(device=device if device == "cuda" else "cpu")
    gen.manual_seed(int(seed) & 0xFFFF_FFFF)
    return gen


def sign01(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def sample_rademacher(shape: Sequence[int], generator: torch.Generator, device: str) -> torch.Tensor:
    return torch.randint(0, 2, tuple(shape), generator=generator, device=device).float() * 2.0 - 1.0


def make_disjoint_triples(n_triples: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri_i = torch.arange(0, 3 * n_triples, 3, device=device)
    tri_j = tri_i + 1
    tri_k = tri_i + 2
    return tri_i, tri_j, tri_k


def make_iid_patterns(n_patterns: int, n_bits: int, seed: int, device: str) -> torch.Tensor:
    gen = make_generator(seed, device)
    return sample_rademacher((n_patterns, n_bits), gen, device)


def make_structured_bank(
    n_patterns: int,
    n_proto: int,
    n_triples: int,
    seed: int,
    device: str,
    shared_template: bool = True,
    group_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = make_generator(seed, device)
    tri_i, tri_j, tri_k = make_disjoint_triples(n_triples, device)
    total_bits = int(n_proto + 3 * n_triples)
    patterns = torch.empty((n_patterns, total_bits), device=device)

    proto = sample_rademacher((n_proto,), gen, device) if n_proto > 0 else None
    triple_base = sample_rademacher((n_triples, 2), gen, device)
    parity = torch.empty((n_patterns, n_triples), device=device)

    if shared_template and group_size > 1:
        n_groups = int(np.ceil(n_triples / group_size))
        group_bits = sample_rademacher((n_patterns, n_groups), gen, device)
        for g in range(n_groups):
            start = g * group_size
            stop = min(n_triples, start + group_size)
            parity[:, start:stop] = group_bits[:, g : g + 1].expand(-1, stop - start)
    else:
        parity = sample_rademacher((n_patterns, n_triples), gen, device)

    for mu in tqdm(range(n_patterns), desc="build patterns", leave=False):
        if n_proto > 0:
            patterns[mu, :n_proto] = proto
        for t in range(n_triples):
            b1, b2 = triple_base[t]
            b3 = parity[mu, t] * b1 * b2
            start = n_proto + 3 * t
            patterns[mu, start : start + 3] = torch.stack([b1, b2, b3])

    return patterns, parity, tri_i + n_proto, tri_j + n_proto, tri_k + n_proto


def make_hybrid_bank(
    n_patterns: int,
    n_proto: int,
    n_triples: int,
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = make_generator(seed, device)
    tri_i, tri_j, tri_k = make_disjoint_triples(n_triples, device)
    total_bits = int(n_proto + 3 * n_triples)

    patterns = torch.empty((n_patterns, total_bits), device=device)
    proto = sample_rademacher((n_patterns, n_proto), gen, device)
    parity = sample_rademacher((n_patterns, n_triples), gen, device)
    triple_base = sample_rademacher((n_patterns, n_triples, 2), gen, device)

    for mu in tqdm(range(n_patterns), desc="build hybrid patterns", leave=False):
        patterns[mu, :n_proto] = proto[mu]
        for t in range(n_triples):
            b1, b2 = triple_base[mu, t]
            b3 = parity[mu, t] * b1 * b2
            start = n_proto + 3 * t
            patterns[mu, start : start + 3] = torch.stack([b1, b2, b3])

    return patterns, parity, tri_i + n_proto, tri_j + n_proto, tri_k + n_proto


def make_grouped_hybrid_bank(
    n_groups: int,
    variants_per_group: int,
    n_proto: int,
    n_triples: int,
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = make_generator(seed, device)
    tri_i, tri_j, tri_k = make_disjoint_triples(n_triples, device)
    n_patterns = n_groups * variants_per_group
    total_bits = int(n_proto + 3 * n_triples)

    group_proto = sample_rademacher((n_groups, n_proto), gen, device)
    variant_parity = sample_rademacher((variants_per_group, n_triples), gen, device)
    triple_base = sample_rademacher((variants_per_group, n_triples, 2), gen, device)

    patterns = torch.empty((n_patterns, total_bits), device=device)
    group_ids = torch.empty((n_patterns,), dtype=torch.long, device=device)
    variant_ids = torch.empty((n_patterns,), dtype=torch.long, device=device)

    idx = 0
    for g in tqdm(range(n_groups), desc="groups", leave=False):
        for v in range(variants_per_group):
            patterns[idx, :n_proto] = group_proto[g]
            for t in range(n_triples):
                b1, b2 = triple_base[v, t]
                b3 = variant_parity[v, t] * b1 * b2
                start = n_proto + 3 * t
                patterns[idx, start : start + 3] = torch.stack([b1, b2, b3])
            group_ids[idx] = g
            variant_ids[idx] = v
            idx += 1

    return patterns, group_ids, variant_ids, tri_i + n_proto, tri_j + n_proto, tri_k + n_proto, group_proto


def make_antipodal_pair(
    n_proto: int,
    n_triples: int,
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bank, parity, tri_i, tri_j, tri_k = make_structured_bank(
        n_patterns=1,
        n_proto=n_proto,
        n_triples=n_triples,
        seed=seed,
        device=device,
        shared_template=False,
        group_size=1,
    )
    a = bank[0]
    b = -a
    patterns = torch.stack([a, b], dim=0)
    return patterns, parity[:1], tri_i, tri_j, tri_k


def corrupt_bits(x: torch.Tensor, noise: float, generator: torch.Generator) -> torch.Tensor:
    if noise <= 0:
        return x.clone()
    y = x.clone()
    mask = torch.rand(y.shape, generator=generator, device=y.device) < noise
    y[mask] = -y[mask]
    return y


def corrupt_triples(x: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor, tri_k: torch.Tensor, noise: float, generator: torch.Generator) -> torch.Tensor:
    if noise <= 0:
        return x.clone()
    y = x.clone()
    tri_mask = torch.rand(tri_i.numel(), generator=generator, device=x.device) < noise
    idx = torch.nonzero(tri_mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return y
    y[tri_i[idx]] *= -1
    y[tri_j[idx]] *= -1
    y[tri_k[idx]] *= -1
    return y


def pairwise_scores(x: torch.Tensor, patterns: torch.Tensor, proto_dim: Optional[int] = None) -> torch.Tensor:
    if proto_dim is not None:
        x = x[..., :proto_dim]
        patterns = patterns[:, :proto_dim]
    overlaps = (x @ patterns.T) / x.shape[-1]
    return overlaps.square()


def triple_signatures(x: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor, tri_k: torch.Tensor) -> torch.Tensor:
    return x[:, tri_i] * x[:, tri_j] * x[:, tri_k]


def cubic_scores(x: torch.Tensor, patterns: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor, tri_k: torch.Tensor) -> torch.Tensor:
    x_sig = triple_signatures(x, tri_i, tri_j, tri_k)
    p_sig = triple_signatures(patterns, tri_i, tri_j, tri_k)
    return (x_sig @ p_sig.T) / max(1, tri_i.numel())


def mixed_scores(
    x: torch.Tensor,
    patterns: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    tri_k: torch.Tensor,
    proto_dim: Optional[int] = None,
    cubic_weight: float = 1.0,
) -> torch.Tensor:
    return pairwise_scores(x, patterns, proto_dim=proto_dim) + cubic_weight * cubic_scores(x, patterns, tri_i, tri_j, tri_k)


def score_model(
    x: torch.Tensor,
    patterns: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    tri_k: torch.Tensor,
    mode: Mode,
    proto_dim: Optional[int] = None,
    cubic_weight: float = 1.0,
) -> torch.Tensor:
    if mode == "pairwise":
        return pairwise_scores(x, patterns, proto_dim=proto_dim)
    if mode == "cubic":
        return cubic_scores(x, patterns, tri_i, tri_j, tri_k)
    return mixed_scores(x, patterns, tri_i, tri_j, tri_k, proto_dim=proto_dim, cubic_weight=cubic_weight)


def retrieve_state(
    x0: torch.Tensor,
    patterns: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    tri_k: torch.Tensor,
    mode: Mode,
    beta: float,
    max_steps: int,
    proto_dim: Optional[int] = None,
    cubic_weight: float = 1.0,
) -> torch.Tensor:
    state = x0.clone()
    for _ in range(max_steps):
        scores = score_model(state.unsqueeze(0), patterns, tri_i, tri_j, tri_k, mode, proto_dim=proto_dim, cubic_weight=cubic_weight)
        weights = torch.softmax(beta * scores, dim=-1)
        state_new = weights @ patterns
        state_new = sign01(state_new.squeeze(0))
        if torch.equal(state_new, state):
            break
        state = state_new
    return state


def classify_queries(
    queries: torch.Tensor,
    patterns: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    tri_k: torch.Tensor,
    mode: Mode,
    proto_dim: Optional[int] = None,
    cubic_weight: float = 1.0,
) -> torch.Tensor:
    scores = score_model(queries, patterns, tri_i, tri_j, tri_k, mode, proto_dim=proto_dim, cubic_weight=cubic_weight)
    return scores.argmax(dim=-1)


def task3_score_separation(cfg: HamSuiteConfig) -> Dict[str, np.ndarray]:
    patterns, _, tri_i, tri_j, tri_k = make_antipodal_pair(n_proto=1, n_triples=21, seed=cfg.seed, device=cfg.device)
    anchor = patterns[0]
    probe = 0.15 * anchor
    probe = probe.unsqueeze(0)

    pair = pairwise_scores(probe, patterns).squeeze(0).detach().cpu().numpy()
    cub = cubic_scores(probe, patterns, tri_i, tri_j, tri_k).squeeze(0).detach().cpu().numpy()
    mix = pair + cub

    return {
        "patterns": patterns.detach().cpu().numpy(),
        "pairwise": pair,
        "cubic": cub,
        "mixed": mix,
        "probe": probe.squeeze(0).detach().cpu().numpy(),
    }


def task1_parity_retrieval(cfg: HamSuiteConfig) -> Dict[str, np.ndarray]:
    n_groups = 4
    variants_per_group = 4
    n_patterns = n_groups * variants_per_group
    n_proto = 8
    n_triples = 8
    n_trials = 128
    patterns, group_ids, variant_ids, tri_i, tri_j, tri_k, _ = make_grouped_hybrid_bank(
        n_groups=n_groups,
        variants_per_group=variants_per_group,
        n_proto=n_proto,
        n_triples=n_triples,
        seed=cfg.seed,
        device=cfg.device,
    )
    gen = make_generator(cfg.seed + 101, cfg.device)
    noise_levels = torch.tensor(cfg.noise_levels, device=cfg.device)
    mixed_weight = 1.0

    acc = {mode: [] for mode in ("pairwise", "cubic", "mixed")}
    overlap = {mode: [] for mode in ("pairwise", "cubic", "mixed")}

    for noise in tqdm(noise_levels.tolist(), desc="noise levels", leave=False):
        mu_idx = torch.randint(0, n_patterns, (n_trials,), generator=gen, device=cfg.device)
        cue = torch.stack([corrupt_bits(patterns[int(mu)], noise, gen) for mu in mu_idx.tolist()], dim=0)
        target = patterns[mu_idx]
        mode_acc = {m: [] for m in acc}
        mode_ov = {m: [] for m in overlap}
        for mode in acc:
            pred = classify_queries(cue, patterns, tri_i, tri_j, tri_k, mode, proto_dim=n_proto, cubic_weight=mixed_weight)
            mode_acc[mode].append((pred == mu_idx).to(dtype=torch.float32).mean().item())
            final = retrieve_state(
                cue,
                patterns,
                tri_i,
                tri_j,
                tri_k,
                mode,
                beta=cfg.beta,
                max_steps=cfg.max_steps,
                proto_dim=n_proto,
                cubic_weight=mixed_weight,
            )
            mode_ov[mode].append(((final * target).sum(dim=1) / target.shape[-1]).mean().item())
        for mode in acc:
            acc[mode].append(float(np.mean(mode_acc[mode])))
            overlap[mode].append(float(np.mean(mode_ov[mode])))

    return {
        "noise_levels": np.array(cfg.noise_levels, dtype=np.float32),
        "accuracy": {k: np.array(v, dtype=np.float32) for k, v in acc.items()},
        "overlap": {k: np.array(v, dtype=np.float32) for k, v in overlap.items()},
        "patterns": patterns.detach().cpu().numpy(),
        "group_ids": group_ids.detach().cpu().numpy(),
        "variant_ids": variant_ids.detach().cpu().numpy(),
    }


def task2_capacity_curve(cfg: HamSuiteConfig) -> Dict[str, np.ndarray]:
    n_bits = 100
    n_triples = n_bits // 3
    n_proto = n_bits - 3 * n_triples
    alpha_grid = np.linspace(0.05, 0.5, 10)
    P_grid = np.unique(np.maximum(4, np.round(alpha_grid * (n_bits**2)).astype(int)))
    n_trials = 64

    out: Dict[str, List[float]] = {
        "iid_pairwise": [],
        "iid_mixed": [],
        "structured_pairwise": [],
        "structured_mixed": [],
    }

    for P in tqdm(P_grid.tolist(), desc="P sweep", leave=False):
        n_classes = int(P)
        seed_base = cfg.seed + P

        iid_patterns = make_iid_patterns(n_classes, n_bits, seed_base, cfg.device)
        iid_tri_i, iid_tri_j, iid_tri_k = make_disjoint_triples(n_triples, cfg.device)
        structured_patterns, _, tri_i, tri_j, tri_k = make_structured_bank(
            n_patterns=n_classes,
            n_proto=n_proto,
            n_triples=n_triples,
            seed=seed_base,
            device=cfg.device,
            shared_template=True,
            group_size=4,
        )

        gen = make_generator(seed_base + 17, cfg.device)
        for mode in ("pairwise", "mixed"):
            mu_iid = torch.randint(0, n_classes, (n_trials,), generator=gen, device=cfg.device)
            mu_struct = torch.randint(0, n_classes, (n_trials,), generator=gen, device=cfg.device)
            cue_iid = torch.stack([corrupt_bits(iid_patterns[int(mu)], 0.2, gen) for mu in mu_iid.tolist()], dim=0)
            cue_struct = torch.stack([corrupt_bits(structured_patterns[int(mu)], 0.2, gen) for mu in mu_struct.tolist()], dim=0)
            pred_iid = classify_queries(cue_iid, iid_patterns, iid_tri_i, iid_tri_j, iid_tri_k, mode)
            pred_struct = classify_queries(cue_struct, structured_patterns, tri_i, tri_j, tri_k, mode)
            out[f"iid_{mode}"].append((pred_iid == mu_iid).to(dtype=torch.float32).mean().item())
            out[f"structured_{mode}"].append((pred_struct == mu_struct).to(dtype=torch.float32).mean().item())

    return {
        "P_grid": P_grid.astype(np.int32),
        "alpha_grid": (P_grid / float(n_bits**2)).astype(np.float32),
        **{k: np.array(v, dtype=np.float32) for k, v in out.items()},
    }


def task4_spurious_census(cfg: HamSuiteConfig) -> Dict[str, np.ndarray]:
    n_bits = 50
    n_triples = n_bits // 3
    n_proto = n_bits - 3 * n_triples
    k_grid = np.array([4, 8, 12, 16], dtype=np.int32)
    n_inits = 256
    beta = 10.0
    max_steps = 8

    final_counts: Dict[str, List[float]] = {"pairwise": [], "mixed": []}
    stored_hits: Dict[str, List[float]] = {"pairwise": [], "mixed": []}

    for k in tqdm(k_grid.tolist(), desc="k sweep", leave=False):
        patterns, _, tri_i, tri_j, tri_k = make_structured_bank(
            n_patterns=k,
            n_proto=n_proto,
            n_triples=n_triples,
            seed=cfg.seed + k,
            device=cfg.device,
            shared_template=True,
            group_size=2,
        )
        gen = make_generator(cfg.seed + 1000 + k, cfg.device)
        for mode in ("pairwise", "mixed"):
            x0 = sample_rademacher((n_inits, n_bits), gen, cfg.device)
            finals_tensor = retrieve_state(x0, patterns, tri_i, tri_j, tri_k, mode, beta=beta, max_steps=max_steps)
            overlaps = (patterns @ finals_tensor.T) / n_bits
            stored = (overlaps.abs().amax(dim=1) >= 0.95).to(dtype=torch.float32).sum().item()
            uniq = torch.unique(finals_tensor, dim=0).shape[0]
            spurious = max(0, int(uniq - patterns.shape[0]))
            final_counts[mode].append(float(spurious))
            stored_hits[mode].append(float(stored / n_inits))

    return {
        "k_grid": k_grid,
        "pairwise_spurious": np.array(final_counts["pairwise"], dtype=np.float32),
        "mixed_spurious": np.array(final_counts["mixed"], dtype=np.float32),
        "pairwise_stored_rate": np.array(stored_hits["pairwise"], dtype=np.float32),
        "mixed_stored_rate": np.array(stored_hits["mixed"], dtype=np.float32),
    }


def task5_triple_basin(cfg: HamSuiteConfig) -> Dict[str, np.ndarray]:
    n_groups = 4
    variants_per_group = 4
    n_patterns = n_groups * variants_per_group
    n_proto = 8
    n_triples = 8
    patterns, _, _, tri_i, tri_j, tri_k, _ = make_grouped_hybrid_bank(
        n_groups=n_groups,
        variants_per_group=variants_per_group,
        n_proto=n_proto,
        n_triples=n_triples,
        seed=cfg.seed + 404,
        device=cfg.device,
    )
    gen = make_generator(cfg.seed + 405, cfg.device)
    noise_grid = torch.tensor(cfg.basin_noise_levels, device=cfg.device)
    n_trials = 96
    mixed_weight = 1.5

    out: Dict[str, List[float]] = {
        "pairwise_bitflip": [],
        "mixed_bitflip": [],
        "pairwise_tripleflip": [],
        "mixed_tripleflip": [],
    }

    for noise in tqdm(noise_grid.tolist(), desc="basin noise", leave=False):
        for mode in ("pairwise", "mixed"):
            mu = torch.randint(0, n_patterns, (n_trials,), generator=gen, device=cfg.device)
            cues = patterns[mu]
            x_bit = torch.stack([corrupt_bits(cue, noise, gen) for cue in cues], dim=0)
            x_tri = torch.stack([corrupt_triples(cue, tri_i, tri_j, tri_k, noise, gen) for cue in cues], dim=0)
            y_bit = retrieve_state(
                x_bit,
                patterns,
                tri_i,
                tri_j,
                tri_k,
                mode,
                beta=cfg.beta,
                max_steps=cfg.max_steps,
                proto_dim=n_proto,
                cubic_weight=mixed_weight,
            )
            y_tri = retrieve_state(
                x_tri,
                patterns,
                tri_i,
                tri_j,
                tri_k,
                mode,
                beta=cfg.beta,
                max_steps=cfg.max_steps,
                proto_dim=n_proto,
                cubic_weight=mixed_weight,
            )
            out[f"{mode}_bitflip"].append(((y_bit * patterns[mu]).mean(dim=1) > 0.9).to(dtype=torch.float32).mean().item())
            out[f"{mode}_tripleflip"].append(((y_tri * patterns[mu]).mean(dim=1) > 0.9).to(dtype=torch.float32).mean().item())

    return {
        "noise_levels": np.array(cfg.basin_noise_levels, dtype=np.float32),
        **{k: np.array(v, dtype=np.float32) for k, v in out.items()},
    }


def plot_report(task3: Dict[str, np.ndarray], task1: Dict[str, np.ndarray], task4: Dict[str, np.ndarray], task2: Dict[str, np.ndarray], out_path: Path) -> None:
    apply_pub_style()
    cp = get_color_palette()
    out_path = Path(out_path)
    out_dir = out_path.parent
    base = out_path.stem

    # Task 3: antipodal null-space (bar plot)
    fig_t3, ax_t3 = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    width = 0.23
    labels = ["A", "B"]
    ax_t3.bar(x - width, task3["pairwise"], width, label="pairwise", color=cp["uncentered"])
    ax_t3.bar(x, task3["cubic"], width, label="cubic", color=cp["centered"])
    ax_t3.bar(x + width, task3["mixed"], width, label="mixed", color=cp["mixed"])
    ax_t3.set_xticks(x)
    ax_t3.set_xticklabels(labels)
    ax_t3.set_title("Task 3: antipodal null-space")
    ax_t3.set_ylabel("score")
    ax_t3.legend(frameon=False)
    ax_t3.text(0.02, 0.02, "pairwise ties\ncubic flips sign", transform=ax_t3.transAxes, fontsize=10, va="bottom")
    path_t3 = out_dir / f"{base}_task3.png"
    fig_t3.tight_layout()
    fig_t3.savefig(path_t3)
    plt.close(fig_t3)

    # Task 1: parity retrieval (line plot)
    fig_t1, ax_t1 = plt.subplots(figsize=(6.5, 4.5))
    colors = {"pairwise": cp["uncentered"], "cubic": cp["centered"], "mixed": cp["mixed"]}
    for mode in ("pairwise", "cubic", "mixed"):
        ax_t1.plot(task1["noise_levels"], task1["accuracy"][mode], "o-", color=colors[mode], label=mode)
    ax_t1.set_title("Task 1: parity retrieval")
    ax_t1.set_xlabel("bit-flip noise")
    ax_t1.set_ylabel("accuracy")
    ax_t1.set_ylim(0.0, 1.02)
    ax_t1.legend(frameon=False)
    path_t1 = out_dir / f"{base}_task1.png"
    fig_t1.tight_layout()
    fig_t1.savefig(path_t1)
    plt.close(fig_t1)

    # Task 4: spurious attractor census
    fig_t4, ax_t4 = plt.subplots(figsize=(6.5, 4.5))
    ax_t4.plot(task4["k_grid"], task4["pairwise_spurious"], "o-", color=cp["uncentered"], label="pairwise")
    ax_t4.plot(task4["k_grid"], task4["mixed_spurious"], "o-", color=cp["mixed"], label="mixed")
    ax_t4.set_title("Task 4: spurious attractor census")
    ax_t4.set_xlabel("stored patterns K")
    ax_t4.set_ylabel("unique non-stored attractors")
    ax_t4.legend(frameon=False)
    path_t4 = out_dir / f"{base}_task4.png"
    fig_t4.tight_layout()
    fig_t4.savefig(path_t4)
    plt.close(fig_t4)

    # Task 2: load curve at fixed budget
    fig_t2, ax_t2 = plt.subplots(figsize=(6.5, 4.5))
    ax_t2.plot(task2["alpha_grid"], task2["iid_pairwise"], "o--", color=cp["uncentered"], label="iid pairwise")
    ax_t2.plot(task2["alpha_grid"], task2["iid_mixed"], "o-", color=cp["mixed"], label="iid mixed")
    ax_t2.plot(task2["alpha_grid"], task2["structured_pairwise"], "s--", color=cp["extra"], label="structured pairwise")
    ax_t2.plot(task2["alpha_grid"], task2["structured_mixed"], "s-", color=cp["centered"], label="structured mixed")
    ax_t2.set_title("Task 2: load curve at fixed budget")
    ax_t2.set_xlabel(r"$\alpha = P / N^2$")
    ax_t2.set_ylabel("classification accuracy")
    ax_t2.set_ylim(0.0, 1.02)
    ax_t2.legend(frameon=False, fontsize=9)
    path_t2 = out_dir / f"{base}_task2.png"
    fig_t2.tight_layout()
    fig_t2.savefig(path_t2)
    plt.close(fig_t2)


def plot_basin(task5: Dict[str, np.ndarray], out_path: Path) -> None:
    apply_pub_style()
    cp = get_color_palette()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    x = task5["noise_levels"]
    ax.plot(x, task5["pairwise_bitflip"], "o-", color=cp["uncentered"], label="pairwise, bit flip")
    ax.plot(x, task5["mixed_bitflip"], "o-", color=cp["mixed"], label="mixed, bit flip")
    ax.plot(x, task5["pairwise_tripleflip"], "s--", color=cp["extra"], label="pairwise, triple flip")
    ax.plot(x, task5["mixed_tripleflip"], "s-", color=cp["centered"], label="mixed, triple flip")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("retrieval success")
    ax.set_title("Task 5: basin width under structured corruption")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_results(path: Path, results: Dict[str, np.ndarray]) -> None:
    arrays: Dict[str, np.ndarray] = {}

    def _flatten(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                next_prefix = f"{prefix}_{sub_key}" if prefix else str(sub_key)
                _flatten(next_prefix, sub_value)
        else:
            arrays[prefix] = np.asarray(value)

    for key, value in results.items():
        _flatten(key, value)
    np.savez(path, **arrays)


def run_suite(cfg: HamSuiteConfig, task: str = "all") -> None:
    seed_all(cfg.seed)
    task3 = task3_score_separation(cfg)
    task1 = task1_parity_retrieval(cfg)
    task2 = task2_capacity_curve(cfg)
    task4 = task4_spurious_census(cfg)
    task5 = task5_triple_basin(cfg)

    save_results(
        cfg.results_path,
        {
            "task3": task3,
            "task1": task1,
            "task2": task2,
            "task4": task4,
            "task5": task5,
        },
    )

    if task == "all":
        plot_report(task3, task1, task4, task2, cfg.report_path)
    elif task == "task5":
        plot_basin(task5, cfg.report_path)
    elif task == "task3":
        plot_report(task3, task1, task4, task2, cfg.report_path)
    else:
        plot_report(task3, task1, task4, task2, cfg.report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal HAM synthetic-task suite.")
    parser.add_argument("--task", default="all", choices=["all", "task1", "task2", "task3", "task4", "task5"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=8.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = HamSuiteConfig(
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        max_steps=args.max_steps,
        beta=args.beta,
        report_path=Path(args.report_path) if args.report_path else HamSuiteConfig.report_path,
        results_path=Path(args.results_path) if args.results_path else HamSuiteConfig.results_path,
    )
    run_suite(cfg, task=args.task)
    print(f"saved report: {cfg.report_path}")
    print(f"saved results: {cfg.results_path}")


if __name__ == "__main__":
    main()
