from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

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

from experiments.common import ensure_dir, save_npz
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors, generate_structured_patterns
from mixed_order.metrics import _DEVICE, run_batched_retrieval_many_seeds, seed_all
from mixed_order.theory import compute_q_from_budget
from mixed_order.utils import make_generator
from mixed_order_model import MixedOrderHopfieldNetwork

RESULT_DIR = Path(__file__).resolve().parent


def _success_curve(
    N: int,
    p: float,
    beta: float,
    lam: float,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    n_trials: int,
    n_seeds: int,
    device: str,
    centered: bool,
    topology_cache: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
    ov = run_batched_retrieval_many_seeds(
        seed_batch=list(range(n_seeds)),
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        lam=float(lam),
        device=device,
        F=F,
        centered=centered,
        noise_level=0.1,
        topology_cache=topology_cache,
    )
    success = (ov >= 0.95).to(dtype=torch.float32).mean(dim=(0, 2)).cpu().numpy()
    overlap = ov.mean(dim=(0, 2)).cpu().numpy()
    return success, overlap


def _alpha_star(alpha_vals: np.ndarray, success: np.ndarray, threshold: float = 0.95) -> float:
    idx = np.where(success < threshold)[0]
    if idx.size == 0:
        return float(alpha_vals[-1])
    if idx[0] == 0:
        return float(alpha_vals[0])
    j = int(idx[0])
    x0, x1 = float(alpha_vals[j - 1]), float(alpha_vals[j])
    y0, y1 = float(success[j - 1]), float(success[j])
    frac = (threshold - y0) / (y1 - y0 + 1e-12)
    return float(x0 + frac * (x1 - x0))


def _field_statistics(
    N: int,
    p: float,
    beta: float,
    lam: float,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    C: torch.Tensor,
    device: str,
    centered: bool,
    seed_offset: int,
) -> Dict[str, np.ndarray]:
    signal = np.zeros(alpha_vals.size, dtype=float)
    drift = np.zeros_like(signal)

    C_cpu = C.detach().cpu().numpy()
    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
    for idx, P in enumerate(p_list):
        # Keep topology and patterns fixed for the condition so the comparison is clean.
        model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=float(lam), device=device)
        model.generate_masks(generator=make_generator(seed_offset, device))
        c_mask = model.c.detach().cpu().numpy()

        patterns = generate_structured_patterns(
            P,
            F,
            device=device,
            generator=make_generator(8000 + seed_offset + idx, device),
        ).to(torch.float32)
        if centered:
            model.store_multiple_p(patterns, [P], centered=True, C=C)
        else:
            model.store_multiple_p(patterns, [P], centered=False)

        mu_idx = torch.arange(min(4, P), device=device) % P
        targets = patterns[mu_idx]
        h_full = model.local_field(targets.unsqueeze(0)).squeeze(0)
        signal[idx] = float((h_full * targets).mean().item())

        if centered:
            drift[idx] = 0.0
        else:
            C_masked = C_cpu * c_mask
            C_masked[np.diag_indices(N)] = 0.0
            drift_field = (P / (p * N)) * torch.matmul(targets, torch.from_numpy(C_masked).to(device=device, dtype=torch.float32).T)
            drift[idx] = float(torch.abs(drift_field).mean().item())

    return {
        "signal": signal,
        "drift": drift,
        "margin": signal - drift,
    }


def run_ch04_ablation_matrix(
    N: int = 96,
    p: float = 0.35,
    beta: float = 0.5,
    D: int = 4,
    factor_seed: int = 1001,
    n_trials: int = 4,
    n_seeds: int = 3,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE

    F = generate_factors(N, D, device=device, generator=make_generator(factor_seed, device))
    C = compute_covariance(F)
    g2 = estimate_g2(C)
    _, q_tilde = compute_q_from_budget(p, N, beta)
    lam_mix = float(q_tilde * (1.0 / p + g2))
    lam_pair = 0.0

    alpha_vals = np.linspace(0.02, 0.16, 11)
    topology_cache: dict = {}

    success_pair_unc, overlap_pair_unc = _success_curve(
        N, p, beta, lam_pair, alpha_vals, F, n_trials, n_seeds, device, centered=False, topology_cache=topology_cache
    )
    success_pair_cen, overlap_pair_cen = _success_curve(
        N, p, beta, lam_pair, alpha_vals, F, n_trials, n_seeds, device, centered=True, topology_cache=topology_cache
    )
    success_mix_unc, overlap_mix_unc = _success_curve(
        N, p, beta, lam_mix, alpha_vals, F, n_trials, n_seeds, device, centered=False, topology_cache=topology_cache
    )
    success_mix_cen, overlap_mix_cen = _success_curve(
        N, p, beta, lam_mix, alpha_vals, F, n_trials, n_seeds, device, centered=True, topology_cache=topology_cache
    )

    stat_pair_unc = _field_statistics(N, p, beta, lam_pair, alpha_vals, F, C, device, centered=False, seed_offset=110)
    stat_pair_cen = _field_statistics(N, p, beta, lam_pair, alpha_vals, F, C, device, centered=True, seed_offset=110)
    stat_mix_unc = _field_statistics(N, p, beta, lam_mix, alpha_vals, F, C, device, centered=False, seed_offset=220)
    stat_mix_cen = _field_statistics(N, p, beta, lam_mix, alpha_vals, F, C, device, centered=True, seed_offset=220)

    alpha_star_pair_unc = _alpha_star(alpha_vals, success_pair_unc)
    alpha_star_pair_cen = _alpha_star(alpha_vals, success_pair_cen)
    alpha_star_mix_unc = _alpha_star(alpha_vals, success_mix_unc)
    alpha_star_mix_cen = _alpha_star(alpha_vals, success_mix_cen)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "D": np.array([D], dtype=np.int32),
        "factor_seed": np.array([factor_seed], dtype=np.int32),
        "g2": np.array([g2], dtype=float),
        "lam_pair": np.array([lam_pair], dtype=float),
        "lam_mix": np.array([lam_mix], dtype=float),
        "alpha_vals": alpha_vals,
        "success_pair_uncentered": success_pair_unc,
        "success_pair_centered": success_pair_cen,
        "success_mix_uncentered": success_mix_unc,
        "success_mix_centered": success_mix_cen,
        "overlap_pair_uncentered": overlap_pair_unc,
        "overlap_pair_centered": overlap_pair_cen,
        "overlap_mix_uncentered": overlap_mix_unc,
        "overlap_mix_centered": overlap_mix_cen,
        "signal_pair_uncentered": stat_pair_unc["signal"],
        "signal_pair_centered": stat_pair_cen["signal"],
        "signal_mix_uncentered": stat_mix_unc["signal"],
        "signal_mix_centered": stat_mix_cen["signal"],
        "drift_pair_uncentered": stat_pair_unc["drift"],
        "drift_pair_centered": stat_pair_cen["drift"],
        "drift_mix_uncentered": stat_mix_unc["drift"],
        "drift_mix_centered": stat_mix_cen["drift"],
        "margin_pair_uncentered": stat_pair_unc["margin"],
        "margin_pair_centered": stat_pair_cen["margin"],
        "margin_mix_uncentered": stat_mix_unc["margin"],
        "margin_mix_centered": stat_mix_cen["margin"],
        "alpha_star_pair_uncentered": np.array([alpha_star_pair_unc], dtype=float),
        "alpha_star_pair_centered": np.array([alpha_star_pair_cen], dtype=float),
        "alpha_star_mix_uncentered": np.array([alpha_star_mix_unc], dtype=float),
        "alpha_star_mix_centered": np.array([alpha_star_mix_cen], dtype=float),
    }

    save_npz(RESULT_DIR / "ch04_ablation_matrix_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch04_ablation_matrix()
