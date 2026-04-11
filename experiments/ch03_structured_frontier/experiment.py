from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from typing import Dict

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
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success
from mixed_order.theory import compute_q_from_budget, optimal_lambda_structured, replica_capacity_structured
from mixed_order.utils import make_generator

RESULT_DIR = Path(__file__).resolve().parent


def _alpha_star_from_pc(pc: float, N: int) -> float:
    return float(pc) / float(N)


def run_ch03_structured_frontier(
    N: int = 200,
    p: float = 0.35,
    beta: float = 0.5,
    gamma_vals: np.ndarray | None = None,
    alpha_vals: np.ndarray | None = None,
    n_trials: int = 4,
    n_seeds: int = 4,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    if gamma_vals is None:
        gamma_vals = np.array([0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0], dtype=float)
    if alpha_vals is None:
        alpha_vals = np.linspace(0.03, 0.5, 16)

    q, q_tilde = compute_q_from_budget(p, N, beta)
    del q

    g2_vals = np.zeros(gamma_vals.size, dtype=float)
    lam_struct_vals = np.zeros_like(g2_vals)
    alpha_pair = np.zeros_like(g2_vals)
    alpha_mix = np.zeros_like(g2_vals)
    alpha_pair_theory = np.zeros_like(g2_vals)
    alpha_mix_theory = np.zeros_like(g2_vals)

    topology_cache: dict = {}
    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()

    for gidx, gamma in enumerate(gamma_vals.tolist()):
        D = max(2, int(round(gamma * N)))
        F = generate_factors(N, D, device=device, generator=make_generator(1001 + gidx, device))
        C = compute_covariance(F)
        g2 = float(estimate_g2(C))
        g2_vals[gidx] = g2

        lam_struct = float(optimal_lambda_structured(p, beta, g2))
        lam_struct_vals[gidx] = lam_struct

        pc_pair = find_empirical_pc_by_success(
            N=N,
            p=p,
            beta=beta,
            lam=0.0,
            n_trials=n_trials,
            n_seeds=n_seeds,
            success_threshold=0.95,
            overlap_threshold=0.95,
            noise_level=0.1,
            device=device,
            F=F,
            C=C,
            centered=False,
            topology_cache=topology_cache,
        )
        pc_mix = find_empirical_pc_by_success(
            N=N,
            p=p,
            beta=beta,
            lam=lam_struct,
            n_trials=n_trials,
            n_seeds=n_seeds,
            success_threshold=0.95,
            overlap_threshold=0.95,
            noise_level=0.1,
            device=device,
            F=F,
            C=C,
            centered=False,
            topology_cache=topology_cache,
        )
        alpha_pair[gidx] = _alpha_star_from_pc(pc_pair, N)
        alpha_mix[gidx] = _alpha_star_from_pc(pc_mix, N)

        alpha_pair_theory[gidx] = float(replica_capacity_structured(p, N, beta=beta, g2=g2, lam=0.0) / N)
        alpha_mix_theory[gidx] = float(
            replica_capacity_structured(p, N, beta=beta, g2=g2, lam=lam_struct) / N
        )

    g2_grid = np.logspace(np.log10(g2_vals.min() * 0.9), np.log10(g2_vals.max() * 1.1), 200)
    lam_grid = optimal_lambda_structured(p, beta, g2_grid)
    alpha_pair_grid = replica_capacity_structured(p, N, beta=beta, g2=g2_grid, lam=0.0) / N
    alpha_mix_grid = replica_capacity_structured(p, N, beta=beta, g2=g2_grid, lam=lam_grid) / N

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "gamma_vals": gamma_vals,
        "g2_vals": g2_vals,
        "lam_struct_vals": lam_struct_vals,
        "alpha_pair": alpha_pair,
        "alpha_mix": alpha_mix,
        "alpha_pair_theory": alpha_pair_theory,
        "alpha_mix_theory": alpha_mix_theory,
        "alpha_gain": alpha_mix - alpha_pair,
        "alpha_gain_theory": alpha_mix_theory - alpha_pair_theory,
        "g2_grid": g2_grid,
        "lam_grid": lam_grid,
        "alpha_pair_grid": alpha_pair_grid,
        "alpha_mix_grid": alpha_mix_grid,
    }

    save_npz(RESULT_DIR / "ch03_structured_frontier_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03_structured_frontier()

