from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from experiments.common import ensure_dir, save_npz
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success
from mixed_order.theory import compute_q_from_budget
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _g2_sweep(N: int, gamma_vals: np.ndarray, n_seeds: int, device: str) -> Dict[str, np.ndarray]:
    g2_mean = []
    g2_std = []
    invD = []
    for gamma in tqdm(gamma_vals.tolist(), desc="g2 sweep"):
        D = max(2, int(round(gamma * N)))
        invD.append(1.0 / D)
        F = generate_factors(N, D, device=device)
        C = compute_covariance(F)
        g2 = estimate_g2(C)
        g2_mean.append(float(g2))
        g2_std.append(0.0)
    return {
        "gamma": gamma_vals,
        "invD": np.asarray(invD),
        "g2_mean": np.asarray(g2_mean),
        "g2_std": np.asarray(g2_std),
    }


def _centering_shift(
    N: int,
    p: float,
    beta: float,
    gamma_vals: np.ndarray,
    lambda_vals: np.ndarray,
    n_trials: int,
    n_seeds: int,
    device: str,
    topology_cache: dict,
) -> Dict[str, np.ndarray]:
    pc_centered = np.zeros((gamma_vals.size, lambda_vals.size), dtype=float)
    pc_uncentered = np.zeros_like(pc_centered)
    g2_vals = np.zeros(gamma_vals.size, dtype=float)

    lam_star_centered = np.zeros(gamma_vals.size, dtype=float)
    lam_star_uncentered = np.zeros(gamma_vals.size, dtype=float)

    for gidx, gamma in enumerate(tqdm(gamma_vals.tolist(), desc="gamma sweep")):
        D = max(2, int(round(gamma * N)))
        F = generate_factors(N, D, device=device)
        C = compute_covariance(F)
        g2 = estimate_g2(C)
        g2_vals[gidx] = g2

        for lidx, lam in enumerate(tqdm(lambda_vals.tolist(), desc="lambda sweep", leave=False)):
            pc_centered[gidx, lidx] = find_empirical_pc_by_success(
                N=N,
                p=p,
                beta=beta,
                lam=float(lam),
                n_trials=n_trials,
                n_seeds=n_seeds,
                success_threshold=0.9,
                overlap_threshold=0.9,
                noise_level=0.1,
                device=device,
                F=F,
                C=C,
                centered=True,
                topology_cache=topology_cache,
            )
            pc_uncentered[gidx, lidx] = find_empirical_pc_by_success(
                N=N,
                p=p,
                beta=beta,
                lam=float(lam),
                n_trials=n_trials,
                n_seeds=n_seeds,
                success_threshold=0.9,
                overlap_threshold=0.9,
                noise_level=0.1,
                device=device,
                F=F,
                centered=False,
                topology_cache=topology_cache,
            )

        lam_star_centered[gidx] = lambda_vals[int(np.argmax(pc_centered[gidx]))]
        lam_star_uncentered[gidx] = lambda_vals[int(np.argmax(pc_uncentered[gidx]))]

    delta_measured = lam_star_centered - lam_star_uncentered
    _, q_tilde = compute_q_from_budget(p, N, beta)
    delta_pred = q_tilde * g2_vals

    return {
        "gamma": gamma_vals,
        "lambda_vals": lambda_vals,
        "pc_centered": pc_centered,
        "pc_uncentered": pc_uncentered,
        "g2": g2_vals,
        "lam_star_centered": lam_star_centered,
        "lam_star_uncentered": lam_star_uncentered,
        "delta_measured": delta_measured,
        "delta_pred": delta_pred,
    }


def run_ch03(
    N: int = 1000,
    p: float = 0.35,
    beta: float = 0.5,
    n_trials: int = 12,
    n_seeds: int = 10,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    topology_cache: dict = {}

    gamma_g2 = np.array([0.2, 0.35, 0.5, 0.75, 1.25, 2.0, 3.0], dtype=float)
    g2_data = _g2_sweep(N, gamma_g2, n_seeds=n_seeds, device=device)

    gamma_shift = np.array([0.35, 0.5, 0.75], dtype=float)
    lambda_vals = np.logspace(np.log10(0.15), np.log10(9.0), 17)
    shift_data = _centering_shift(
        N=N,
        p=p,
        beta=beta,
        gamma_vals=gamma_shift,
        lambda_vals=lambda_vals,
        n_trials=n_trials,
        n_seeds=n_seeds,
        device=device,
        topology_cache=topology_cache,
    )

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "g2_gamma": g2_data["gamma"],
        "g2_invD": g2_data["invD"],
        "g2_mean": g2_data["g2_mean"],
        "g2_std": g2_data["g2_std"],
        "shift_gamma": shift_data["gamma"],
        "shift_lambda_vals": shift_data["lambda_vals"],
        "shift_pc_centered": shift_data["pc_centered"],
        "shift_pc_uncentered": shift_data["pc_uncentered"],
        "shift_g2": shift_data["g2"],
        "shift_lam_star_centered": shift_data["lam_star_centered"],
        "shift_lam_star_uncentered": shift_data["lam_star_uncentered"],
        "shift_delta_measured": shift_data["delta_measured"],
        "shift_delta_pred": shift_data["delta_pred"],
    }

    save_npz(RESULT_DIR / "ch03_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03()
