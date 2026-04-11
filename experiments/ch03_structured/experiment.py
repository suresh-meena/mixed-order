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
from typing import Dict

import numpy as np
import torch

from experiments.common import ensure_dir, save_npz
from mixed_order.data.structured import generate_structured_patterns
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success, seed_all
from mixed_order_model import MixedOrderHopfieldNetwork
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
        vals = []
        for _ in range(n_seeds):
            F = generate_factors(N, D, device=device)
            C = compute_covariance(F)
            vals.append(float(estimate_g2(C)))
        vals_np = np.asarray(vals, dtype=float)
        g2_mean.append(float(vals_np.mean()))
        g2_std.append(float(vals_np.std(ddof=0)))
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
                success_threshold=0.99,
                overlap_threshold=0.99,
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
                success_threshold=0.99,
                overlap_threshold=0.99,
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


def _alpha_vs_g2(
    N: int,
    p: float,
    beta: float,
    gamma_vals: np.ndarray,
    lambda_vals: np.ndarray,
    alpha_vals: np.ndarray,
    n_trials: int,
    n_seeds: int,
    device: str,
    topology_cache: dict,
) -> Dict[str, np.ndarray]:
    alpha_star = np.zeros((lambda_vals.size, gamma_vals.size), dtype=float)
    g2_vals = np.zeros(gamma_vals.size, dtype=float)

    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
    for gidx, gamma in enumerate(tqdm(gamma_vals.tolist(), desc="alpha-g2 gamma sweep")):
        D = max(2, int(round(gamma * N)))
        F = generate_factors(N, D, device=device)
        C = compute_covariance(F)
        g2_vals[gidx] = estimate_g2(C)
        for lidx, lam in enumerate(tqdm(lambda_vals.tolist(), desc="lambda sweep", leave=False)):
            pc = find_empirical_pc_by_success(
                N=N,
                p=p,
                beta=beta,
                lam=float(lam),
                n_trials=n_trials,
                n_seeds=n_seeds,
                success_threshold=0.99,
                overlap_threshold=0.99,
                noise_level=0.1,
                device=device,
                F=F,
                C=C,
                centered=True,
                topology_cache=topology_cache,
            )
            # Reuse the exact same alpha sweep logic via a coarse local scan around Pc.
            alpha_star[lidx, gidx] = float(pc) / float(N)
    return {"g2": g2_vals, "alpha_star": alpha_star}


def _find_learning_boundary_from_drift(
    N: int,
    p: float,
    beta: float,
    lam: float,
    F: torch.Tensor,
    n_trials: int,
    n_seeds: int,
    device: str,
) -> float:
    model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=lam, device=device)
    model.generate_masks()
    C = compute_covariance(F)
    C_masked = C.to(device) * model.topology.c
    C_masked.fill_diagonal_(0.0)

    P_max = max(20, int(round(0.25 * N)))
    P_list = np.unique(np.round(np.linspace(2, P_max, 12)).astype(int))

    alpha_crossings = []
    for P in P_list.tolist():
        seed_crossings = []
        patterns = generate_structured_patterns(P, F, device=device)
        model.store_multiple_p(patterns, [P], centered=False)
        for seed in range(n_seeds):
            seed_all(seed)
            mu_idx = torch.arange(n_trials, device=device) % P
            targets = patterns[mu_idx]
            states = targets.unsqueeze(0)
            h_full = model.local_field(states).squeeze(0)
            signal = (h_full * targets).mean().item()
            drift_field = (P / (p * N)) * torch.matmul(targets, C_masked.T)
            drift = torch.abs(drift_field).mean().item()
            seed_crossings.append(signal - drift)
        alpha_crossings.append(float(np.mean(seed_crossings)))

    alpha_crossings = np.asarray(alpha_crossings, dtype=float)
    for i in range(len(P_list) - 1):
        if alpha_crossings[i] > 0 and alpha_crossings[i + 1] <= 0:
            frac = alpha_crossings[i] / (alpha_crossings[i] - alpha_crossings[i + 1])
            return float((P_list[i] + frac * (P_list[i + 1] - P_list[i])) / N)
    return float(P_list[-1] / N)


def _storage_learning_phase_diagram(
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
    g2_vals = np.zeros(gamma_vals.size, dtype=float)
    alpha_storage = np.zeros((lambda_vals.size, gamma_vals.size), dtype=float)
    alpha_learn = np.zeros_like(alpha_storage)

    for gidx, gamma in enumerate(tqdm(gamma_vals.tolist(), desc="phase gamma sweep")):
        D = max(2, int(round(gamma * N)))
        F = generate_factors(N, D, device=device)
        C = compute_covariance(F)
        g2_vals[gidx] = estimate_g2(C)

        for lidx, lam in enumerate(tqdm(lambda_vals.tolist(), desc="phase lambda sweep", leave=False)):
            pc = find_empirical_pc_by_success(
                N=N,
                p=p,
                beta=beta,
                lam=float(lam),
                n_trials=n_trials,
                n_seeds=n_seeds,
                success_threshold=0.99,
                overlap_threshold=0.99,
                noise_level=0.1,
                device=device,
                F=F,
                C=C,
                centered=True,
                topology_cache=topology_cache,
            )
            alpha_storage[lidx, gidx] = float(pc) / float(N)
            alpha_learn[lidx, gidx] = _find_learning_boundary_from_drift(
                N=N,
                p=p,
                beta=beta,
                lam=float(lam),
                F=F,
                n_trials=n_trials,
                n_seeds=n_seeds,
                device=device,
            )

    return {
        "g2": g2_vals,
        "alpha_storage": alpha_storage,
        "alpha_learn": alpha_learn,
    }


def run_ch03(
    N: int = 500,
    p: float = 0.35,
    beta: float = 0.5,
    n_trials: int = 12,
    n_seeds: int = 5,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    topology_cache: dict = {}

    gamma_g2 = np.array([0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0], dtype=float)
    g2_data = _g2_sweep(N, gamma_g2, n_seeds=n_seeds, device=device)

    gamma_shift = np.array([0.35, 1.0], dtype=float)
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

    alpha_g2_gamma = np.array([0.5, 0.75, 1.0, 1.25, 2.0], dtype=float)
    alpha_g2_lam = np.array([0.5, 2.0], dtype=float)
    alpha_g2_vals = np.linspace(0.03, 0.65, 20)
    alpha_g2_data = _alpha_vs_g2(
        N=N,
        p=p,
        beta=beta,
        gamma_vals=alpha_g2_gamma,
        lambda_vals=alpha_g2_lam,
        alpha_vals=alpha_g2_vals,
        n_trials=n_trials,
        n_seeds=n_seeds,
        device=device,
        topology_cache=topology_cache,
    )

    # Use a wider g2 span and more contrasted lambda values so the separation
    # between the storage and learning boundaries is visually obvious.
    phase_gamma = np.array([0.18, 0.22, 0.28, 0.35, 0.45, 0.6, 0.8, 1.0, 1.25, 1.7, 2.3, 3.0], dtype=float)
    phase_lambda = np.array([0.25, 4.0], dtype=float)
    phase_data = _storage_learning_phase_diagram(
        N=N,
        p=p,
        beta=beta,
        gamma_vals=phase_gamma,
        lambda_vals=phase_lambda,
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
        "alpha_g2_gamma": alpha_g2_gamma,
        "alpha_g2_lambda_vals": alpha_g2_lam,
        "alpha_g2_vals": alpha_g2_vals,
        "alpha_g2_g2": alpha_g2_data["g2"],
        "alpha_g2_alpha_star": alpha_g2_data["alpha_star"],
        "phase_gamma": phase_gamma,
        "phase_lambda_vals": phase_lambda,
        "phase_g2": phase_data["g2"],
        "phase_alpha_storage": phase_data["alpha_storage"],
        "phase_alpha_learn": phase_data["alpha_learn"],
    }

    save_npz(RESULT_DIR / "ch03_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03()
