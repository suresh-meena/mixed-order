from __future__ import annotations

# Ensure repository root is on sys.path when running this script directly
import sys
import pathlib
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
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors, generate_structured_patterns
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success
from mixed_order.theory import optimal_lambda, optimal_lambda_structured, replica_capacity_structured
from mixed_order.utils import make_generator
from mixed_order_model import MixedOrderHopfieldNetwork


RESULT_DIR = Path(__file__).resolve().parent


def _teacher_sign_claim(rho_vals: np.ndarray, n_samples: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    empirical = np.zeros(rho_vals.size, dtype=float)
    empirical_se = np.zeros_like(empirical)
    theory = (2.0 / np.pi) * np.arcsin(rho_vals)

    for i, rho in enumerate(rho_vals.tolist()):
        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        samples = rng.multivariate_normal(np.zeros(2), cov, size=n_samples)
        signs = np.sign(samples)
        signs[signs == 0.0] = 1.0
        prod = signs[:, 0] * signs[:, 1]
        empirical[i] = float(prod.mean())
        empirical_se[i] = float(prod.std(ddof=0) / np.sqrt(n_samples))

    return {
        "teacher_rho": rho_vals,
        "teacher_empirical": empirical,
        "teacher_empirical_se": empirical_se,
        "teacher_theory": theory,
    }


def _g2_scaling_claim(N: int, gamma_vals: np.ndarray, n_seeds: int, device: str) -> Dict[str, np.ndarray]:
    D_vals = np.maximum(2, np.round(gamma_vals * N).astype(int))
    g2_mean = np.zeros(gamma_vals.size, dtype=float)
    g2_std = np.zeros_like(g2_mean)

    for i, D in enumerate(D_vals.tolist()):
        vals = []
        for seed in range(n_seeds):
            F = generate_factors(N, D, device=device, generator=make_generator(1017 + 97 * i + seed, device))
            C = compute_covariance(F)
            vals.append(float(estimate_g2(C)))
        vals_np = np.asarray(vals, dtype=float)
        g2_mean[i] = float(vals_np.mean())
        g2_std[i] = float(vals_np.std(ddof=0))

    invD = 1.0 / D_vals.astype(float)
    slope, intercept = np.polyfit(invD, g2_mean, 1)

    return {
        "g2_gamma": gamma_vals,
        "g2_D": D_vals.astype(np.int32),
        "g2_invD": invD,
        "g2_mean": g2_mean,
        "g2_std": g2_std,
        "g2_fit_slope": np.array([slope], dtype=float),
        "g2_fit_intercept": np.array([intercept], dtype=float),
    }


def _pairwise_drift_claim(
    N: int,
    p: float,
    beta: float,
    D: int,
    P_fracs: np.ndarray,
    n_samples: int,
    seed: int,
    device: str,
) -> Dict[str, np.ndarray]:
    F = generate_factors(N, D, device=device, generator=make_generator(seed, device))
    C = compute_covariance(F).detach().cpu()

    model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=0.0, device=device)
    model.generate_masks(generator=make_generator(seed + 1, device))
    mask = torch.triu(model.c, diagonal=1).bool().detach().cpu()

    P_vals = np.maximum(2, np.round(P_fracs * N).astype(int))
    P_max = int(P_vals.max())
    patterns = generate_structured_patterns(P_max, F, device=device, generator=make_generator(seed + 2, device))

    emp = np.zeros((P_vals.size, n_samples), dtype=float)
    pred = np.zeros_like(emp)
    slopes = np.zeros(P_vals.size, dtype=float)
    cors = np.zeros_like(slopes)

    for i, P in enumerate(P_vals.tolist()):
        model.store_multiple_p(patterns, [P], centered=False, C=C.to(device))
        J = model.J[0].detach().cpu()
        pred_full = (float(P) / (p * N)) * C

        emp_i = J[mask].flatten().numpy()
        pred_i = pred_full[mask].flatten().numpy()
        if emp_i.size < n_samples:
            raise RuntimeError("Not enough pairwise samples for drift claim plot")

        rng = np.random.default_rng(seed + 1000 + i)
        take = rng.choice(emp_i.size, size=n_samples, replace=False)
        emp[i] = emp_i[take]
        pred[i] = pred_i[take]
        slopes[i] = float(np.polyfit(pred[i], emp[i], 1)[0])
        cors[i] = float(np.corrcoef(pred[i], emp[i])[0, 1])

    return {
        "drift_N": np.array([N], dtype=np.int32),
        "drift_p": np.array([p], dtype=float),
        "drift_beta": np.array([beta], dtype=float),
        "drift_D": np.array([D], dtype=np.int32),
        "drift_P_vals": P_vals.astype(np.int32),
        "drift_P_over_N": (P_vals / float(N)).astype(float),
        "drift_empirical": emp,
        "drift_predicted": pred,
        "drift_slopes": slopes,
        "drift_correlations": cors,
    }


def _cubic_residual_claim(
    N_vals: np.ndarray,
    p: float,
    beta: float,
    lam: float,
    gamma: float,
    alpha: float,
    n_seeds: int,
    device: str,
) -> Dict[str, np.ndarray]:
    mean_w = np.zeros((N_vals.size, n_seeds), dtype=float)
    g4_proxy = np.zeros_like(mean_w)
    n_tri = np.zeros_like(mean_w)
    D_vals = np.maximum(2, np.round(gamma * N_vals).astype(int))
    P_vals = np.maximum(2, np.round(alpha * N_vals).astype(int))

    for i, N in enumerate(N_vals.tolist()):
        D = int(D_vals[i])
        P = int(P_vals[i])
        for seed in range(n_seeds):
            F = generate_factors(N, D, device=device, generator=make_generator(4000 + 211 * i + seed, device))
            C = compute_covariance(F)
            model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=lam, device=device)
            model.generate_masks(generator=make_generator(5000 + 211 * i + seed, device))
            patterns = generate_structured_patterns(P, F, device=device, generator=make_generator(6000 + 211 * i + seed, device))
            model.store_multiple_p(patterns, [P], centered=False, C=C)
            W = model.W_vals[0].detach().cpu().float()
            mean_w[i, seed] = float(W.mean().item())
            g4_proxy[i, seed] = float((W * W).mean().item())
            n_tri[i, seed] = float(W.numel())

    mean_w_avg = mean_w.mean(axis=1)
    mean_w_std = mean_w.std(axis=1, ddof=0)
    g4_mean = g4_proxy.mean(axis=1)
    g4_std = g4_proxy.std(axis=1, ddof=0)

    slope, intercept = np.polyfit(np.log(N_vals.astype(float)), np.log(g4_mean), 1)

    return {
        "cubic_N": N_vals.astype(np.int32),
        "cubic_D": D_vals.astype(np.int32),
        "cubic_P": P_vals.astype(np.int32),
        "cubic_mean_w_seed": mean_w,
        "cubic_g4_proxy_seed": g4_proxy,
        "cubic_n_tri_seed": n_tri,
        "cubic_mean_w": mean_w_avg,
        "cubic_mean_w_std": mean_w_std,
        "cubic_g4_proxy": g4_mean,
        "cubic_g4_proxy_std": g4_std,
        "cubic_g4_fit_slope": np.array([slope], dtype=float),
        "cubic_g4_fit_intercept": np.array([intercept], dtype=float),
    }


def _structured_capacity_claim(
    N: int,
    p: float,
    beta: float,
    gamma_vals: np.ndarray,
    lambda_vals: np.ndarray,
    n_seeds: int,
    device: str,
) -> Dict[str, np.ndarray]:
    g2_vals = np.zeros(gamma_vals.size, dtype=float)
    alpha_curves = np.zeros((gamma_vals.size, lambda_vals.size), dtype=float)
    alpha_uncentered = np.zeros_like(alpha_curves)
    lam_opt_vals = np.zeros(gamma_vals.size, dtype=float)
    lam_opt_uncentered = np.zeros(gamma_vals.size, dtype=float)
    D_vals = np.maximum(2, np.round(gamma_vals * N).astype(int))

    for i, D in enumerate(D_vals.tolist()):
        F = generate_factors(N, D, device=device, generator=make_generator(8000 + 173 * i, device))
        C = compute_covariance(F)
        g2_vals[i] = float(estimate_g2(C))
        lam_opt_vals[i] = float(optimal_lambda_structured(p, beta, g2_vals[i]))
        lam_opt_uncentered[i] = float(optimal_lambda(p, beta))

        topology_cache: dict = {}
        for j, lam in enumerate(lambda_vals.tolist()):
            alpha_curves[i, j] = float(
                find_empirical_pc_by_success(
                    N=N,
                    p=p,
                    beta=beta,
                    lam=float(lam),
                    n_trials=10,
                    n_seeds=n_seeds,
                    success_threshold=0.95,
                    overlap_threshold=0.95,
                    noise_level=0.1,
                    device=device,
                    F=F,
                    C=C,
                    centered=True,
                    topology_cache=topology_cache,
                )
            ) / float(N)
            alpha_uncentered[i, j] = float(
                find_empirical_pc_by_success(
                    N=N,
                    p=p,
                    beta=beta,
                    lam=float(lam),
                    n_trials=10,
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
            ) / float(N)

    g2_grid = np.linspace(max(1e-3, float(g2_vals.min()) * 0.75), float(g2_vals.max()) * 1.25, 240)
    lam_opt_grid = optimal_lambda_structured(p, beta, g2_grid)
    lam_iid = float(optimal_lambda(p, beta))

    return {
        "capacity_N": np.array([N], dtype=np.int32),
        "capacity_p": np.array([p], dtype=float),
        "capacity_beta": np.array([beta], dtype=float),
        "capacity_gamma": gamma_vals,
        "capacity_D": D_vals.astype(np.int32),
        "capacity_g2": g2_vals,
        "capacity_lambda_vals": lambda_vals,
        "capacity_alpha_curves": alpha_curves,
        "capacity_alpha_uncentered": alpha_uncentered,
        "capacity_lam_opt": lam_opt_vals,
        "capacity_lam_opt_uncentered": lam_opt_uncentered,
        "capacity_lam_iid": np.array([lam_iid], dtype=float),
        "capacity_g2_grid": g2_grid,
        "capacity_lam_opt_grid": lam_opt_grid,
    }


def run_ch03_claims(
    N: int = 160,
    p: float = 0.35,
    beta: float = 0.5,
    n_teacher_samples: int = 6000,
    n_g2_seeds: int = 5,
    n_drift_samples: int = 2500,
    n_cubic_seeds: int = 4,
    n_capacity_seeds: int = 4,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE

    teacher_rho = np.linspace(-0.95, 0.95, 19, dtype=float)
    teacher = _teacher_sign_claim(teacher_rho, n_samples=n_teacher_samples, seed=123)

    gamma_vals = np.array([0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=float)
    g2 = _g2_scaling_claim(N=N, gamma_vals=gamma_vals, n_seeds=n_g2_seeds, device=device)

    drift = _pairwise_drift_claim(
        N=N,
        p=p,
        beta=beta,
        D=max(2, int(round(0.4 * N))),
        P_fracs=np.array([0.05, 0.12, 0.20, 0.30], dtype=float),
        n_samples=n_drift_samples,
        seed=321,
        device=device,
    )

    cubic = _cubic_residual_claim(
        N_vals=np.array([64, 96, 128, 160, 224], dtype=np.int32),
        p=p,
        beta=beta,
        lam=2.0,
        gamma=1.0,
        alpha=0.18,
        n_seeds=n_cubic_seeds,
        device=device,
    )

    capacity = _structured_capacity_claim(
        N=N,
        p=p,
        beta=beta,
        gamma_vals=np.array([0.35, 1.0, 2.0], dtype=float),
        lambda_vals=np.logspace(np.log10(0.2), np.log10(12.0), 13),
        n_seeds=n_capacity_seeds,
        device=device,
    )

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
    }
    results.update(teacher)
    results.update(g2)
    results.update(drift)
    results.update(cubic)
    results.update(capacity)

    save_npz(RESULT_DIR / "ch03_claims_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03_claims()
