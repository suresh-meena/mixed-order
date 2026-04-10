from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from experiments.common import ensure_dir, save_npz
from mixed_order.data.structured import compute_covariance, estimate_g2, generate_factors, generate_structured_patterns
from mixed_order.metrics import _DEVICE, run_batched_retrieval_many_seeds
from mixed_order.utils import make_generator
from mixed_order_model import MixedOrderHopfieldNetwork
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _success_overlap_curves(
    N: int,
    p: float,
    beta: float,
    lam_vals: np.ndarray,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    n_trials: int,
    n_seeds: int,
    device: str,
    topology_cache: dict,
) -> Dict[str, np.ndarray]:
    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
    success = np.zeros((lam_vals.size, alpha_vals.size), dtype=float)
    overlap = np.zeros_like(success)

    for lidx, lam in enumerate(tqdm(lam_vals.tolist(), desc="lambda sweep", leave=False)):
        seed_batch = list(range(n_seeds))
        ov_t = run_batched_retrieval_many_seeds(
            seed_batch=seed_batch,
            N=N,
            p_list=p_list,
            p=p,
            beta=beta,
            n_trials=n_trials,
            lam=float(lam),
            device=device,
            F=F,
            centered=False,
            noise_level=0.1,
            topology_cache=topology_cache,
        )
        success[lidx] = (ov_t >= 0.9).to(dtype=torch.float32).mean(dim=(0, 2)).cpu().numpy()
        overlap[lidx] = ov_t.mean(dim=(0, 2)).cpu().numpy()

    return {"success": success, "overlap": overlap}


def _drift_alignment(
    N: int,
    p: float,
    beta: float,
    lam_vals: np.ndarray,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    C: torch.Tensor,
    device: str,
) -> Dict[str, np.ndarray]:
    drift = np.zeros((lam_vals.size, alpha_vals.size), dtype=float)
    align = np.zeros_like(drift)

    C_cpu = C.detach().cpu().numpy()
    iu = np.triu_indices(N, 1)

    for lidx, lam in enumerate(tqdm(lam_vals.tolist(), desc="lambda sweep", leave=False)):
        model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=float(lam), device=device)
        model.generate_masks(generator=make_generator(123 + lidx, device))
        c_mask = model.c.detach().cpu().numpy()

        C_masked = C_cpu * c_mask
        C_masked[np.diag_indices(N)] = 0.0
        p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
        p_max = int(max(p_list))
        patterns = generate_structured_patterns(p_max, F, device=device, generator=make_generator(8000 + lidx, device)).to(torch.float32)
        model.store_multiple_p(patterns, p_list, centered=False)

        mu_idx = torch.tensor([lidx % P for P in p_list], dtype=torch.long, device=device)
        targets = patterns[mu_idx].detach().cpu().numpy()  # (B, N)
        p_arr = np.asarray(p_list, dtype=float)
        drift_field = (p_arr[:, None] / (p * N)) * (targets @ C_masked.T)
        drift[lidx] = np.mean(np.abs(drift_field), axis=1)

        # Vectorized correlation between each J(P) upper-triangle and C upper-triangle.
        J_batch = model.J.detach().cpu().numpy()  # (B, N, N)
        x = J_batch[:, iu[0], iu[1]]  # (B, M)
        y = C_masked[iu]  # (M,)
        y_mean = y.mean()
        y_std = y.std()
        x_mean = x.mean(axis=1)
        x_std = x.std(axis=1)
        cov = np.mean((x - x_mean[:, None]) * (y[None, :] - y_mean), axis=1)
        corr = np.zeros_like(x_mean)
        valid = (x_std > 1e-12) & (y_std > 1e-12)
        corr[valid] = cov[valid] / (x_std[valid] * y_std)
        align[lidx] = corr

    return {"drift": drift, "align": align}


def _alpha_star_from_success(alpha_vals: np.ndarray, success: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    out = np.zeros(success.shape[0], dtype=float)
    for i in range(success.shape[0]):
        s = success[i]
        idx = np.where(s < threshold)[0]
        if idx.size == 0:
            out[i] = alpha_vals[-1]
        elif idx[0] == 0:
            out[i] = alpha_vals[0]
        else:
            j = idx[0]
            x0, x1 = alpha_vals[j - 1], alpha_vals[j]
            y0, y1 = s[j - 1], s[j]
            frac = (threshold - y0) / (y1 - y0 + 1e-12)
            out[i] = x0 + frac * (x1 - x0)
    return out


def _phase_labels(success_map: np.ndarray, overlap_map: np.ndarray) -> np.ndarray:
    labels = np.full(success_map.shape, 2, dtype=np.int32)  # 2=failure
    labels[(success_map < 0.9) & (overlap_map >= 0.5)] = 1  # 1=learning
    labels[success_map >= 0.9] = 0  # 0=storage
    return labels


def run_ch04(
    N: int = 1000,
    p: float = 0.35,
    beta: float = 0.5,
    gamma: float = 0.75,
    n_trials: int = 12,
    n_seeds: int = 10,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE

    D = max(2, int(round(gamma * N)))
    F = generate_factors(N, D, device=device)
    C = compute_covariance(F)
    g2 = estimate_g2(C)

    lam_vals = np.linspace(0.0, 6.0, 11)
    alpha_vals = np.linspace(0.03, 0.65, 24)

    topology_cache: dict = {}
    curves = _success_overlap_curves(N, p, beta, lam_vals, alpha_vals, F, n_trials=n_trials, n_seeds=n_seeds, device=device, topology_cache=topology_cache)
    drift_align = _drift_alignment(N, p, beta, lam_vals, alpha_vals, F, C, device=device)

    alpha_star = _alpha_star_from_success(alpha_vals, curves["success"], threshold=0.9)
    x_scale = (1.0 + lam_vals / 2.0) / np.sqrt(g2)
    c_fit = float(np.dot(x_scale, alpha_star) / (np.dot(x_scale, x_scale) + 1e-12))
    alpha_pred = c_fit * x_scale

    phase_labels = _phase_labels(curves["success"], curves["overlap"])

    # boundary curve for phase map: first alpha where success drops below 0.9
    boundary_alpha = _alpha_star_from_success(alpha_vals, curves["success"], threshold=0.9)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "gamma": np.array([gamma], dtype=float),
        "g2": np.array([g2], dtype=float),
        "lam_vals": lam_vals,
        "alpha_vals": alpha_vals,
        "success_map": curves["success"],
        "overlap_map": curves["overlap"],
        "drift_map": drift_align["drift"],
        "align_map": drift_align["align"],
        "alpha_star": alpha_star,
        "alpha_pred": alpha_pred,
        "c_fit": np.array([c_fit], dtype=float),
        "phase_labels": phase_labels,
        "boundary_alpha": boundary_alpha,
    }

    save_npz(RESULT_DIR / "ch04_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch04()
