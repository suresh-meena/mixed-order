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
from mixed_order.metrics import _DEVICE, run_batched_retrieval_many_seeds
from mixed_order.theory import optimal_lambda_structured, compute_q_from_budget
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
    overlaps = run_batched_retrieval_many_seeds(
        seed_batch=list(range(n_seeds)),
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        lam=float(lam),
        device=device,
        topology_cache=topology_cache,
        F=F,
        centered=centered,
        noise_level=0.1,
    ).detach().cpu().numpy()
    success = (overlaps >= 0.95).mean(axis=2).mean(axis=0)
    overlap = overlaps.mean(axis=2).mean(axis=0)
    return success, overlap


def _success_heatmap(
    N: int,
    p: float,
    beta: float,
    lambda_vals: np.ndarray,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    n_trials: int,
    n_seeds: int,
    device: str,
    topology_cache: dict,
) -> np.ndarray:
    out = np.zeros((lambda_vals.size, alpha_vals.size), dtype=float)
    for i, lam in enumerate(lambda_vals.tolist()):
        success, _ = _success_curve(
            N=N,
            p=p,
            beta=beta,
            lam=float(lam),
            alpha_vals=alpha_vals,
            F=F,
            n_trials=n_trials,
            n_seeds=n_seeds,
            device=device,
            centered=False,
            topology_cache=topology_cache,
        )
        out[i] = success
    return out


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


def _field_decomposition(
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
        model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=float(lam), device=device)
        model.generate_masks(generator=make_generator(seed_offset, device))
        c_mask = model.c.detach().cpu().numpy()

        patterns = generate_structured_patterns(
            P,
            F,
            device=device,
            generator=make_generator(9000 + seed_offset + idx, device),
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


def run_ch03_structured_compare(
    N: int = 200,
    p: float = 0.35,
    beta: float = 0.5,
    gamma: float = 1.0,
    n_trials: int = 4,
    n_seeds: int = 4,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE

    D = max(2, int(round(gamma * N)))
    F = generate_factors(N, D, device=device, generator=make_generator(1001, device))
    C = compute_covariance(F)
    g2 = estimate_g2(C)

    _, q_tilde = compute_q_from_budget(p, N, beta)
    lam_struct = float(optimal_lambda_structured(p, beta, g2))
    lam_iid = float(q_tilde / p)

    alpha_vals = np.linspace(0.03, 0.5, 16)
    topology_cache: dict = {}

    success_pair, overlap_pair = _success_curve(
        N=N,
        p=p,
        beta=beta,
        lam=0.0,
        alpha_vals=alpha_vals,
        F=F,
        n_trials=n_trials,
        n_seeds=n_seeds,
        device=device,
        centered=False,
        topology_cache=topology_cache,
    )
    success_mix, overlap_mix = _success_curve(
        N=N,
        p=p,
        beta=beta,
        lam=lam_struct,
        alpha_vals=alpha_vals,
        F=F,
        n_trials=n_trials,
        n_seeds=n_seeds,
        device=device,
        centered=False,
        topology_cache=topology_cache,
    )
    lambda_vals = np.linspace(0.0, max(1.5 * lam_struct, 1.5 * lam_iid), 7)
    success_lambda_grid = _success_heatmap(
        N=N,
        p=p,
        beta=beta,
        lambda_vals=lambda_vals,
        alpha_vals=alpha_vals,
        F=F,
        n_trials=n_trials,
        n_seeds=n_seeds,
        device=device,
        topology_cache=topology_cache,
    )
    gain_lambda_grid = success_lambda_grid - success_pair[None, :]

    field_pair = _field_decomposition(
        N=N,
        p=p,
        beta=beta,
        lam=0.0,
        alpha_vals=alpha_vals,
        F=F,
        C=C,
        device=device,
        centered=False,
        seed_offset=110,
    )
    field_mix = _field_decomposition(
        N=N,
        p=p,
        beta=beta,
        lam=lam_struct,
        alpha_vals=alpha_vals,
        F=F,
        C=C,
        device=device,
        centered=False,
        seed_offset=220,
    )

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "gamma": np.array([gamma], dtype=float),
        "D": np.array([D], dtype=np.int32),
        "g2": np.array([g2], dtype=float),
        "lam_struct": np.array([lam_struct], dtype=float),
        "lam_iid": np.array([lam_iid], dtype=float),
        "alpha_vals": alpha_vals,
        "lambda_vals": lambda_vals,
        "success_pair_uncentered": success_pair,
        "success_mix_uncentered": success_mix,
        "success_lambda_grid": success_lambda_grid,
        "gain_lambda_grid": gain_lambda_grid,
        "overlap_pair_uncentered": overlap_pair,
        "overlap_mix_uncentered": overlap_mix,
        "alpha_star_pair_uncentered": np.array([_alpha_star(alpha_vals, success_pair)], dtype=float),
        "alpha_star_mix_uncentered": np.array([_alpha_star(alpha_vals, success_mix)], dtype=float),
        "signal_pair_uncentered": field_pair["signal"],
        "drift_pair_uncentered": field_pair["drift"],
        "margin_pair_uncentered": field_pair["margin"],
        "signal_mix_uncentered": field_mix["signal"],
        "drift_mix_uncentered": field_mix["drift"],
        "margin_mix_uncentered": field_mix["margin"],
    }

    from experiments.common import save_npz

    save_npz(RESULT_DIR / "ch03_structured_compare_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03_structured_compare()
