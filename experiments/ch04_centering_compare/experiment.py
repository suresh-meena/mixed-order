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
from mixed_order.metrics import _DEVICE
from mixed_order.utils import make_generator
from mixed_order_model import MixedOrderHopfieldNetwork
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _drift_curves(
    N: int,
    p: float,
    beta: float,
    lam_vals: np.ndarray,
    alpha_vals: np.ndarray,
    F: torch.Tensor,
    C: torch.Tensor,
    device: str,
    centered: bool,
) -> np.ndarray:
    drift = np.zeros((lam_vals.size, alpha_vals.size), dtype=float)
    if centered:
        return drift

    C_cpu = C.detach().cpu().numpy()
    for lidx, lam in enumerate(tqdm(lam_vals.tolist(), desc="drift sweep", leave=False)):
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
        targets = patterns[mu_idx].detach().cpu().numpy()
        p_arr = np.asarray(p_list, dtype=float)
        drift_field = (p_arr[:, None] / (p * N)) * (targets @ C_masked.T)
        drift[lidx] = np.mean(np.abs(drift_field), axis=1)

    return drift


def _pairwise_scatter(
    N: int,
    p: float,
    beta: float,
    lam: float,
    alpha_ref: float,
    F: torch.Tensor,
    C: torch.Tensor,
    device: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    model = MixedOrderHopfieldNetwork(N=N, p=p, beta=beta, lam=float(lam), device=device)
    model.generate_masks(generator=make_generator(seed, device))
    c_mask = model.c.detach().cpu().numpy()
    C_cpu = C.detach().cpu().numpy()
    C_masked = C_cpu * c_mask
    C_masked[np.diag_indices(N)] = 0.0

    P = max(2, int(round(alpha_ref * N)))
    patterns = generate_structured_patterns(P, F, device=device, generator=make_generator(9000 + seed, device)).to(torch.float32)
    model.store_multiple_p(patterns, [P], centered=False)
    J_unc = model.J[0].detach().cpu().numpy()
    model.store_multiple_p(patterns, [P], centered=True, C=C)
    J_cen = model.J[0].detach().cpu().numpy()

    iu = np.triu_indices(N, 1)
    mask = np.triu(c_mask, 1) > 0
    idx = np.where(mask[iu])[0]
    if idx.size == 0:
        idx = np.arange(iu[0].size)
    rng = np.random.default_rng(seed + 77)
    if idx.size > 2500:
        idx = rng.choice(idx, size=2500, replace=False)

    C_s = C_masked[iu][idx]
    J_u_s = J_unc[iu][idx]
    J_c_s = J_cen[iu][idx]

    unc_slope, unc_intercept = np.polyfit(C_s, J_u_s, 1)
    cen_slope, cen_intercept = np.polyfit(C_s, J_c_s, 1)

    return {
        "pair_C": C_s,
        "pair_J_uncentered": J_u_s,
        "pair_J_centered": J_c_s,
        "pair_unc_slope": np.array([unc_slope], dtype=float),
        "pair_unc_intercept": np.array([unc_intercept], dtype=float),
        "pair_cen_slope": np.array([cen_slope], dtype=float),
        "pair_cen_intercept": np.array([cen_intercept], dtype=float),
        "pair_P": np.array([P], dtype=np.int32),
        "pair_alpha_ref": np.array([alpha_ref], dtype=float),
        "pair_lambda_ref": np.array([lam], dtype=float),
    }


def run_ch04_centering_compare(
    N: int = 96,
    p: float = 0.35,
    beta: float = 0.5,
    D: int = 4,
    factor_seed: int = 1001,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE

    F = generate_factors(N, D, device=device, generator=make_generator(factor_seed, device))
    C = compute_covariance(F)
    g2 = estimate_g2(C)

    lam_vals = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    alpha_vals = np.linspace(0.02, 0.14, 13)
    drift_unc = _drift_curves(N, p, beta, lam_vals, alpha_vals, F, C, device, centered=False)
    drift_cen = _drift_curves(N, p, beta, lam_vals, alpha_vals, F, C, device, centered=True)
    pair = _pairwise_scatter(N, p, beta, lam=0.5, alpha_ref=0.10, F=F, C=C, device=device, seed=factor_seed)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "D": np.array([D], dtype=np.int32),
        "factor_seed": np.array([factor_seed], dtype=np.int32),
        "g2": np.array([g2], dtype=float),
        "lam_vals": lam_vals,
        "alpha_vals": alpha_vals,
        "drift_uncentered": drift_unc,
        "drift_centered": drift_cen,
    }
    results.update(pair)

    save_npz(RESULT_DIR / "ch04_centering_compare_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch04_centering_compare()
