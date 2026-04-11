from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

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
from mixed_order.theory import compute_q_from_budget, optimal_lambda, optimal_lambda_structured
from mixed_order.utils import make_generator, seed_all

RESULT_DIR = Path(__file__).resolve().parent


def _parabolic_peak(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the peak of y(x) with a quadratic fit around the best sample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float(x[int(np.argmax(y))])

    idx = int(np.argmax(y))
    lo = max(0, idx - 2)
    hi = min(x.size, idx + 3)
    xx = np.log(x[lo:hi])
    yy = y[lo:hi]
    if xx.size < 3 or np.allclose(yy, yy[0]):
        return float(x[idx])

    a, b, c = np.polyfit(xx, yy, 2)
    if a >= 0:
        return float(x[idx])
    peak = -b / (2.0 * a)
    return float(np.exp(np.clip(peak, xx.min(), xx.max())))


def run_ch03_lambda_shift(
    N: int = 500,
    p: float = 0.35,
    beta: float = 0.5,
    gamma_vals: np.ndarray | None = None,
    lambda_vals: np.ndarray | None = None,
    n_seeds: int = 4,
    seed: int = 1234,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    seed_all(seed)
    if gamma_vals is None:
        gamma_vals = np.array([0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35], dtype=float)
    if lambda_vals is None:
        lambda_vals = np.logspace(np.log10(0.2), np.log10(32.0), 19)

    _, q_tilde = compute_q_from_budget(p, N, beta)
    lam_iid = float(optimal_lambda(p, beta))

    g2_vals = np.zeros(gamma_vals.size, dtype=float)
    lam_opt_emp = np.zeros_like(g2_vals)
    lam_opt_theory = np.zeros_like(g2_vals)
    lam_shift_emp = np.zeros_like(g2_vals)
    lam_shift_theory = np.zeros_like(g2_vals)

    topology_cache: dict = {}

    for i, gamma in enumerate(tqdm(gamma_vals.tolist(), desc="gamma sweep")):
        D = max(2, int(round(gamma * N)))
        pc_vals = np.zeros((3, lambda_vals.size), dtype=float)
        g2_local = []
        for s in tqdm(range(3), desc="teacher draws", leave=False):
            teacher_seed = seed + 9000 + 173 * i + 1009 * s
            F = generate_factors(N, D, device=device, generator=make_generator(teacher_seed, device))
            C = compute_covariance(F)
            g2_local.append(float(estimate_g2(C)))
            for j, lam in enumerate(tqdm(lambda_vals.tolist(), desc="lambda sweep", leave=False)):
                pc = find_empirical_pc_by_success(
                    N=N,
                    p=p,
                    beta=beta,
                    lam=float(lam),
                    n_trials=4,
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
                pc_vals[s, j] = float(pc) / float(N)

        g2 = float(np.mean(g2_local))
        g2_vals[i] = g2
        lam_opt_theory[i] = float(optimal_lambda_structured(p, beta, g2))
        lam_shift_theory[i] = lam_opt_theory[i] - lam_iid

        pc_mean = pc_vals.mean(axis=0)
        lam_opt_emp[i] = _parabolic_peak(lambda_vals, pc_mean)
        lam_shift_emp[i] = lam_opt_emp[i] - lam_iid

    g2_grid = np.linspace(max(1e-3, g2_vals.min() * 0.75), g2_vals.max() * 1.15, 240)
    lam_theory_grid = optimal_lambda_structured(p, beta, g2_grid)
    shift_theory_grid = lam_theory_grid - lam_iid

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "gamma_vals": gamma_vals,
        "g2_vals": g2_vals,
        "lambda_vals": lambda_vals,
        "lam_iid": np.array([lam_iid], dtype=float),
        "lam_opt_emp": lam_opt_emp,
        "lam_opt_theory": lam_opt_theory,
        "lam_shift_emp": lam_shift_emp,
        "lam_shift_theory": lam_shift_theory,
        "g2_grid": g2_grid,
        "lam_theory_grid": lam_theory_grid,
        "shift_theory_grid": shift_theory_grid,
    }

    save_npz(RESULT_DIR / "ch03_lambda_shift_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch03_lambda_shift()
