from __future__ import annotations

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

from experiments.common import ensure_dir, save_npz
from mixed_order.data.structured import compute_covariance, generate_factors
from mixed_order.metrics import _DEVICE, run_batched_retrieval_many_seeds
from mixed_order.theory import optimal_lambda
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _classical_success_curve(
    N: int,
    alpha_vals: np.ndarray,
    n_trials: int,
    n_seeds: int,
    noise_level: float,
) -> np.ndarray:
    p_list = np.maximum(1, np.round(alpha_vals * N).astype(int)).tolist()
    overlaps = run_batched_retrieval_many_seeds(
        seed_batch=list(range(n_seeds)),
        N=N,
        p_list=p_list,
        p=1.0,
        beta=0.5,
        n_trials=n_trials,
        lam=0.0,
        noise_level=noise_level,
        device=_DEVICE,
    ).detach().cpu().numpy()
    return (overlaps >= 0.99).mean(axis=2).mean(axis=0)


def _mixed_success_curve(
    N: int,
    alpha_vals: np.ndarray,
    p: float,
    beta: float,
    lam: float,
    n_trials: int,
    n_seeds: int,
    noise_level: float,
) -> np.ndarray:
    success = np.zeros(alpha_vals.size, dtype=float)
    seed_batch = list(range(n_seeds))
    F = generate_factors(N, max(2, int(round(0.5 * N))), device=_DEVICE)
    C = compute_covariance(F)
    topology_cache: dict = {}
    p_list = np.maximum(1, np.round(alpha_vals * N).astype(int)).tolist()
    overlaps = run_batched_retrieval_many_seeds(
        seed_batch=seed_batch,
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        lam=lam,
        device=_DEVICE,
        topology_cache=topology_cache,
        F=F,
        C=C,
        centered=True,
        noise_level=noise_level,
    ).detach().cpu().numpy()
    success = (overlaps >= 0.99).mean(axis=2).mean(axis=0)
    return success


def run_comparison(
    N: int = 1000,
    p: float = 0.35,
    beta: float = 0.5,
    lam_mixed: float = 2.0,
    n_alpha: int = 32,
    n_trials: int = 12,
    n_seeds: int = 5,
    noise_level: float = 0.1,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    alpha_vals = np.linspace(0.04, 0.22, n_alpha)
    classical_success = _classical_success_curve(N, alpha_vals, n_trials=n_trials, n_seeds=n_seeds, noise_level=noise_level)
    mixed_success = _mixed_success_curve(N, alpha_vals, p=p, beta=beta, lam=lam_mixed, n_trials=n_trials, n_seeds=n_seeds, noise_level=noise_level)
    lam_opt = float(optimal_lambda(p, beta))
    mixed_opt_success = _mixed_success_curve(N, alpha_vals, p=p, beta=beta, lam=lam_opt, n_trials=n_trials, n_seeds=n_seeds, noise_level=noise_level)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
        "lam_mixed": np.array([lam_mixed], dtype=float),
        "lam_opt": np.array([lam_opt], dtype=float),
        "alpha_vals": alpha_vals,
        "classical_success": classical_success,
        "mixed_success": mixed_success,
        "mixed_opt_success": mixed_opt_success,
    }
    save_npz(RESULT_DIR / "comparison_pairwise_mixed_results.npz", results)
    return results


if __name__ == "__main__":
    run_comparison()
