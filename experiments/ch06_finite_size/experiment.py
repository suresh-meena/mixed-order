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
from mixed_order.data.structured import compute_covariance, generate_factors
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success, run_batched_retrieval_many_seeds
from mixed_order.theory import optimal_lambda
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _alpha_star(N: int, p: float, beta: float, lam: float, F, n_trials: int, n_seeds: int, device: str, topology_cache: dict) -> float:
    alpha_vals = np.linspace(0.03, 0.65, 20)
    p_list = np.maximum(2, np.round(alpha_vals * N).astype(int)).tolist()
    ov = run_batched_retrieval_many_seeds(
        seed_batch=list(range(n_seeds)),
        N=N,
        p_list=p_list,
        p=p,
        beta=beta,
        n_trials=n_trials,
        lam=lam,
        device=device,
        F=F,
        centered=False,
        noise_level=0.1,
        topology_cache=topology_cache,
    )
    success = (ov >= 0.99).to(dtype=torch.float32).mean(dim=(0, 2)).cpu().numpy()

    idx = np.where(success < 0.99)[0]
    if idx.size == 0:
        return float(alpha_vals[-1])
    j = idx[0]
    if j == 0:
        return float(alpha_vals[0])
    x0, x1 = alpha_vals[j - 1], alpha_vals[j]
    y0, y1 = success[j - 1], success[j]
    frac = (0.99 - y0) / (y1 - y0 + 1e-12)
    return float(x0 + frac * (x1 - x0))


def run_ch06(
    n_values: tuple[int, ...] = (50, 100, 200, 500, 750,  1000),
    p: float = 0.35,
    beta: float = 0.5,
    n_trials: int = 12,
    n_seeds: int = 5,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    device = _DEVICE
    topology_cache: dict = {}

    pc_vals = []
    lam_star_struct = []
    alpha_star_vals = []

    lam_grid = np.logspace(np.log10(0.15), np.log10(9.0), 16)

    for N in tqdm(n_values, desc="N sweep"):
        lam_opt = optimal_lambda(p, beta)
        pc = find_empirical_pc_by_success(
            N=N,
            p=p,
            beta=beta,
            lam=float(lam_opt),
            n_trials=n_trials,
            n_seeds=n_seeds,
            success_threshold=0.99,
            overlap_threshold=0.99,
            noise_level=0.1,
            device=device,
            topology_cache=topology_cache,
        )
        pc_vals.append(pc)

        D = N
        F = generate_factors(N, D, device=device)
        C = compute_covariance(F)

        pc_struct = []
        for lam in tqdm(lam_grid.tolist(), desc="lambda sweep", leave=False):
            pc_l = find_empirical_pc_by_success(
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
            pc_struct.append(pc_l)
        lam_star_struct.append(float(lam_grid[int(np.argmax(np.asarray(pc_struct)))]))

        alpha_star_vals.append(_alpha_star(N, p, beta, lam=2.0, F=F, n_trials=n_trials, n_seeds=n_seeds, device=device, topology_cache=topology_cache))

    results: Dict[str, np.ndarray] = {
        "n_values": np.asarray(n_values, dtype=np.int32),
        "pc_vals": np.asarray(pc_vals),
        "pc_over_n": np.asarray(pc_vals) / np.asarray(n_values, dtype=float),
        "lam_star_struct": np.asarray(lam_star_struct),
        "alpha_star_learn": np.asarray(alpha_star_vals),
        "p": np.array([p], dtype=float),
        "beta": np.array([beta], dtype=float),
    }

    save_npz(RESULT_DIR / "ch06_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch06()
