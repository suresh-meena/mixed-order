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
from typing import Dict, Tuple

import numpy as np

from experiments.common import apply_pub_style, ensure_dir, save_npz
from mixed_order.metrics import _DEVICE, find_empirical_pc_by_success, run_batched_retrieval_many_seeds
from tqdm import tqdm

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco


RESULT_DIR = Path(__file__).resolve().parent


@njit(cache=True)
def _energy(J: np.ndarray, s: np.ndarray) -> float:
    return -0.5 * float(s @ J @ s)


@njit(cache=True)
def _sync_step(J: np.ndarray, s: np.ndarray) -> np.ndarray:
    h = J @ s
    out = np.ones_like(s)
    out[h < 0.0] = -1.0
    return out


@njit(cache=True)
def _async_run(J: np.ndarray, s0: np.ndarray, n_sweeps: int) -> Tuple[np.ndarray, np.ndarray]:
    s = s0.copy()
    trace = np.empty(n_sweeps + 1, dtype=np.float64)
    trace[0] = _energy(J, s)
    for t in range(n_sweeps):
        for i in range(s.shape[0]):
            h_i = 0.0
            for j in range(s.shape[0]):
                h_i += J[i, j] * s[j]
            s[i] = 1.0 if h_i >= 0.0 else -1.0
        trace[t + 1] = _energy(J, s)
    return s, trace


@njit(cache=True)
def _sync_run(J: np.ndarray, s0: np.ndarray, n_steps: int) -> np.ndarray:
    s = s0.copy()
    for _ in range(n_steps):
        s_new = _sync_step(J, s)
        if np.all(s_new == s):
            return s_new
        s = s_new
    return s


def _build_hopfield_weights(patterns: np.ndarray) -> np.ndarray:
    n = patterns.shape[1]
    J = (patterns.T @ patterns) / n
    np.fill_diagonal(J, 0.0)
    return J


def _generate_iid_patterns(P: int, N: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(P, N))


def _flip_noise(x: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    n_flip = max(1, int(frac * x.size))
    idx = rng.choice(np.arange(x.size), size=n_flip, replace=False)
    y[idx] *= -1.0
    return y


def _alpha_curve(N: int, alpha_grid: np.ndarray, n_seeds: int, n_trials: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    p_list = np.maximum(1, np.round(alpha_grid * N).astype(int)).tolist()
    seed_batch = list(range(n_seeds))
    overlaps = run_batched_retrieval_many_seeds(
        seed_batch=seed_batch,
        N=N,
        p_list=p_list,
        p=1.0,
        beta=0.5,
        n_trials=n_trials,
        lam=0.0,
        noise_level=noise,
        device=_DEVICE,
    ).detach().cpu().numpy()
    success = (overlaps >= 0.99).mean(axis=2)
    mean_m = overlaps.mean(axis=2)
    return success.mean(axis=0), mean_m.mean(axis=0)


def run_ch01(
    n_values: Tuple[int, ...] = (200, 500, 1000),
    n_alpha: int = 32,
    n_trials: int = 16,
    n_seeds: int = 5,
    noise_level: float = 0.1,
    update_N: int = 500,
    update_seeds: int = 5,
    update_noise: float = 0.1,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    alpha_grid = np.linspace(0.04, 0.20, n_alpha)

    pc_vals = []
    curves_success = []
    curves_m = []
    for N in tqdm(n_values, desc="N values"):
        succ, mean_m = _alpha_curve(N, alpha_grid, n_seeds=n_seeds, n_trials=n_trials, noise=noise_level)
        curves_success.append(succ)
        curves_m.append(mean_m)
        pc = find_empirical_pc_by_success(
            N=N,
            p=1.0,
            beta=0.5,
            lam=0.0,
            n_trials=n_trials,
            n_seeds=n_seeds,
            success_threshold=0.99,
            overlap_threshold=0.99,
            noise_level=noise_level,
            device=_DEVICE,
        )
        pc_vals.append(pc)

    # 1.2 update-rule sanity check at one subcritical and one near-critical load,
    # selected from the N=update_N curve.
    if update_N in n_values:
        ref_idx = int(np.where(np.asarray(n_values) == update_N)[0][0])
    else:
        ref_idx = int(np.argmin(np.abs(np.asarray(n_values) - update_N)))
    ref_success = np.asarray(curves_success)[ref_idx]
    idx_sub = int(np.argmax(ref_success >= 0.97)) if np.any(ref_success >= 0.97) else 0
    idx_near = int(np.argmin(np.abs(ref_success - 0.99)))
    alpha_update = np.unique(np.asarray([alpha_grid[idx_sub], alpha_grid[idx_near]], dtype=float))
    async_curve = []
    sync_curve = []
    async_m = []
    sync_m = []
    representative_trace = None

    for a_idx, alpha in enumerate(tqdm(alpha_update, desc="alpha update")):
        P = max(2, int(round(alpha * update_N)))
        succ_async_seed = []
        succ_sync_seed = []
        m_async_seed = []
        m_sync_seed = []
        for seed in tqdm(range(update_seeds), desc="seeds", leave=False):
            rng = np.random.default_rng(seed)
            patterns = _generate_iid_patterns(P, update_N, rng)
            J = _build_hopfield_weights(patterns)
            target = patterns[seed % P]
            init = _flip_noise(target, update_noise, rng)

            final_async, et = _async_run(J, init, n_sweeps=25)
            final_sync = _sync_run(J, init, n_steps=25)
            m_a = float(np.mean(final_async * target))
            m_s = float(np.mean(final_sync * target))
            succ_async_seed.append(m_a > 0.99)
            succ_sync_seed.append(m_s > 0.99)
            m_async_seed.append(m_a)
            m_sync_seed.append(m_s)
            if representative_trace is None and a_idx == 0 and seed == 0:
                representative_trace = et

        async_curve.append(float(np.mean(succ_async_seed)))
        sync_curve.append(float(np.mean(succ_sync_seed)))
        async_m.append(float(np.mean(m_async_seed)))
        sync_m.append(float(np.mean(m_sync_seed)))

    results: Dict[str, np.ndarray] = {
        "n_values": np.asarray(n_values, dtype=np.int32),
        "alpha_grid": alpha_grid,
        "success_curves": np.asarray(curves_success),
        "mean_overlap_curves": np.asarray(curves_m),
        "pc_vals": np.asarray(pc_vals),
        "pc_over_n": np.asarray(pc_vals) / np.asarray(n_values),
        "alpha_update": alpha_update,
        "async_success": np.asarray(async_curve),
        "sync_success": np.asarray(sync_curve),
        "async_mean_overlap": np.asarray(async_m),
        "sync_mean_overlap": np.asarray(sync_m),
        "energy_trace_async": np.asarray(representative_trace if representative_trace is not None else []),
        "meta": np.array([n_trials, n_seeds, noise_level, update_N, update_seeds, update_noise], dtype=float),
    }

    save_npz(RESULT_DIR / "ch01_results.npz", results)
    return results


if __name__ == "__main__":
    apply_pub_style()
    run_ch01()
