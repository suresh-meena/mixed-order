from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from experiments.common import ensure_dir, save_npz
from tqdm import tqdm

RESULT_DIR = Path(__file__).resolve().parent


def _sign(x: np.ndarray) -> np.ndarray:
    y = np.ones_like(x)
    y[x < 0] = -1.0
    return y


def _build_triples(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tri = N // 3
    idx = np.arange(3 * n_tri)
    return idx[0::3], idx[1::3], idx[2::3]


def _corrupt_bits(x: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    mask = rng.random(y.shape) < noise
    y[mask] *= -1.0
    return y


def _corrupt_bits_batch(x: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    if noise <= 0.0:
        return y
    mask = rng.random(y.shape) < noise
    y[mask] *= -1.0
    return y


def _pairwise_scores(x: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    ov = (x @ patterns.T) / x.shape[-1]
    return ov * ov


def _cubic_scores(x: np.ndarray, patterns: np.ndarray, ti: np.ndarray, tj: np.ndarray, tk: np.ndarray) -> np.ndarray:
    x_sig = x[:, ti] * x[:, tj] * x[:, tk]
    p_sig = patterns[:, ti] * patterns[:, tj] * patterns[:, tk]
    return (x_sig @ p_sig.T) / ti.size


def _mixed_scores(x: np.ndarray, patterns: np.ndarray, ti: np.ndarray, tj: np.ndarray, tk: np.ndarray, cubic_weight: float) -> np.ndarray:
    return _pairwise_scores(x, patterns) + cubic_weight * _cubic_scores(x, patterns, ti, tj, tk)


def _retrieve(
    x0: np.ndarray,
    patterns: np.ndarray,
    ti: np.ndarray,
    tj: np.ndarray,
    tk: np.ndarray,
    mode: str,
    beta: float,
    max_steps: int,
    cubic_weight: float,
) -> np.ndarray:
    state = x0.copy()
    for _ in range(max_steps):
        if mode == "pairwise":
            scores = _pairwise_scores(state[None, :], patterns)[0]
        else:
            scores = _mixed_scores(state[None, :], patterns, ti, tj, tk, cubic_weight)[0]
        logits = beta * scores
        logits = logits - logits.max()
        w = np.exp(logits)
        w = w / (w.sum() + 1e-12)
        new_state = _sign(w @ patterns)
        if np.all(new_state == state):
            break
        state = new_state
    return state


def _retrieve_batch(
    x0: np.ndarray,
    patterns: np.ndarray,
    ti: np.ndarray,
    tj: np.ndarray,
    tk: np.ndarray,
    mode: str,
    beta: float,
    max_steps: int,
    cubic_weight: float,
) -> np.ndarray:
    state = x0.copy()
    p_sig = patterns[:, ti] * patterns[:, tj] * patterns[:, tk]
    inv_n = np.float32(1.0 / state.shape[-1])
    inv_t = np.float32(1.0 / max(1, ti.size))
    for _ in range(max_steps):
        pair = ((state @ patterns.T) * inv_n) ** 2
        if mode == "pairwise":
            scores = pair
        else:
            x_sig = state[:, ti] * state[:, tj] * state[:, tk]
            cubic = (x_sig @ p_sig.T) * inv_t
            scores = pair + cubic_weight * cubic
        logits = beta * scores
        logits = logits - logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        new_state = _sign(w @ patterns)
        if np.array_equal(new_state, state):
            break
        state = new_state
    return state


def run_ch05(
    N: int = 1000,
    n_trials: int = 800,
    n_seeds: int = 10,
    beta: float = 8.0,
    max_steps: int = 10,
    cubic_weight: float = 1.0,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    noise_levels = np.linspace(0.0, 0.5, 21)
    acc_pair_seed = []
    acc_mixed_seed = []
    basin_pair_seed = []
    basin_mixed_seed = []
    rep_example = None

    for s in tqdm(range(n_seeds), desc="seeds"):
        rng = np.random.default_rng(seed + s)
        base = rng.choice(np.array([-1.0, 1.0]), size=(N,)).astype(np.float32)
        patterns = np.stack([base, -base], axis=0).astype(np.float32)
        ti, tj, tk = _build_triples(N)
        p_sig = patterns[:, ti] * patterns[:, tj] * patterns[:, tk]
        inv_n = np.float32(1.0 / N)
        inv_t = np.float32(1.0 / max(1, ti.size))

        acc_pair = []
        acc_mixed = []
        basin_pair = []
        basin_mixed = []

        for noise in tqdm(noise_levels.tolist(), desc="noise levels", leave=False):
            # Vectorized latent/task generation and corruption for all trials.
            u = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_trials, 3))
            y = np.prod(u, axis=1).astype(np.float32)
            target_idx = (y < 0).astype(np.int64)  # y>0 -> idx 0, y<0 -> idx 1
            target = patterns[target_idx]

            cue = y[:, None] * base[None, :]
            cue = _corrupt_bits_batch(cue, noise, rng).astype(np.float32)

            pair_scores = ((cue @ patterns.T) * inv_n) ** 2
            cue_sig = cue[:, ti] * cue[:, tj] * cue[:, tk]
            mix_scores = pair_scores + cubic_weight * ((cue_sig @ p_sig.T) * inv_t)
            pred_pair = np.argmax(pair_scores, axis=1)
            pred_mix = np.argmax(mix_scores, axis=1)

            final_pair = _retrieve_batch(cue, patterns, ti, tj, tk, mode="pairwise", beta=beta, max_steps=max_steps, cubic_weight=cubic_weight)
            final_mix = _retrieve_batch(cue, patterns, ti, tj, tk, mode="mixed", beta=beta, max_steps=max_steps, cubic_weight=cubic_weight)

            succ_pair = (np.mean(final_pair * target, axis=1) > 0.9).astype(np.float32)
            succ_mix = (np.mean(final_mix * target, axis=1) > 0.9).astype(np.float32)

            acc_pair.append(float(np.mean(pred_pair == target_idx)))
            acc_mixed.append(float(np.mean(pred_mix == target_idx)))
            basin_pair.append(float(np.mean(succ_pair)))
            basin_mixed.append(float(np.mean(succ_mix)))

            if rep_example is None and abs(noise - 0.25) < 1e-6:
                rep_example = np.stack([cue[0], final_pair[0], final_mix[0], target[0]], axis=0)

        acc_pair_seed.append(acc_pair)
        acc_mixed_seed.append(acc_mixed)
        basin_pair_seed.append(basin_pair)
        basin_mixed_seed.append(basin_mixed)

    if rep_example is None:
        rep_example = np.stack([patterns[0], patterns[0], patterns[0], patterns[0]], axis=0)

    results: Dict[str, np.ndarray] = {
        "N": np.array([N], dtype=np.int32),
        "noise_levels": noise_levels,
        "accuracy_pairwise": np.mean(np.asarray(acc_pair_seed), axis=0),
        "accuracy_mixed": np.mean(np.asarray(acc_mixed_seed), axis=0),
        "basin_pairwise": np.mean(np.asarray(basin_pair_seed), axis=0),
        "basin_mixed": np.mean(np.asarray(basin_mixed_seed), axis=0),
        "n_seeds": np.array([n_seeds], dtype=np.int32),
        "n_trials": np.array([n_trials], dtype=np.int32),
        "representative_states": rep_example,
    }

    save_npz(RESULT_DIR / "ch05_results.npz", results)
    return results


if __name__ == "__main__":
    run_ch05()
