from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments.plot_helpers import apply_pub_style, get_color_palette, plt, save_fig


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = Path(__file__).resolve().parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_npz(path: Path, payload: Dict[str, Any]) -> None:
    flat: Dict[str, np.ndarray] = {}

    def _pack(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                name = f"{prefix}__{k}" if prefix else str(k)
                _pack(name, v)
            return
        flat[prefix] = np.asarray(value)

    for k, v in payload.items():
        _pack(str(k), v)
    np.savez(path, **flat)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def unpack_prefixed(data: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    p = f"{prefix}__"
    for key, val in data.items():
        if key.startswith(p):
            out[key[len(p) :]] = val
    return out


def success_rate(overlaps: np.ndarray, threshold: float = 0.9) -> float:
    return float((overlaps >= threshold).mean())


def build_alpha_grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, int(n), dtype=float)


__all__ = [
    "ROOT",
    "EXPERIMENTS_ROOT",
    "ensure_dir",
    "save_npz",
    "load_npz",
    "unpack_prefixed",
    "success_rate",
    "build_alpha_grid",
    "apply_pub_style",
    "get_color_palette",
    "plt",
    "save_fig",
]
