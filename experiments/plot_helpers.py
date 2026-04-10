"""Shared plotting and path-setup helpers for experiments.

This module centralizes:
- adding `src/` to `sys.path` so `mixed_order` can be imported from experiments
- matplotlib backend setup
- common plotting utilities (save_fig, edges_from_centers)

Importing this module performs path setup automatically.
"""
from __future__ import annotations

import os
import sys
import typing as _t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project `src` is on sys.path so `mixed_order` imports work.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def apply_pub_style() -> None:
    """Apply publication plotting style from `mixed_order.plotting.style`."""
    try:
        from mixed_order.plotting.style import apply_pub_style as _apply

        _apply()
    except Exception:
        # Fall back silently if style module isn't available yet
        return


def get_color_palette() -> dict:
    try:
        from mixed_order.plotting.style import get_color_palette as _get

        return _get()
    except Exception:
        # Minimal fallback palette
        return {
            "uncentered": "#EE7733",
            "centered": "#0077BB",
            "mixed": "#009988",
            "extra": "#CC3311",
            "theory": "#222222",
        }


def save_fig(fig: plt.Figure, path: str, tight: bool = True) -> None:
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path)


def edges_from_centers(vals: _t.Sequence[float]) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return edges


__all__ = ["apply_pub_style", "get_color_palette", "save_fig", "edges_from_centers", "plt"]
