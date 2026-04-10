from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from experiment import heatmap_p_lambda

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mixed_order.metrics import _DEVICE


def main() -> None:
    N = 1000
    beta = 0.5
    n_p = 10
    n_lam = 10
    n_trials = 8
    n_seeds = 5
    device = _DEVICE
    topology_device = device
    n_jobs = 1 if device == "cuda" else max(1, (os.cpu_count() or 2) // 2)
    seed_batch_size = min(n_seeds, 4) if device == "cuda" else 1
    triangle_chunk_size = 8192 if device == "cuda" else 4096

    heatmap_p_lambda(
        N,
        beta,
        n_p,
        n_lam,
        n_trials,
        n_seeds,
        device=device,
        topology_device=topology_device,
        n_jobs=n_jobs,
        seed_batch_size=seed_batch_size,
        triangle_chunk_size=triangle_chunk_size,
    )


if __name__ == "__main__":
    main()