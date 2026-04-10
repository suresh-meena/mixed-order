from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from experiment import limiting_cases


def main() -> None:
    N, n_trials, n_seeds = 1000, 4, 5
    beta = 0.5
    limiting_cases(N, beta, n_trials, n_seeds)


if __name__ == "__main__":
    main()