from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from experiment import learning_boundary_experiment


def main() -> None:
    learning_boundary_experiment()


if __name__ == "__main__":
    main()