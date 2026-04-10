from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from experiment import structured_experiment


def main() -> None:
    structured_experiment()


if __name__ == "__main__":
    main()