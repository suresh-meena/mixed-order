from __future__ import annotations

from experiments.ch02_capacity.experiment import run_ch02
from experiments.ch02_capacity.plot import main as plot_main


def main() -> None:
    run_ch02()
    plot_main()


if __name__ == "__main__":
    main()
