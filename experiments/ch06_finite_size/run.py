from __future__ import annotations

from experiments.ch06_finite_size.experiment import run_ch06
from experiments.ch06_finite_size.plot import main as plot_main


def main() -> None:
    run_ch06()
    plot_main()


if __name__ == "__main__":
    main()
