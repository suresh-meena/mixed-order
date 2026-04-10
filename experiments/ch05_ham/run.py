from __future__ import annotations

from experiments.ch05_ham.experiment import run_ch05
from experiments.ch05_ham.plot import main as plot_main


def main() -> None:
    run_ch05()
    plot_main()


if __name__ == "__main__":
    main()
