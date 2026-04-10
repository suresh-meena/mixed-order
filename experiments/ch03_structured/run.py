from __future__ import annotations

from experiments.ch03_structured.experiment import run_ch03
from experiments.ch03_structured.plot import main as plot_main


def main() -> None:
    run_ch03()
    plot_main()


if __name__ == "__main__":
    main()
