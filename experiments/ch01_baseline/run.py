from __future__ import annotations

from experiments.common import apply_pub_style
from experiments.ch01_baseline.experiment import run_ch01
from experiments.ch01_baseline.plot import main as plot_main


def main() -> None:
    apply_pub_style()
    run_ch01()
    plot_main()


if __name__ == "__main__":
    main()
