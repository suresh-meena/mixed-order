from __future__ import annotations

import pathlib
import sys

_file = pathlib.Path(__file__).resolve()
_repo_root = None
for _ancestor in _file.parents:
    if _ancestor.name == "experiments":
        _repo_root = _ancestor.parent
        break
if _repo_root is None:
    _repo_root = _file.parents[1] if len(_file.parents) >= 2 else _file.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.ch03_structured_frontier.experiment import run_ch03_structured_frontier
from experiments.ch03_structured_frontier.plot import main as plot_main


def main() -> None:
    run_ch03_structured_frontier()
    plot_main()


if __name__ == "__main__":
    main()
