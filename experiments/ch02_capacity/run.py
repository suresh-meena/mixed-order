from __future__ import annotations

import sys
import pathlib
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

from experiments.ch02_capacity.experiment import run_ch02
from experiments.ch02_capacity.plot import main as plot_main


def main() -> None:
    run_ch02()
    plot_main()


if __name__ == "__main__":
    main()
