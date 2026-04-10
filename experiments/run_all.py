from __future__ import annotations

import os
import sys
import subprocess
import time
from typing import List, Sequence

from tqdm import tqdm

# Group assignment: first three experiments -> GPU 0, rest -> GPU 1
GROUP_0: Sequence[str] = (
    "experiments.ch02_capacity.run",
    "experiments.ch03_structured.run",
    "experiments.ch04_learning.run",
)

GROUP_1: Sequence[str] = (
    "experiments.ch01_baseline.run",
    "experiments.ch05_ham.run",
    "experiments.ch06_finite_size.run",
)


def _make_cmd(modules: Sequence[str]) -> List[str]:
    # Build a single Python -c invocation that imports each module's `main`
    imports = []
    calls = []
    for i, m in enumerate(modules):
        imports.append(f"from {m} import main as _m{i}")
        calls.append(f"_m{i}()")
    code = "; ".join(imports + calls)
    return [sys.executable, "-u", "-c", code]


def _run_group(modules: Sequence[str], cuda_visible_devices: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    # Unbuffered output so logs stream to console
    env["PYTHONUNBUFFERED"] = "1"
    cmd = _make_cmd(modules)
    print(f"Starting group on CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}: {modules}")
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    groups = [
        ("GPU 0", _run_group(GROUP_0, "0")),
        ("GPU 1", _run_group(GROUP_1, "1")),
    ]

    rc_map = {}
    with tqdm(total=len(groups), desc="Experiment groups", unit="group") as bar:
        while len(rc_map) < len(groups):
            for label, proc in groups:
                if label in rc_map:
                    continue
                rc = proc.poll()
                if rc is None:
                    continue
                rc_map[label] = rc
                bar.update(1)
                bar.set_postfix_str(f"{label} done")
            if len(rc_map) < len(groups):
                time.sleep(0.5)

    rc0 = rc_map["GPU 0"]
    rc1 = rc_map["GPU 1"]

    if rc0 != 0 or rc1 != 0:
        raise SystemExit(f"One or more experiment groups failed (rc0={rc0}, rc1={rc1})")
    print("All experiment groups finished successfully.")


if __name__ == "__main__":
    main()
