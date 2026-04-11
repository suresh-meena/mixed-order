from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np

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

from experiments.common import ensure_dir, load_npz, save_npz

RESULT_DIR = Path(__file__).resolve().parent

_SOURCE_FILES = {
    "ch03": RESULT_DIR.parent / "ch03_claims" / "ch03_claims_results.npz",
    "ch04_centering": RESULT_DIR.parent / "ch04_centering_compare" / "ch04_centering_compare_results.npz",
    "ch04_learning": RESULT_DIR.parent / "ch04_learning" / "ch04_results.npz",
}


def _ensure_source_result(name: str) -> None:
    path = _SOURCE_FILES[name]
    if path.exists():
        return

    if name == "ch03":
        from experiments.ch03_claims.experiment import run_ch03_claims

        run_ch03_claims()
    elif name == "ch04_centering":
        from experiments.ch04_centering_compare.experiment import run_ch04_centering_compare

        run_ch04_centering_compare()
    elif name == "ch04_learning":
        from experiments.ch04_learning.experiment import run_ch04

        run_ch04()
    else:
        raise KeyError(name)


def run_publication_story() -> Dict[str, np.ndarray]:
    ensure_dir(RESULT_DIR)
    for key in _SOURCE_FILES:
        _ensure_source_result(key)

    ch03 = load_npz(_SOURCE_FILES["ch03"])
    ch04c = load_npz(_SOURCE_FILES["ch04_centering"])
    ch04l = load_npz(_SOURCE_FILES["ch04_learning"])

    summary: Dict[str, np.ndarray] = {
        # Chapter 3: structured teacher and weak-correlation scaling.
        "teacher_rho": ch03["teacher_rho"],
        "teacher_empirical": ch03["teacher_empirical"],
        "teacher_empirical_se": ch03["teacher_empirical_se"],
        "teacher_theory": ch03["teacher_theory"],
        "g2_invD": ch03["g2_invD"],
        "g2_mean": ch03["g2_mean"],
        "g2_std": ch03["g2_std"],
        "g2_fit_slope": ch03["g2_fit_slope"],
        "g2_fit_intercept": ch03["g2_fit_intercept"],
        # Chapter 4: centered vs uncentered.
        "pair_C": ch04c["pair_C"],
        "pair_J_uncentered": ch04c["pair_J_uncentered"],
        "pair_J_centered": ch04c["pair_J_centered"],
        "pair_unc_slope": ch04c["pair_unc_slope"],
        "pair_unc_intercept": ch04c["pair_unc_intercept"],
        "pair_cen_slope": ch04c["pair_cen_slope"],
        "pair_cen_intercept": ch04c["pair_cen_intercept"],
        "drift_uncentered": ch04c["drift_uncentered"],
        "drift_centered": ch04c["drift_centered"],
        "lam_vals_centering": ch04c["lam_vals"],
        "alpha_vals_centering": ch04c["alpha_vals"],
        "g2_centering": ch04c["g2"],
        # Chapter 4: learning boundary scaling.
        "lam_vals_learning": ch04l["lam_vals"],
        "alpha_star_uncentered": ch04l["alpha_star_uncentered"],
        "alpha_pred": ch04l["alpha_pred"],
        "g2_learning": ch04l["g2"],
    }

    save_npz(RESULT_DIR / "ch03_ch04_publication_results.npz", summary)
    return summary


if __name__ == "__main__":
    run_publication_story()

