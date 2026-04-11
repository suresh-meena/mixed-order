from __future__ import annotations

import pathlib
import sys
from pathlib import Path

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

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def _smooth_curve(x: np.ndarray, y: np.ndarray, degree: int = 3, n_grid: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    if xs.size < degree + 1:
        return xs, ys
    coef = np.polyfit(np.log(xs), ys, degree)
    grid = np.logspace(np.log10(xs.min()), np.log10(xs.max()), n_grid)
    return grid, np.polyval(coef, np.log(grid))


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch03_lambda_shift_results.npz")

    g2 = data["g2_vals"]
    lam_iid = float(data["lam_iid"][0])
    lam_emp = data["lam_opt_emp"]
    lam_th = data["lam_opt_theory"]
    shift_emp = data["lam_shift_emp"]
    shift_th = data["lam_shift_theory"]
    g2_grid = data["g2_grid"]
    lam_theory_grid = data["lam_theory_grid"]
    shift_theory_grid = data["shift_theory_grid"]
    lam_emp_s, lam_emp_fit = _smooth_curve(g2, lam_emp)
    shift_emp_s, shift_emp_fit = _smooth_curve(g2, shift_emp)

    fig, (ax_lam, ax_shift) = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    ax_lam.plot(g2_grid, lam_theory_grid, color=cp["theory"], lw=2.2, label=r"$\lambda^*_{\mathrm{struct}}(g_2)$")
    ax_lam.plot(lam_emp_s, lam_emp_fit, color=cp["mixed"], lw=2.0, alpha=0.9, label="empirical fit")
    ax_lam.scatter(g2, lam_emp, s=42, color=cp["mixed"], alpha=0.7, label="empirical optimum")
    ax_lam.scatter(g2, lam_th, s=34, color=cp["centered"], marker="s", alpha=0.75, label="theory at sampled $g_2$")
    ax_lam.axhline(lam_iid, color=cp["extra"], ls="--", lw=1.5, label=rf"iid optimum $\lambda^*={lam_iid:.2f}$")
    ax_lam.set_xscale("log")
    ax_lam.set_xlabel(r"Covariance strength $g_2$")
    ax_lam.set_ylabel(r"Optimal $\lambda^*$")
    ax_lam.set_title("Shift of the mixed-order optimum")
    ax_lam.legend(frameon=False, fontsize=8)

    ax_shift.plot(g2_grid, shift_theory_grid, color=cp["theory"], lw=2.2, label=r"theory $\Delta\lambda^*$")
    ax_shift.plot(shift_emp_s, shift_emp_fit, color=cp["mixed"], lw=2.0, alpha=0.9, label="empirical fit")
    ax_shift.scatter(g2, shift_emp, s=42, color=cp["mixed"], alpha=0.7, label="empirical shift")
    ax_shift.scatter(g2, shift_th, s=34, color=cp["centered"], marker="s", alpha=0.75, label="theory shift at sampled $g_2$")
    ax_shift.axhline(0.0, color="black", lw=0.9, alpha=0.5)
    ax_shift.set_xscale("log")
    ax_shift.set_xlabel(r"Covariance strength $g_2$")
    ax_shift.set_ylabel(r"$\Delta\lambda^* = \lambda^*_{\mathrm{struct}} - \lambda^*_{\mathrm{iid}}$")
    ax_shift.set_title("Lambda shift induced by covariance")
    ax_shift.legend(frameon=False, fontsize=8)

    save_fig(fig, RESULT_DIR / "ch03_lambda_shift.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
