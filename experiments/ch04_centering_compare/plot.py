from __future__ import annotations

# Ensure repository root is on sys.path when running this script directly
import sys
import pathlib
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


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch04_centering_compare_results.npz")

    lam = data["lam_vals"]
    alpha = data["alpha_vals"]
    drift_u = data["drift_uncentered"]
    drift_c = data["drift_centered"]
    g2 = float(data["g2"][0])
    pair_C = data["pair_C"]
    pair_J_unc = data["pair_J_uncentered"]
    pair_J_cen = data["pair_J_centered"]
    unc_slope = float(np.asarray(data["pair_unc_slope"]).item())
    unc_intercept = float(np.asarray(data["pair_unc_intercept"]).item())
    cen_slope = float(np.asarray(data["pair_cen_slope"]).item())
    cen_intercept = float(np.asarray(data["pair_cen_intercept"]).item())

    fig1, (ax_sc_u, ax_sc_c) = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
    ax_sc_u.scatter(pair_C, pair_J_unc, s=9, alpha=0.35, color=cp["uncentered"])
    xx = np.linspace(float(pair_C.min()), float(pair_C.max()), 100)
    ax_sc_u.plot(xx, unc_slope * xx + unc_intercept, color=cp["theory"], lw=2.0)
    ax_sc_u.axhline(0.0, color="black", lw=0.8, alpha=0.4)
    ax_sc_u.set_title("Uncentered weights")
    ax_sc_u.set_xlabel(r"$C_{ij}$")
    ax_sc_u.set_ylabel(r"$J_{ij}$")

    ax_sc_c.scatter(pair_C, pair_J_cen, s=9, alpha=0.35, color=cp["centered"])
    ax_sc_c.plot(xx, cen_slope * xx + cen_intercept, color=cp["theory"], lw=2.0)
    ax_sc_c.axhline(0.0, color="black", lw=0.8, alpha=0.4)
    ax_sc_c.set_title("Centered weights")
    ax_sc_c.set_xlabel(r"$C_{ij}$")

    save_fig(fig1, RESULT_DIR / "ch04_centering_compare_scatter.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.6, 5.0))
    for i, lam_i in enumerate(lam.tolist()):
        ax2.plot(alpha, drift_u[i], "o-", label=rf"uncentered $\lambda={lam_i:.2f}$", color=cp["uncentered"] if i == 0 else cp["extra"])
    ax2.plot(alpha, np.zeros_like(alpha), "k--", lw=1.6, label="centered baseline")
    ax2.set_xlabel(r"$\alpha=P/N$")
    ax2.set_ylabel("Deterministic drift")
    ax2.set_title(rf"Centering removes drift ($g_2={g2:.3f}$)")
    ax2.legend(frameon=False, fontsize=8)
    save_fig(fig2, RESULT_DIR / "ch04_centering_compare_drift.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
