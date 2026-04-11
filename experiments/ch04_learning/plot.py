from __future__ import annotations

# Ensure repository root is on sys.path when running this script directly
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

from pathlib import Path

import numpy as np

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch04_results.npz")

    lam = data["lam_vals"]
    alpha = data["alpha_vals"]
    success_unc = data["success_map_uncentered"]
    success_cen = data["success_map_centered"]
    drift_unc = data["drift_map_uncentered"]
    drift_cen = data["drift_map_centered"]
    alpha_star_unc = data["alpha_star_uncentered"]
    alpha_star_cen = data["alpha_star_centered"]
    alpha_pred = data["alpha_pred"]
    g2 = float(data["g2"][0])
    # Use a sparse subset of lambda values in panel 4.1 to avoid over-crowded legends.
    n_show = min(5, lam.size)
    show_idx = np.unique(np.linspace(0, lam.size - 1, n_show).astype(int))

    # Plot: centered vs uncentered success curves.
    fig_succ, (ax_succ_u, ax_succ_c) = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True, constrained_layout=True)
    for i in show_idx.tolist():
        lam_i = lam[i]
        ax_succ_u.plot(alpha, success_unc[i], "o-", label=rf"$\lambda={lam_i:.2f}$")
        ax_succ_c.plot(alpha, success_cen[i], "o-", label=rf"$\lambda={lam_i:.2f}$")
    ax_succ_u.set_xlabel(r"$\alpha=P/N$")
    ax_succ_u.set_ylabel("Retrieval success")
    ax_succ_u.set_ylim(0.0, 1.02)
    ax_succ_u.set_title("Uncentered structured data")
    ax_succ_u.legend(frameon=False, fontsize=8)
    ax_succ_c.set_xlabel(r"$\alpha=P/N$")
    ax_succ_c.set_title("Centered structured data")
    ax_succ_c.legend(frameon=False, fontsize=8)
    save_fig(fig_succ, RESULT_DIR / "ch04_success_vs_load.png")
    plt.close(fig_succ)

    # Plot: deterministic drift vs load.
    fig_drift, ax_drift = plt.subplots(figsize=(8.6, 5.0))
    for i in show_idx.tolist():
        lam_i = lam[i]
        ax_drift.plot(alpha, drift_unc[i], "o-", label=rf"uncentered $\lambda={lam_i:.2f}$")
    ax_drift.plot(alpha, np.zeros_like(alpha), "k--", lw=1.5, label="centered baseline")
    ax_drift.set_xlabel(r"$\alpha=P/N$")
    ax_drift.set_ylabel("Deterministic drift")
    ax_drift.set_title("Centering removes the coherent drift")
    ax_drift.legend(frameon=False, fontsize=8)
    save_fig(fig_drift, RESULT_DIR / "ch04_drift_vs_load.png")
    plt.close(fig_drift)

    # Plot: measured alpha_star vs lambda with predicted scaling.
    fig_alpha, ax_alpha = plt.subplots(figsize=(8.6, 5.0))
    ax_alpha.scatter(lam, alpha_star_unc, s=55, color=cp["uncentered"], label=r"uncentered $\alpha^*_{learn}$")
    ax_alpha.plot(lam, alpha_pred, "--", color=cp["theory"], label=r"fit $\propto (1+\lambda/2)/\sqrt{g_2}$")
    ax_alpha.set_xlabel(r"$\lambda$")
    ax_alpha.set_ylabel(r"$\alpha^*_{learn}$")
    ax_alpha.set_title(rf"Uncentered learning boundary scaling ($g_2={g2:.3f}$)")
    ax_alpha.legend(frameon=False)
    save_fig(fig_alpha, RESULT_DIR / "ch04_alpha_scaling.png")
    plt.close(fig_alpha)


if __name__ == "__main__":
    main()
