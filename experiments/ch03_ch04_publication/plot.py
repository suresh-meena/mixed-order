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


def _save_teacher_validation(data: dict, cp: dict) -> None:
    fig, (ax_t, ax_g2) = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)

    rho = data["teacher_rho"]
    ax_t.errorbar(
        rho,
        data["teacher_empirical"],
        yerr=data["teacher_empirical_se"],
        fmt="o",
        color=cp["mixed"],
        capsize=3,
        label="Monte Carlo",
    )
    ax_t.plot(rho, data["teacher_theory"], "-", color=cp["theory"], lw=2.2, label=r"$\frac{2}{\pi}\arcsin(\rho)$")
    ax_t.set_xlabel(r"Latent cosine $\rho$")
    ax_t.set_ylabel(r"$\mathbb{E}[\xi_i \xi_j]$")
    ax_t.set_title("Gaussian-sign teacher")
    ax_t.legend(frameon=False)

    invD = data["g2_invD"]
    ax_g2.errorbar(
        invD,
        data["g2_mean"],
        yerr=data["g2_std"],
        fmt="o",
        color=cp["centered"],
        capsize=3,
        label=r"Empirical $g_2$",
    )
    fit_x = np.linspace(float(invD.min()) * 0.95, float(invD.max()) * 1.05, 120)
    fit_y = float(np.asarray(data["g2_fit_slope"]).item()) * fit_x + float(np.asarray(data["g2_fit_intercept"]).item())
    ax_g2.plot(fit_x, fit_y, "--", color=cp["theory"], lw=2.0, label=r"linear fit in $1/D$")
    ax_g2.set_xlabel(r"Inverse latent dimension $1/D$")
    ax_g2.set_ylabel(r"$g_2$")
    ax_g2.set_title("Weak-correlation scaling")
    ax_g2.legend(frameon=False)

    save_fig(fig, RESULT_DIR / "ch03_ch04_publication_teacher_validation.png")
    plt.close(fig)


def _save_centering_compare(data: dict, cp: dict) -> None:
    pair_C = data["pair_C"]
    pair_J_unc = data["pair_J_uncentered"]
    pair_J_cen = data["pair_J_centered"]
    unc_slope = float(np.asarray(data["pair_unc_slope"]).item())
    unc_intercept = float(np.asarray(data["pair_unc_intercept"]).item())
    cen_slope = float(np.asarray(data["pair_cen_slope"]).item())
    cen_intercept = float(np.asarray(data["pair_cen_intercept"]).item())
    lam = data["lam_vals_centering"]
    alpha = data["alpha_vals_centering"]
    drift_u = data["drift_uncentered"]
    g2 = float(np.asarray(data["g2_centering"]).item())

    fig, (ax_sc_u, ax_sc_c) = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
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

    save_fig(fig, RESULT_DIR / "ch03_ch04_publication_centering_scatter.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.8, 5.0))
    show_idx = np.unique(np.linspace(0, lam.size - 1, min(4, lam.size)).astype(int))
    for i in show_idx.tolist():
        lam_i = lam[i]
        ax2.plot(alpha, drift_u[i], "o-", label=rf"uncentered $\lambda={lam_i:.2f}$")
    ax2.plot(alpha, np.zeros_like(alpha), "k--", lw=1.6, label="centered baseline")
    ax2.set_xlabel(r"$\alpha=P/N$")
    ax2.set_ylabel("Deterministic drift")
    ax2.set_title(rf"Centering removes the coherent drift ($g_2={g2:.3f}$)")
    ax2.legend(frameon=False, fontsize=8)
    save_fig(fig2, RESULT_DIR / "ch03_ch04_publication_centering_drift.png")
    plt.close(fig2)


def _save_learning_scaling(data: dict, cp: dict) -> None:
    lam = data["lam_vals_learning"]
    alpha_star = data["alpha_star_uncentered"]
    alpha_pred = data["alpha_pred"]
    g2 = float(np.asarray(data["g2_learning"]).item())

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.scatter(lam, alpha_star, s=55, color=cp["uncentered"], label=r"measured $\alpha^*_{learn}$")
    ax.plot(lam, alpha_pred, "--", color=cp["theory"], label=r"fit $\propto (1+\lambda/2)/\sqrt{g_2}$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\alpha^*_{learn}$")
    ax.set_title(rf"Storage-learning boundary scaling ($g_2={g2:.3f}$)")
    ax.legend(frameon=False)
    save_fig(fig, RESULT_DIR / "ch03_ch04_publication_learning_scaling.png")
    plt.close(fig)


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch03_ch04_publication_results.npz")

    _save_teacher_validation(data, cp)
    _save_centering_compare(data, cp)
    _save_learning_scaling(data, cp)


if __name__ == "__main__":
    main()

