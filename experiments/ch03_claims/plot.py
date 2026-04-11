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


def _identity_line(ax, xs: np.ndarray, color: str, label: str | None = None, **kwargs) -> None:
    lo = float(np.min(xs))
    hi = float(np.max(xs))
    ax.plot([lo, hi], [lo, hi], color=color, label=label, **kwargs)


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch03_claims_results.npz")

    # Figure 1: teacher correlation and g2 scaling.
    fig1, (ax_t, ax_g2) = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

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

    save_fig(fig1, RESULT_DIR / "ch03_claims_teacher_g2.png")
    plt.close(fig1)

    # Figure 2: deterministic pairwise drift.
    P_over_N = data["drift_P_over_N"]
    emp = data["drift_empirical"]
    pred = data["drift_predicted"]
    slopes = data["drift_slopes"]
    cors = data["drift_correlations"]

    fig2, axes2 = plt.subplots(2, 2, figsize=(12.4, 11.0), constrained_layout=True)
    axes2 = axes2.ravel()
    for i, ax in enumerate(axes2):
        ax.scatter(pred[i], emp[i], s=7, alpha=0.45, color=cp["mixed"])
        _identity_line(ax, pred[i], cp["theory"], label="identity")
        ax.set_title(rf"$\alpha={P_over_N[i]:.2f}$, slope={slopes[i]:.2f}, $r$={cors[i]:.2f}")
        ax.set_xlabel(r"Predicted drift $(P/pN)\,C_{ij}$")
        ax.set_ylabel(r"Empirical $J_{ij}$")
        ax.legend(frameon=False, loc="upper left")

    save_fig(fig2, RESULT_DIR / "ch03_claims_pairwise_drift.png")
    plt.close(fig2)

    # Figure 3: cubic channel residual scaling.
    cubic_N = data["cubic_N"].astype(float)
    mean_w = data["cubic_mean_w"]
    mean_w_std = data["cubic_mean_w_std"]
    g4_proxy = data["cubic_g4_proxy"]
    g4_std = data["cubic_g4_proxy_std"]
    g4_slope = float(np.asarray(data["cubic_g4_fit_slope"]).item())
    g4_intercept = float(np.asarray(data["cubic_g4_fit_intercept"]).item())

    fig3, (ax_w, ax_g4) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    ax_w.errorbar(cubic_N, mean_w, yerr=mean_w_std, fmt="o-", color=cp["extra"], capsize=3, label=r"Mean $W_{ijk}$")
    ax_w.axhline(0.0, color=cp["theory"], ls="--", lw=1.5)
    ax_w.set_xlabel(r"Network size $N$")
    ax_w.set_ylabel(r"$\mathbb{E}[W_{ijk}]$")
    ax_w.set_title("Three-body channel immunity")
    ax_w.legend(frameon=False)

    ax_g4.errorbar(cubic_N, g4_proxy, yerr=g4_std, fmt="o-", color=cp["centered"], capsize=3, label=r"Proxy $g_4=\langle W^2\rangle$")
    fit_x = np.linspace(float(cubic_N.min()) * 0.95, float(cubic_N.max()) * 1.05, 160)
    fit_y = np.exp(g4_intercept) * fit_x ** g4_slope
    ax_g4.plot(fit_x, fit_y, "--", color=cp["theory"], lw=2.0, label=rf"fit slope={g4_slope:.2f}")
    ax_g4.set_xscale("log")
    ax_g4.set_yscale("log")
    ax_g4.set_xlabel(r"Network size $N$")
    ax_g4.set_ylabel(r"$g_4$ proxy")
    ax_g4.set_title(r"Residual cubic correction scales to zero")
    ax_g4.legend(frameon=False)

    save_fig(fig3, RESULT_DIR / "ch03_claims_cubic_channel.png")
    plt.close(fig3)

    # Figure 4: structured capacity and optimal lambda shift.
    gamma = data["capacity_gamma"]
    g2 = data["capacity_g2"]
    lambda_vals = data["capacity_lambda_vals"]
    alpha_curves = data["capacity_alpha_curves"]
    alpha_uncentered = data["capacity_alpha_uncentered"]
    lam_opt = data["capacity_lam_opt"]
    lam_opt_uncentered = data["capacity_lam_opt_uncentered"]
    lam_iid = float(np.asarray(data["capacity_lam_iid"]).item())
    g2_grid = data["capacity_g2_grid"]
    lam_opt_grid = data["capacity_lam_opt_grid"]

    fig4, (ax_cap, ax_opt) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    colors = [cp["centered"], cp["mixed"], cp["extra"]]
    for i, gamma_i in enumerate(gamma.tolist()):
        ax_cap.plot(
            lambda_vals,
            alpha_curves[i],
            "o-",
            color=colors[i],
            label=rf"$\gamma={gamma_i:.2f}$, $g_2={g2[i]:.2f}$",
        )
        ax_cap.plot(
            lambda_vals,
            alpha_uncentered[i],
            "s--",
            color=colors[i],
            alpha=0.45,
            label=rf"uncentered $\gamma={gamma_i:.2f}$" if i == 0 else None,
        )
        ax_cap.axvline(lam_opt[i], color=colors[i], ls=":", alpha=0.8)
        ax_cap.axvline(lam_opt_uncentered[i], color=colors[i], ls="--", alpha=0.35)
    ax_cap.set_xscale("log")
    ax_cap.set_xlabel(r"$\lambda$")
    ax_cap.set_ylabel(r"Empirical capacity load $\alpha_c = P_c/N$")
    ax_cap.set_title("Structured storage capacity: centered vs uncentered")
    ax_cap.legend(frameon=False, fontsize=9)

    ax_opt.plot(g2_grid, lam_opt_grid, color=cp["theory"], lw=2.2, label=r"$\lambda^*_{\mathrm{struct}} = \tilde q(1/p + g_2)$")
    ax_opt.scatter(g2, lam_opt, s=60, color=cp["mixed"], zorder=3, label="Measured from curves")
    ax_opt.scatter(g2, lam_opt_uncentered, s=52, color=cp["uncentered"], zorder=3, marker="s", label="Uncentered optimum")
    ax_opt.axhline(lam_iid, color=cp["extra"], ls="--", lw=1.6, label=rf"iid optimum $\lambda^*={lam_iid:.2f}$")
    ax_opt.set_xlabel(r"Covariance strength $g_2$")
    ax_opt.set_ylabel(r"Optimal $\lambda^*$")
    ax_opt.set_title("Empirical optimum vs covariance strength")
    ax_opt.set_ylim(bottom=0.0)
    ax_opt.legend(frameon=False, fontsize=9)

    save_fig(fig4, RESULT_DIR / "ch03_claims_capacity.png")
    plt.close(fig4)


if __name__ == "__main__":
    main()
