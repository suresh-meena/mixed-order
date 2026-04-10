from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch03_results.npz")

    fig1, ax1 = plt.subplots(1, 1, figsize=(7.2, 5.2))
    gamma = data["g2_gamma"]
    g2_mean = data["g2_mean"]
    g2_std = data["g2_std"]
    invD = data["g2_invD"]

    ax1.errorbar(gamma, g2_mean, yerr=g2_std, fmt="o-", color=cp["mixed"], capsize=4, label=r"$g_2$ empirical")
    coef = np.polyfit(invD, g2_mean, 1)
    fit = np.polyval(coef, invD)
    ax1.plot(gamma, fit, "--", color=cp["theory"], label=r"linear in $1/D$")
    ax1.set_xlabel(r"$\gamma=D/N$")
    ax1.set_ylabel(r"$g_2$")
    ax1.set_title("Chapter 3.1 covariance strength")
    ax1.legend(frameon=False)
    save_fig(fig1, RESULT_DIR / "ch03_g2.png")
    plt.close(fig1)

    gamma_s = data["shift_gamma"]
    lam = data["shift_lambda_vals"]
    pc_c = data["shift_pc_centered"]
    pc_u = data["shift_pc_uncentered"]
    delta_m = data["shift_delta_measured"]
    delta_p = data["shift_delta_pred"]

    # Plot: Pc vs lambda (centered vs uncentered) as a single figure
    fig_pc, ax_pc = plt.subplots(figsize=(8.6, 5.2))
    for i, g in enumerate(gamma_s.tolist()):
        ax_pc.plot(lam, pc_u[i], "s--", label=rf"uncentered $\gamma={g}$", color=cp["uncentered"] if i == 0 else cp["extra"])
        ax_pc.plot(lam, pc_c[i], "o-", label=rf"centered $\gamma={g}$", color=cp["centered"] if i == 0 else cp["mixed"])
    ax_pc.set_xscale("log")
    ax_pc.set_xlabel(r"$\lambda$")
    ax_pc.set_ylabel(r"$P_c$")
    ax_pc.set_title("Chapter 3.2 centering shifts optimum")
    ax_pc.legend(frameon=False, fontsize=9)
    save_fig(fig_pc, RESULT_DIR / "ch03_centering_pc_vs_lambda.png")
    plt.close(fig_pc)

    # Plot: measured vs predicted shift scaling
    fig_shift, ax_shift = plt.subplots(figsize=(8.6, 5.2))
    ax_shift.scatter(delta_p, delta_m, color=cp["mixed"], s=60)
    if delta_p.size >= 2:
        c1, c0 = np.polyfit(delta_p, delta_m, 1)
        xx = np.linspace(delta_p.min() * 0.9, delta_p.max() * 1.1, 100)
        ax_shift.plot(xx, c1 * xx + c0, "--", color=cp["theory"], label=rf"fit slope={c1:.2f}")
    ax_shift.set_xlabel(r"predicted $\tilde q g_2$")
    ax_shift.set_ylabel(r"measured $\Delta\lambda^*$")
    ax_shift.set_title("Shift scaling")
    ax_shift.legend(frameon=False)
    save_fig(fig_shift, RESULT_DIR / "ch03_centering_shift_scaling.png")
    plt.close(fig_shift)


if __name__ == "__main__":
    main()
