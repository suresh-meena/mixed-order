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
    alpha_g2 = data["alpha_g2_g2"]
    alpha_star = data["alpha_g2_alpha_star"]
    lam_two = data["alpha_g2_lambda_vals"]
    phase_g2 = data["phase_g2"]
    phase_lam = data["phase_lambda_vals"]
    phase_storage = data["phase_alpha_storage"]
    phase_learn = data["phase_alpha_learn"]
    phase_order = np.argsort(phase_g2)
    phase_g2_sorted = phase_g2[phase_order]
    phase_x = np.logspace(np.log10(float(np.min(phase_g2_sorted))), np.log10(float(np.max(phase_g2_sorted))), 300)

    fig_phase, ax_phase = plt.subplots(figsize=(8.6, 5.6))
    colors = [cp["centered"], cp["mixed"]]
    for i, lam_i in enumerate(phase_lam.tolist()):
        storage_sorted = phase_storage[i][phase_order]
        learn_sorted = phase_learn[i][phase_order]
        storage_smooth = np.interp(phase_x, phase_g2_sorted, storage_sorted)
        learn_smooth = np.interp(phase_x, phase_g2_sorted, learn_sorted)
        ax_phase.fill_between(
            phase_x,
            storage_smooth,
            learn_smooth,
            color=colors[i],
            alpha=0.12,
        )
        ax_phase.plot(
            phase_x,
            storage_smooth,
            "-",
            color=colors[i],
            lw=2.2,
            label=rf"storage boundary, $\lambda={lam_i:.2f}$",
        )
        ax_phase.plot(
            phase_x,
            learn_smooth,
            "--",
            color=colors[i],
            alpha=0.95,
            lw=2.0,
            label=rf"learning boundary, $\lambda={lam_i:.2f}$",
        )
        ax_phase.scatter(phase_g2_sorted, storage_sorted, color=colors[i], s=18, alpha=0.65)
        ax_phase.scatter(phase_g2_sorted, learn_sorted, color=colors[i], s=18, alpha=0.65, marker="s")
    ax_phase.set_xlabel(r"Covariance strength $g_2$")
    ax_phase.set_ylabel(r"Storage load $\alpha = P/N$")
    ax_phase.set_title("Storage-learning phase diagram")
    ax_phase.set_xscale("log")
    ax_phase.set_xlim(float(np.min(phase_g2)) * 0.95, float(np.max(phase_g2)) * 1.05)
    ax_phase.legend(frameon=False, fontsize=9, ncol=2)
    save_fig(fig_phase, RESULT_DIR / "ch03_storage_learning_phase_diagram.png")
    plt.close(fig_phase)

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

    fig_ag2, ax_ag2 = plt.subplots(figsize=(8.6, 5.2))
    for i, lam_i in enumerate(lam_two.tolist()):
        ax_ag2.plot(alpha_g2, alpha_star[i], "o-", label=rf"$\lambda={lam_i:.2f}$")
    ax_ag2.set_xlabel(r"$g_2$")
    ax_ag2.set_ylabel(r"$\alpha^*_{learn}$")
    ax_ag2.set_title(r"Chapter 3: $\alpha^*$ vs $g_2$ for two $\lambda$")
    ax_ag2.legend(frameon=False)
    save_fig(fig_ag2, RESULT_DIR / "ch03_alpha_vs_g2.png")
    plt.close(fig_ag2)


if __name__ == "__main__":
    main()
