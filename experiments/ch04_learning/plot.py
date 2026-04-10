from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch04_results.npz")

    lam = data["lam_vals"]
    alpha = data["alpha_vals"]
    success = data["success_map"]
    drift = data["drift_map"]
    alpha_star = data["alpha_star"]
    g2 = float(data["g2"][0])
    # Use a sparse subset of lambda values in panel 4.1 to avoid over-crowded legends.
    n_show = min(5, lam.size)
    show_idx = np.unique(np.linspace(0, lam.size - 1, n_show).astype(int))

    # Plot: success vs load for each lambda (single plot)
    fig_succ, ax_succ = plt.subplots(figsize=(8.6, 5.0))
    for i in show_idx.tolist():
        lam_i = lam[i]
        ax_succ.plot(alpha, success[i], "o-", label=rf"$\lambda={lam_i:.2f}$")
    ax_succ.set_xlabel(r"$\alpha=P/N$")
    ax_succ.set_ylabel("Retrieval success")
    ax_succ.set_ylim(0.0, 1.02)
    ax_succ.set_title("Chapter 4.1 success vs load")
    ax_succ.legend(frameon=False, fontsize=8)
    save_fig(fig_succ, RESULT_DIR / "ch04_success_vs_load.png")
    plt.close(fig_succ)

    # Plot: deterministic drift vs load (single plot)
    fig_drift, ax_drift = plt.subplots(figsize=(8.6, 5.0))
    for i in show_idx.tolist():
        lam_i = lam[i]
        ax_drift.plot(alpha, drift[i], "o-", label=rf"$\lambda={lam_i:.2f}$")
    ax_drift.set_xlabel(r"$\alpha=P/N$")
    ax_drift.set_ylabel("Deterministic drift")
    ax_drift.set_title("Drift vs load")
    save_fig(fig_drift, RESULT_DIR / "ch04_drift_vs_load.png")
    plt.close(fig_drift)

    # Plot: measured alpha_star vs lambda without straight guide lines
    fig_alpha, ax_alpha = plt.subplots(figsize=(8.6, 5.0))
    ax_alpha.scatter(lam, alpha_star, s=55, color=cp["mixed"], label=r"measured $\alpha^*_{learn}$")
    lam_star = float(lam[int(np.argmax(alpha_star))])
    ax_alpha.scatter([lam_star], [alpha_star.max()], s=70, color=cp["extra"], label=rf"$\lambda^*={lam_star:.2f}$")
    ax_alpha.set_xlabel(r"$\lambda$")
    ax_alpha.set_ylabel(r"$\alpha^*_{learn}$")
    ax_alpha.set_title(rf"Scaling check ($g_2={g2:.3f}$)")
    ax_alpha.legend(frameon=False)
    save_fig(fig_alpha, RESULT_DIR / "ch04_alpha_scaling.png")
    plt.close(fig_alpha)

    labels = data["phase_labels"].astype(int)

    fig2, ax4 = plt.subplots(1, 1, figsize=(7.6, 5.6))
    cmap = ListedColormap(["#1b9e77", "#d95f02", "#7570b3"])
    im = ax4.imshow(
        labels.T,
        origin="lower",
        aspect="auto",
        extent=[lam.min(), lam.max(), alpha.min(), alpha.max()],
        cmap=cmap,
        vmin=0,
        vmax=2,
    )
    ax4.set_xlabel(r"$\lambda$")
    ax4.set_ylabel(r"$\alpha=P/N$")
    ax4.set_title("Chapter 4.2 phase map")

    cbar = fig2.colorbar(im, ax=ax4, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["storage", "learning", "failure"])

    save_fig(fig2, RESULT_DIR / "ch04_phase_map.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
