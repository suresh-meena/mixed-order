from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch01_results.npz")

    n_values = data["n_values"].astype(int)
    alpha_grid = data["alpha_grid"]
    success_curves = data["success_curves"]
    pc_over_n = data["pc_over_n"]

    # Plot: success curves for different N
    fig_success, ax_success = plt.subplots(figsize=(8.6, 5.2))
    for i, N in enumerate(n_values.tolist()):
        ax_success.plot(alpha_grid, success_curves[i], "o-", label=f"N={N}")
    ax_success.axvline(0.138, ls="--", color=cp["theory"], label=r"$\alpha_c\approx0.138$")
    ax_success.set_xlabel(r"$\alpha=P/N$")
    ax_success.set_ylabel("Retrieval success")
    ax_success.set_title("Chapter 1.1: Classical Hopfield threshold")
    ax_success.set_ylim(0.0, 1.02)
    ax_success.legend(frameon=False)
    save_fig(fig_success, RESULT_DIR / "ch01_success_curves.png")
    plt.close(fig_success)

    # Plot: finite-size scaling of Pc/N
    fig_finite, ax_finite = plt.subplots(figsize=(8.6, 5.2))
    ax_finite.plot(n_values, pc_over_n, "o-", color=cp["mixed"], label=r"$P_c/N$")
    ax_finite.axhline(0.138, ls="--", color=cp["theory"], label=r"asymptote 0.138")
    ax_finite.set_xlabel("N")
    ax_finite.set_ylabel(r"$P_c/N$")
    ax_finite.set_title("Finite-size scaling")
    ax_finite.legend(frameon=False)
    save_fig(fig_finite, RESULT_DIR / "ch01_finite_size.png")
    plt.close(fig_finite)

    alpha_update = data["alpha_update"]
    async_success = data["async_success"]
    sync_success = data["sync_success"]
    energy_trace = data["energy_trace_async"]

    # Plot: async vs sync success
    fig_update, ax_update = plt.subplots(figsize=(8.6, 5.2))
    ax_update.plot(alpha_update, async_success, "o-", color=cp["centered"], label="async")
    ax_update.plot(alpha_update, sync_success, "s--", color=cp["extra"], label="sync")
    ax_update.set_xlabel(r"$\alpha=P/N$")
    ax_update.set_ylabel("Retrieval success")
    ax_update.set_ylim(0.0, 1.02)
    ax_update.set_title("Chapter 1.2: async vs sync")
    ax_update.legend(frameon=False)
    save_fig(fig_update, RESULT_DIR / "ch01_async_vs_sync.png")
    plt.close(fig_update)

    # Plot: representative async energy trace
    if energy_trace.size > 0:
        fig_energy, ax_energy = plt.subplots(figsize=(8.6, 5.2))
        ax_energy.plot(np.arange(energy_trace.size), energy_trace, "o-", color=cp["mixed"])
        ax_energy.set_xlabel("Async sweep")
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("Representative async energy trace")
        save_fig(fig_energy, RESULT_DIR / "ch01_energy_trace.png")
        plt.close(fig_energy)


if __name__ == "__main__":
    main()
