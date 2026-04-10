from __future__ import annotations

import os
import numpy as np

from experiments.plot_helpers import apply_pub_style, get_color_palette, save_fig, plt

RESULT_DIR = os.path.dirname(__file__)


def main() -> None:
    apply_pub_style()
    loaded = np.load(os.path.join(RESULT_DIR, "structured_results.npz"), allow_pickle=True)
    N = int(loaded["N"]) if "N" in loaded.files else 128
    p = float(loaded["p"]) if "p" in loaded.files else 0.35
    results = loaded["results"].item()
    panel_a = results["panel_a"]
    panel_b = results["panel_b"]
    cp = get_color_palette()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax_a.plot(panel_a["P"] / N, panel_a["drift_uncen"], color=cp["uncentered"], marker="o", ls="-", label="Uncentered")
    ax_a.plot(panel_a["P"] / N, panel_a["drift_cen"], color=cp["centered"], marker="o", ls="-", label="Centered")
    ax_a.set_xlabel(r"Load $\alpha = P/N$")
    ax_a.set_ylabel("Drift Amplitude")
    ax_a.set_title("(a) Pairwise drift suppression")
    ax_a.legend()

    colors = [cp["centered"], cp["mixed"], cp["extra"]]
    lambda_vals = np.logspace(np.log10(0.2), np.log10(8.0), 11)
    for i, data_item in enumerate(panel_b):
        lbl = f"D={data_item['D']} ($g_2$={data_item['g2']:.2f})"
        ax_b.plot(lambda_vals, data_item["pcs_cen"], "o-", color=colors[i], label=f"Empirical ({lbl})")
        ax_b.plot(lambda_vals, data_item["th_pc"], "--", color=colors[i], alpha=0.6)
        ax_b.axvline(data_item["lam_opt"], color=colors[i], ls=":", alpha=0.8)
        if data_item["pcs_uncen"] is not None:
            ax_b.plot(lambda_vals, data_item["pcs_uncen"], marker="s", color=cp["uncentered"], ls="-", label="Empirical (Uncentered)")

    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"$\lambda = b/a$")
    ax_b.set_ylabel(r"Capacity $P_c$")
    ax_b.set_title(rf"(b) Optimal $\lambda$ shift  ($p={p}$)")
    ax_b.legend(fontsize=9)

    path = os.path.join(RESULT_DIR, "structured_capacity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    main()