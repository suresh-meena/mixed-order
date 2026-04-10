from __future__ import annotations

from pathlib import Path

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch05_results.npz")

    noise = data["noise_levels"]
    acc_pw = data["accuracy_pairwise"]
    acc_mx = data["accuracy_mixed"]
    basin_pw = data["basin_pairwise"]
    basin_mx = data["basin_mixed"]
    rep = data["representative_states"]

    # Accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(8.6, 5.0))
    ax_acc.plot(noise, acc_pw, "o--", color=cp["uncentered"], label="pairwise")
    ax_acc.plot(noise, acc_mx, "o-", color=cp["mixed"], label="mixed-order")
    ax_acc.axhline(0.5, ls=":", color=cp["theory"], label="chance")
    ax_acc.set_xlabel("cue corruption")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_title("Chapter 5.1 parity task")
    ax_acc.legend(frameon=False)
    save_fig(fig_acc, RESULT_DIR / "ch05_accuracy.png")
    plt.close(fig_acc)

    # Basin-size plot
    fig_basin, ax_basin = plt.subplots(figsize=(8.6, 5.0))
    ax_basin.plot(noise, basin_pw, "o--", color=cp["uncentered"], label="pairwise basin")
    ax_basin.plot(noise, basin_mx, "o-", color=cp["mixed"], label="mixed basin")
    ax_basin.set_xlabel("cue corruption")
    ax_basin.set_ylabel("retrieval success")
    ax_basin.set_ylim(0.0, 1.02)
    ax_basin.set_title("Basin-size comparison")
    ax_basin.legend(frameon=False)
    save_fig(fig_basin, RESULT_DIR / "ch05_basin_size.png")
    plt.close(fig_basin)

    # Representative panels: save each subplot as an individual figure
    labels = ["cue", "pairwise_final", "mixed_final", "target"]
    for i in range(rep.shape[0]):
        fig_r, ax_r = plt.subplots(figsize=(10.0, 2.0))
        ax_r.plot(rep[i], lw=1.2)
        ax_r.set_ylabel(labels[i])
        if i == rep.shape[0] - 1:
            ax_r.set_xlabel("dimension")
        fig_r.suptitle(f"Representative parity-case retrieval: {labels[i]}")
        save_fig(fig_r, RESULT_DIR / f"ch05_example_{labels[i]}.png")
        plt.close(fig_r)


if __name__ == "__main__":
    main()
