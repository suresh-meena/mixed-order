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


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch04_ablation_matrix_results.npz")

    alpha = data["alpha_vals"]
    g2 = float(data["g2"][0])
    lam_mix = float(data["lam_mix"][0])
    alpha_star = {
        "pair_unc": float(data["alpha_star_pair_uncentered"][0]),
        "pair_cen": float(data["alpha_star_pair_centered"][0]),
        "mix_unc": float(data["alpha_star_mix_uncentered"][0]),
        "mix_cen": float(data["alpha_star_mix_centered"][0]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), constrained_layout=True)

    # Top-left: pairwise success.
    ax = axes[0, 0]
    ax.plot(alpha, data["success_pair_uncentered"], "o-", color=cp["uncentered"], label="uncentered")
    ax.plot(alpha, data["success_pair_centered"], "o-", color=cp["centered"], label="centered")
    ax.axhline(0.95, color=cp["theory"], ls="--", lw=1.2)
    ax.axvline(alpha_star["pair_unc"], color=cp["uncentered"], ls=":", lw=1.2)
    ax.axvline(alpha_star["pair_cen"], color=cp["centered"], ls=":", lw=1.2)
    ax.set_title("Pairwise")
    ax.set_xlabel(r"$\alpha=P/N$")
    ax.set_ylabel("Retrieval success")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False)

    # Top-right: mixed-order success.
    ax = axes[0, 1]
    ax.plot(alpha, data["success_mix_uncentered"], "o-", color=cp["uncentered"], label="uncentered")
    ax.plot(alpha, data["success_mix_centered"], "o-", color=cp["centered"], label="centered")
    ax.axhline(0.95, color=cp["theory"], ls="--", lw=1.2)
    ax.axvline(alpha_star["mix_unc"], color=cp["uncentered"], ls=":", lw=1.2)
    ax.axvline(alpha_star["mix_cen"], color=cp["centered"], ls=":", lw=1.2)
    ax.set_title(rf"Mixed-order ($\lambda_{{mix}}={lam_mix:.2f}$)")
    ax.set_xlabel(r"$\alpha=P/N$")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False)

    # Bottom-left: pairwise field decomposition.
    ax = axes[1, 0]
    ax.plot(alpha, data["signal_pair_uncentered"], "o-", color=cp["uncentered"], label="aligned field, uncentered")
    ax.plot(alpha, data["signal_pair_centered"], "o-", color=cp["centered"], label="aligned field, centered")
    ax.plot(alpha, data["drift_pair_uncentered"], "--", color=cp["extra"], lw=1.8, label="pairwise drift")
    ax.plot(alpha, np.zeros_like(alpha), "k:", lw=1.4, label="centered drift")
    ax.set_title("Pairwise field decomposition")
    ax.set_xlabel(r"$\alpha=P/N$")
    ax.set_ylabel("Field scale")
    ax.legend(frameon=False, fontsize=8)

    # Bottom-right: mixed-order field decomposition.
    ax = axes[1, 1]
    ax.plot(alpha, data["signal_mix_uncentered"], "o-", color=cp["uncentered"], label="aligned field, uncentered")
    ax.plot(alpha, data["signal_mix_centered"], "o-", color=cp["centered"], label="aligned field, centered")
    ax.plot(alpha, data["drift_mix_uncentered"], "--", color=cp["extra"], lw=1.8, label="pairwise drift")
    ax.plot(alpha, np.zeros_like(alpha), "k:", lw=1.4, label="centered drift")
    ax.set_title(rf"Mixed-order field decomposition ($\lambda_{{mix}}={lam_mix:.2f}$)")
    ax.set_xlabel(r"$\alpha=P/N$")
    ax.set_ylabel("Field scale")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(rf"Ablation matrix on structured data ($g_2={g2:.3f}$)", y=1.01)
    save_fig(fig, RESULT_DIR / "ch04_ablation_matrix.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.8, 5.0))
    labels = ["pairwise\nuncentered", "pairwise\ncentered", "mixed-order\nuncentered", "mixed-order\ncentered"]
    vals = [
        float(data["margin_pair_uncentered"][-1]),
        float(data["margin_pair_centered"][-1]),
        float(data["margin_mix_uncentered"][-1]),
        float(data["margin_mix_centered"][-1]),
    ]
    colors = [cp["uncentered"], cp["centered"], cp["uncentered"], cp["centered"]]
    ax2.bar(labels, vals, color=colors, alpha=0.9)
    ax2.set_ylabel(r"Aligned-field margin at highest $\alpha$")
    ax2.set_title(rf"High-load margin summary ($\lambda_{{mix}}={lam_mix:.2f}$)")
    ax2.set_ylim(0.0, max(vals) * 1.15)
    save_fig(fig2, RESULT_DIR / "ch04_ablation_matrix_summary.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
