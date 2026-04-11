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
    data = load_npz(RESULT_DIR / "ch03_structured_compare_results.npz")

    alpha = data["alpha_vals"]
    g2 = float(data["g2"][0])
    lam_struct = float(data["lam_struct"][0])
    lam_iid = float(data["lam_iid"][0])
    lambda_vals = data["lambda_vals"]

    fig, (ax_succ, ax_field) = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    ax_succ.plot(alpha, data["success_pair_uncentered"], "o-", color=cp["uncentered"], lw=2.1, label="pairwise classical")
    ax_succ.plot(alpha, data["success_mix_uncentered"], "^-", color=cp["mixed"], lw=2.2, label=rf"mixed-order structured-optimal ($\lambda^*={lam_struct:.2f}$)")
    ax_succ.axhline(0.95, color=cp["theory"], ls="--", lw=1.2)
    ax_succ.set_xlabel(r"Load $\alpha=P/N$")
    ax_succ.set_ylabel("Retrieval success")
    ax_succ.set_ylim(0.0, 1.02)
    ax_succ.set_title(rf"Structured teacher comparison ($g_2={g2:.3f}$, iid $\lambda^*={lam_iid:.2f}$)")
    ax_succ.legend(frameon=False, fontsize=8)

    ax_field.plot(alpha, data["signal_pair_uncentered"], "o-", color=cp["uncentered"], label="pairwise signal")
    ax_field.plot(alpha, data["drift_pair_uncentered"], "--", color=cp["extra"], lw=1.8, label="pairwise drift")
    ax_field.plot(alpha, data["signal_mix_uncentered"], "o-", color=cp["mixed"], label="mixed-order signal")
    ax_field.plot(alpha, data["drift_mix_uncentered"], "--", color=cp["theory"], lw=1.8, label="mixed-order drift")
    ax_field.set_xlabel(r"Load $\alpha=P/N$")
    ax_field.set_ylabel("Aligned field scale")
    ax_field.set_title("Signal versus drift")
    ax_field.legend(frameon=False, fontsize=8, ncol=1)

    save_fig(fig, RESULT_DIR / "ch03_structured_compare.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.8, 5.0))
    labels = ["pairwise\nclassical", "mixed-order\nstructured-optimal"]
    vals = [
        float(data["alpha_star_pair_uncentered"][0]),
        float(data["alpha_star_mix_uncentered"][0]),
    ]
    colors = [cp["uncentered"], cp["mixed"]]
    ax2.bar(labels, vals, color=colors, alpha=0.9)
    ax2.set_ylabel(r"$\alpha^*_{learn}$")
    ax2.set_title(rf"Learning boundary summary ($\lambda^*={lam_struct:.2f}$)")
    ax2.set_ylim(0.0, max(vals) * 1.15)
    save_fig(fig2, RESULT_DIR / "ch03_structured_compare_summary.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8.8, 5.0))
    success_gap = data["success_mix_uncentered"] - data["success_pair_uncentered"]
    ax3.plot(alpha, success_gap, "o-", color=cp["mixed"], lw=2.2)
    ax3.axhline(0.0, color=cp["theory"], ls="--", lw=1.2)
    ax3.set_xlabel(r"Load $\alpha=P/N$")
    ax3.set_ylabel("Success gain")
    ax3.set_title("Mixed-order advantage over pairwise")
    ax3.set_ylim(min(-0.05, float(np.min(success_gap)) * 1.1), max(0.05, float(np.max(success_gap)) * 1.1))
    save_fig(fig3, RESULT_DIR / "ch03_structured_compare_gap.png")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9.0, 5.4))
    alpha_edges = np.concatenate(
        [
            [alpha[0] - 0.5 * (alpha[1] - alpha[0])],
            0.5 * (alpha[:-1] + alpha[1:]),
            [alpha[-1] + 0.5 * (alpha[-1] - alpha[-2])],
        ]
    )
    lambda_edges = np.concatenate(
        [
            [lambda_vals[0] - 0.5 * (lambda_vals[1] - lambda_vals[0])],
            0.5 * (lambda_vals[:-1] + lambda_vals[1:]),
            [lambda_vals[-1] + 0.5 * (lambda_vals[-1] - lambda_vals[-2])],
        ]
    )
    hm = ax4.pcolormesh(
        alpha_edges,
        lambda_edges,
        data["success_lambda_grid"],
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    ax4.axhline(lam_struct, color=cp["mixed"], ls="-", lw=2.4, label=rf"structured $\lambda^*={lam_struct:.2f}$")
    ax4.set_xlabel(r"Load $\alpha=P/N$")
    ax4.set_ylabel(r"$\lambda$")
    ax4.set_title("Mixed-order success heatmap")
    cbar = fig4.colorbar(hm, ax=ax4, pad=0.02)
    cbar.set_label("Retrieval success")
    ax4.legend(frameon=False, fontsize=8, loc="lower left")
    save_fig(fig4, RESULT_DIR / "ch03_structured_compare_heatmap.png")
    plt.close(fig4)


if __name__ == "__main__":
    main()
