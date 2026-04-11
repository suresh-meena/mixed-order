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
    data = load_npz(RESULT_DIR / "ch03_structured_frontier_results.npz")

    g2 = data["g2_vals"]
    alpha_pair = data["alpha_pair"]
    alpha_mix = data["alpha_mix"]
    alpha_pair_th = data["alpha_pair_theory"]
    alpha_mix_th = data["alpha_mix_theory"]
    lam_struct = data["lam_struct_vals"]

    fig, (ax_frontier, ax_gain) = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    ax_frontier.plot(g2, alpha_pair, "o", color=cp["uncentered"], ms=6.5, label="pairwise empirical")
    ax_frontier.plot(g2, alpha_mix, "o", color=cp["mixed"], ms=6.5, label="mixed-order empirical")
    ax_frontier.plot(g2, alpha_pair_th, "-", color=cp["uncentered"], lw=2.0, alpha=0.75, label="pairwise theory")
    ax_frontier.plot(g2, alpha_mix_th, "-", color=cp["mixed"], lw=2.0, alpha=0.75, label="mixed-order theory")
    ax_frontier.set_xscale("log")
    ax_frontier.set_xlabel(r"Covariance strength $g_2$")
    ax_frontier.set_ylabel(r"Critical load $\alpha^* = P_c/N$")
    ax_frontier.set_title("Structured capacity frontier")
    ax_frontier.legend(frameon=False, fontsize=8)

    ax_gain.plot(g2, data["alpha_gain"], "o-", color=cp["mixed"], lw=2.2, label="empirical gain")
    ax_gain.plot(g2, data["alpha_gain_theory"], "--", color=cp["theory"], lw=2.0, label="theory gain")
    ax_gain.axhline(0.0, color="black", lw=0.9, alpha=0.5)
    ax_gain.set_xscale("log")
    ax_gain.set_xlabel(r"Covariance strength $g_2$")
    ax_gain.set_ylabel(r"$\alpha^*_{\mathrm{mix}} - \alpha^*_{\mathrm{pair}}$")
    ax_gain.set_title(rf"Mixed-order advantage ($\lambda^* \uparrow$ with $g_2$)")
    ax_gain.legend(frameon=False, fontsize=8)

    save_fig(fig, RESULT_DIR / "ch03_structured_frontier.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.7, 5.0))
    ax2.plot(g2, lam_struct, "o-", color=cp["mixed"], lw=2.2)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Covariance strength $g_2$")
    ax2.set_ylabel(r"Structured optimum $\lambda^*$")
    ax2.set_title("Optimal mixed-order ratio grows with covariance")
    save_fig(fig2, RESULT_DIR / "ch03_structured_frontier_lambda.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()

