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

import os
import numpy as np
import matplotlib.ticker as mticker

from experiments.plot_helpers import apply_pub_style, edges_from_centers, plt
from mixed_order.theory import optimal_lambda

RESULT_DIR = os.path.dirname(__file__)


def _edges_from_centers(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return edges


def main() -> None:
    apply_pub_style()
    results = np.load(os.path.join(RESULT_DIR, "heatmap_p_lambda_results.npz"), allow_pickle=True)
    N = int(results["N"])
    beta = float(results["beta"])
    p_vals = results["p_vals"]
    lam_vals = results["lam_vals"]
    pc_anal = results["pc_anal"]
    pc_emp = results["pc_emp"]
    n_trials = int(results["n_trials"])
    n_seeds = int(results["n_seeds"])
    success_threshold = float(results["success_threshold"])

    p_line = np.linspace(0.05, 0.95, 300)
    lam_opt = np.array([optimal_lambda(pp, beta) for pp in p_line])
    in_range = (lam_opt >= lam_vals.min()) & (lam_opt <= lam_vals.max())

    p_edges = edges_from_centers(p_vals)
    lam_edges = edges_from_centers(lam_vals)
    vmax = max(float(pc_anal.max()), float(pc_emp.max()))

    # Analytical heatmap (single figure)
    fig_th, ax_th = plt.subplots(figsize=(8.6, 6.0))
    im_th = ax_th.pcolormesh(p_edges, lam_edges, pc_anal, cmap="plasma", shading="auto", vmin=0, vmax=vmax)
    ax_th.plot(p_line[in_range], lam_opt[in_range], color="white", lw=2.0, ls="--", label=r"$\lambda^*(p)=\tilde{q}/p$")
    ax_th.set_yscale("log")
    ax_th.set_ylim(lam_vals[0], lam_vals[-1])
    ax_th.set_xlim(p_edges[0], p_edges[-1])
    ax_th.set_xlabel(r"Edge density $p$")
    ax_th.set_ylabel(r"$\lambda = b/a$")
    ax_th.set_title("(a) Analytical replica theory", pad=7)
    ax_th.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}$"))
    ax_th.legend(loc="upper right", framealpha=0.75, labelcolor="white", facecolor="#333333", edgecolor="none", fontsize=10)
    cb_th = fig_th.colorbar(im_th, ax=ax_th, pad=0.02, fraction=0.046)
    cb_th.set_label(r"$P_c$  (critical capacity)", labelpad=8)
    fig_th.suptitle(f"Mixed-Order Hopfield Network capacity heatmap (N={N}, beta={beta})")
    path_th = os.path.join(RESULT_DIR, "heatmap_p_lambda_analytical.png")
    fig_th.savefig(path_th)
    plt.close(fig_th)
    print(f"saved plot to {path_th}")

    # Empirical heatmap (single figure)
    fig_emp, ax_emp = plt.subplots(figsize=(8.6, 6.0))
    im_emp = ax_emp.pcolormesh(p_edges, lam_edges, pc_emp, cmap="plasma", shading="auto", vmin=0, vmax=vmax)
    ax_emp.plot(p_line[in_range], lam_opt[in_range], color="white", lw=2.0, ls="--", label=r"$\lambda^*(p)=\tilde{q}/p$")
    ax_emp.set_yscale("log")
    ax_emp.set_ylim(lam_vals[0], lam_vals[-1])
    ax_emp.set_xlim(p_edges[0], p_edges[-1])
    ax_emp.set_xlabel(r"Edge density $p$")
    ax_emp.set_ylabel(r"$\lambda = b/a$")
    ax_emp.set_title("(b) Empirical simulation", pad=7)
    ax_emp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}$"))
    ax_emp.legend(loc="upper right", framealpha=0.75, labelcolor="white", facecolor="#333333", edgecolor="none", fontsize=10)
    cb_emp = fig_emp.colorbar(im_emp, ax=ax_emp, pad=0.02, fraction=0.046)
    cb_emp.set_label(r"$P_c$  (critical capacity)", labelpad=8)

    valid_mask = pc_anal > 0
    diff_pct = 100.0 * (pc_emp[valid_mask] - pc_anal[valid_mask]) / pc_anal[valid_mask]
    mape = float(np.mean(np.abs(diff_pct)))
    info_text = (
        f"MAPE vs theory: {mape:.1f}%\n"
        f"Seeds: {n_seeds}, Trials: {n_trials}\n"
        f"Succ. thresh: {success_threshold}"
    )
    ax_emp.text(
        0.03,
        0.04,
        info_text,
        transform=ax_emp.transAxes,
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec="none"),
    )

    fig_emp.suptitle(f"Mixed-Order Hopfield Network capacity heatmap (N={N}, beta={beta})")
    path_emp = os.path.join(RESULT_DIR, "heatmap_p_lambda_empirical.png")
    fig_emp.savefig(path_emp)
    plt.close(fig_emp)
    print(f"saved plot to {path_emp}")


if __name__ == "__main__":
    main()