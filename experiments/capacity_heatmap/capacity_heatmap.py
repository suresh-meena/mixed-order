import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

# Add parent directory of 'experiments' to sys.path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style
from mixed_order.theory import compute_q_from_budget, optimal_lambda, replica_capacity
from mixed_order.metrics import find_empirical_pc_by_success, _DEVICE

# Results will be saved in the same folder as the script
RESULT_DIR = os.path.dirname(__file__)

apply_pub_style()

def heatmap_p_lambda(N, beta, n_p, n_lam, n_trials, n_seeds, success_threshold=0.9, overlap_threshold=0.95):
    alpha_c  = 0.138
    p_vals   = np.linspace(0.15, 0.75, n_p)
    lam_vals = np.logspace(np.log10(0.2), np.log10(6.0), n_lam)

    # Analytical prediction
    p_g, lam_g = np.meshgrid(p_vals, lam_vals)
    q_g        = 6.0 * beta - 3.0 * p_g
    valid      = (p_g > 0) & (q_g > 0)
    pc_anal    = np.where(
        valid,
        replica_capacity(p_g, N, beta, alpha_c=alpha_c, lam=lam_g),
        0.0,
    )

    # Empirical grid
    pc_emp = np.zeros_like(pc_anal)
    cells  = [
        (i, j, float(lam), float(p))
        for i, lam in enumerate(lam_vals)
        for j, p   in enumerate(p_vals)
    ]

    with tqdm(cells, desc="  Heatmap", unit="cell", ncols=90) as pbar:
        for i, j, lam, p in pbar:
            _, qt = compute_q_from_budget(p, N, beta)
            if qt < 0.01:
                continue
            pc_emp[i, j] = find_empirical_pc_by_success(
                N, p, beta, lam, n_trials, n_seeds, 
                success_threshold=success_threshold,
                overlap_threshold=overlap_threshold,
                pbar=pbar,
            )

    # Save raw results
    results_path = os.path.join(RESULT_DIR, "heatmap_p_lambda_results.npz")
    np.savez(results_path, 
             N=N, beta=beta, p_vals=p_vals, lam_vals=lam_vals,
             n_trials=n_trials, n_seeds=n_seeds,
             success_threshold=success_threshold,
             overlap_threshold=overlap_threshold,
             pc_anal=pc_anal, pc_emp=pc_emp)
    print(f"  Saved raw results to {results_path}")

    # ── Build plot arrays ─────────────────────────────────────────────────────
    p_line   = np.linspace(0.05, 0.95, 300)
    lam_opt  = np.array([optimal_lambda(pp, beta) for pp in p_line])
    in_range = (lam_opt >= lam_vals.min()) & (lam_opt <= lam_vals.max())

    def _edges_from_centers(vals):
        vals    = np.asarray(vals, dtype=float)
        edges   = np.empty(vals.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
        edges[0]    = vals[0]  - 0.5 * (vals[1]  - vals[0])
        edges[-1]   = vals[-1] + 0.5 * (vals[-1] - vals[-2])
        return edges

    p_edges   = _edges_from_centers(p_vals)
    lam_edges = _edges_from_centers(lam_vals)

    vmax = max(float(pc_anal.max()), float(pc_emp.max()))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, squeeze=False)
    ax_theory = axes[0, 0]
    ax_emp = axes[0, 1]

    panel_info = [
        (ax_theory, pc_anal, "(a) Analytical replica theory"),
        (ax_emp, pc_emp,  "(b) Empirical simulation"),
    ]
    for ax, data, title in panel_info:
        im = ax.pcolormesh(
            p_edges, lam_edges, data,
            cmap="plasma", shading="auto",
            vmin=0, vmax=vmax,
        )
        ax.plot(p_line[in_range], lam_opt[in_range],
                color="white", lw=2.0, ls="--",
                label=r"$\lambda^*(p)=\tilde{q}/p$")
        ax.set_yscale("log")
        ax.set_ylim(lam_vals[0], lam_vals[-1])
        ax.set_xlim(p_edges[0],  p_edges[-1])
        ax.set_xlabel(r"Edge density $p$")
        ax.set_ylabel(r"$\lambda = b/a$")
        ax.set_title(title, pad=7)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"${y:g}$"))
        ax.legend(loc="upper right", framealpha=0.75,
                  labelcolor="white", facecolor="#333333",
                  edgecolor="none", fontsize=10)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(r"$P_c$  (critical capacity)", labelpad=8)

    # Annotate empirical panel
    valid_mask = pc_anal > 0
    diff_pct = 100.0 * (pc_emp[valid_mask] - pc_anal[valid_mask]) / pc_anal[valid_mask]
    mape  = float(np.mean(np.abs(diff_pct)))
    
    info_text = (
        f"MAPE vs theory: {mape:.1f}%\n"
        f"Seeds: {n_seeds}, Trials: {n_trials}\n"
        f"Succ. thresh: {success_threshold}"
    )
    ax_emp.text(
        0.03, 0.04,
        info_text,
        transform=ax_emp.transAxes, fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec="none"),
    )

    fig.suptitle(
        rf"Mixed-Order Hopfield Network capacity heatmap  ($N={N},\;\beta={beta}$)",
    )

    path = os.path.join(RESULT_DIR, "heatmap_p_lambda.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")

if __name__ == "__main__":
    # Parameters from Implementation Plan Step 5.5
    N = 256
    beta = 0.5
    n_p = 8
    n_lam = 10
    n_trials = 16
    n_seeds = 8
    
    heatmap_p_lambda(N, beta, n_p, n_lam, n_trials, n_seeds)
