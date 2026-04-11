import os, sys
import numpy as np
import matplotlib.ticker as mticker
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from mixed_order.plotting.style import apply_pub_style
from mixed_order.theory import compute_q_from_budget, optimal_lambda, replica_capacity
from mixed_order.metrics import find_empirical_pc_by_success, _DEVICE

# Results will be saved in the same folder as the script
RESULT_DIR = os.path.dirname(__file__)

apply_pub_style()


def _evaluate_cell(args):
    i, j, lam, p, N, beta, n_trials, n_seeds, success_threshold, overlap_threshold, device, topology_device = args
    _, qt = compute_q_from_budget(p, N, beta)
    if qt < 0.01:
        return i, j, 0.0

    pc = find_empirical_pc_by_success(
        N, p, beta, lam, n_trials, n_seeds,
        success_threshold=success_threshold,
        overlap_threshold=overlap_threshold,
        device=device,
        topology_device=topology_device,
    )
    return i, j, float(pc)


def heatmap_p_lambda(N, beta, n_p, n_lam, n_trials, n_seeds,
                     success_threshold=0.99, overlap_threshold=0.99,
                     device=None, topology_device=None, n_jobs=1,
                     seed_batch_size=None, triangle_chunk_size=8192):
    if device is None:
        device = _DEVICE
    if topology_device is None:
        topology_device = device

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

    # Parallel path is CPU-only. CUDA + multiprocessing often hurts due to
    # context startup and memory duplication.
    use_parallel = (n_jobs is not None) and (int(n_jobs) > 1) and (device == "cpu")

    if use_parallel:
        task_args = [
            (i, j, lam, p, N, beta, n_trials, n_seeds,
             success_threshold, overlap_threshold, device, topology_device)
            for i, j, lam, p in cells
        ]
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            for i, j, pc in tqdm(ex.map(_evaluate_cell, task_args), total=len(task_args), desc="  Heatmap", unit="cell", ncols=90):
                pc_emp[i, j] = pc
    else:
        # Shared cache for topology masks to avoid recomputing expensive
        # triangle/edge masks for identical (N, p, beta, seed) tuples.
        topology_cache = {}
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
                    device=device,
                    topology_cache=topology_cache,
                    topology_device=topology_device,
                    seed_batch_size=seed_batch_size,
                    triangle_chunk_size=triangle_chunk_size,
                )

    # Save raw results
    results_path = os.path.join(RESULT_DIR, "heatmap_p_lambda_results.npz")
    np.savez(results_path, 
             N=N, beta=beta, p_vals=p_vals, lam_vals=lam_vals,
             n_trials=n_trials, n_seeds=n_seeds,
             device=device, topology_device=topology_device,
             n_jobs=int(n_jobs),
             success_threshold=success_threshold,
             overlap_threshold=overlap_threshold,
             pc_anal=pc_anal, pc_emp=pc_emp)
    print(f"  Saved raw results to {results_path}")

    # ── Build plot arrays ─────────────────────────────────────────────────────
    p_line   = np.linspace(0.05, 0.95, 300)
    lam_opt  = np.array([optimal_lambda(pp, beta) for pp in p_line])
    in_range = (lam_opt >= lam_vals.min()) & (lam_opt <= lam_vals.max())

    # Use centralized helper
    _edges_from_centers = edges_from_centers

    p_edges   = _edges_from_centers(p_vals)
    lam_edges = _edges_from_centers(lam_vals)

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
    path_th = os.path.join(RESULT_DIR, "heatmap_p_lambda_analytical.png")
    fig_th.savefig(path_th)
    plt.close(fig_th)
    print(f"  Saved {path_th}")

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

    path_emp = os.path.join(RESULT_DIR, "heatmap_p_lambda_empirical.png")
    fig_emp.savefig(path_emp)
    plt.close(fig_emp)
    print(f"  Saved {path_emp}")
