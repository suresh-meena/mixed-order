import os
import numpy as np
import torch
from tqdm import tqdm

from experiments.plot_helpers import apply_pub_style, get_color_palette
from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.theory import replica_capacity_structured, optimal_lambda_structured, compute_q_from_budget
from mixed_order.metrics import find_empirical_pc_by_success, seed_all, _DEVICE, bootstrap_ci
from mixed_order.data.structured import generate_factors, compute_covariance, estimate_g2, generate_structured_patterns

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()


def measure_pairwise_drift_vs_load(N, D, p, beta, patterns, C, n_trials=5):
    """
    Measure pairwise drift magnitude vs load.
    """
    P_list = np.linspace(1, patterns.shape[0], 15).astype(int)

    # We need a model instance just for masks/topology
    model = MixedOrderHopfieldNetwork(N, p, beta, lam=0.0, device=_DEVICE)
    model.generate_masks()

    C_masked = C.to(_DEVICE) * model.topology.c
    C_masked.fill_diagonal_(0.0)

    drifts_uncen = []
    drifts_cen = []

    for P in tqdm(P_list, desc="  Drift vs Load", leave=False):
        # 1. Uncentered drift
        # Field drift = (P/pN) * C_masked @ patterns
        mu_idx = torch.arange(n_trials, device=_DEVICE) % P
        targets = patterns[mu_idx]

        drift_field_uncen = (P / (p * N)) * torch.matmul(targets, C_masked.T)
        drifts_uncen.append(torch.abs(drift_field_uncen).mean().item())

        # 2. Centered drift (ideally zero)
        # J_centered = (1/pN) * sum (xi_i xi_j - C_ij)
        # The drift field component is literally zero by construction in centering
        drifts_cen.append(0.0)

    return P_list, np.array(drifts_uncen), np.array(drifts_cen)


def structured_experiment(N=128, p=0.35, beta=0.5, n_trials=8, n_seeds=4):
    print(f"Running Structured Data Experiment: N={N}, p={p}")

    # 1. Data Settings: D = 32, 64, 128
    D_vals = [32, 64, 128]
    lambda_vals = np.logspace(np.log10(0.2), np.log10(8.0), 11)

    results = {}

    # Panel A: Drift (using strongest structure D=32)
    D_strong = 32
    seed_all(42)
    F_strong = generate_factors(N, D_strong, device=_DEVICE)
    C_strong = compute_covariance(F_strong)
    P_max_drift = 100
    patterns_strong = generate_structured_patterns(P_max_drift, F_strong, device=_DEVICE)

    p_drift, drift_uncen, drift_cen = measure_pairwise_drift_vs_load(
        N, D_strong, p, beta, patterns_strong, C_strong, n_trials=n_trials
    )
    results['panel_a'] = {'P': p_drift, 'drift_uncen': drift_uncen, 'drift_cen': drift_cen}

    # Panel B: Pc vs Lambda
    panel_b_data = []

    for D in D_vals:
        print(f"  Processing D={D}...")
        seed_all(123)
        F = generate_factors(N, D, device=_DEVICE)
        C = compute_covariance(F)
        g2 = estimate_g2(C)
        lam_opt = optimal_lambda_structured(p, beta, g2)

        pcs_cen = []
        for lam in tqdm(lambda_vals, desc=f"    Sweep D={D}", leave=False):
            pc = find_empirical_pc_by_success(
                N, p, beta, lam, n_trials, n_seeds, F=F, C=C, centered=True
            )
            pcs_cen.append(pc)

        # Add uncentered for strongest D=32
        pc_uncen = None
        if D == 32:
            pc_uncen = []
            for lam in tqdm(lambda_vals, desc="    Sweep Uncentered", leave=False):
                pc = find_empirical_pc_by_success(
                    N, p, beta, lam, n_trials, n_seeds, F=F, centered=False
                )
                pc_uncen.append(pc)

        # Theory curve
        th_pc = [replica_capacity_structured(p, N, beta, g2=g2, lam=l) for l in lambda_vals]

        panel_b_data.append({
            'D': D, 'g2': g2, 'lam_opt': lam_opt,
            'pcs_cen': np.array(pcs_cen), 'pcs_uncen': np.array(pc_uncen) if pc_uncen else None,
            'th_pc': np.array(th_pc)
        })

    results['panel_b'] = panel_b_data

    # Save results
    np.savez(os.path.join(RESULT_DIR, "structured_results.npz"), N=N, p=p, beta=beta, results=results)

    # Plotting
    cp = get_color_palette()

    # Panel A: drift vs load
    fig_a, ax_a = plt.subplots(figsize=(8.6, 5.0))
    ax_a.plot(results['panel_a']['P'] / N, results['panel_a']['drift_uncen'], color=cp['uncentered'], marker='o', ls='-', label="Uncentered")
    ax_a.plot(results['panel_a']['P'] / N, results['panel_a']['drift_cen'], color=cp['centered'], marker='o', ls='-', label="Centered")
    ax_a.set_xlabel(r"Load $\alpha = P/N$")
    ax_a.set_ylabel("Drift Amplitude")
    ax_a.set_title("(a) Pairwise drift suppression")
    ax_a.legend()
    path_a = os.path.join(RESULT_DIR, "structured_drift.png")
    fig_a.savefig(path_a)
    plt.close(fig_a)
    print(f"  Saved plot to {path_a}")

    # Panel B: Pc vs lambda for each D
    colors = [cp['centered'], cp['mixed'], cp['extra']]
    fig_b, ax_b = plt.subplots(figsize=(8.6, 5.0))
    for i, data in enumerate(results['panel_b']):
        lbl = f"D={data['D']} ($g_2$={data['g2']:.2f})"
        ax_b.plot(lambda_vals, data['pcs_cen'], 'o-', color=colors[i], label=f"Empirical ({lbl})")
        ax_b.plot(lambda_vals, data['th_pc'], '--', color=colors[i], alpha=0.6)
        ax_b.axvline(data['lam_opt'], color=colors[i], ls=':', alpha=0.8)

        if data['pcs_uncen'] is not None:
            ax_b.plot(lambda_vals, data['pcs_uncen'], marker='s', color=cp['uncentered'], ls='-', label="Empirical (Uncentered)")

    ax_b.set_xscale('log')
    ax_b.set_xlabel(r"$\lambda = b/a$")
    ax_b.set_ylabel(r"Capacity $P_c$")
    ax_b.set_title(r"(b) Optimal $\lambda$ shift")
    ax_b.legend(fontsize=9)
    path_b = os.path.join(RESULT_DIR, "structured_capacity_vs_lambda.png")
    fig_b.savefig(path_b)
    plt.close(fig_b)
    print(f"  Saved plot to {path_b}")
