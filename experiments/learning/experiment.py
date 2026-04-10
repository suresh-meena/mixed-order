import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.optimize import curve_fit

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style, get_color_palette
from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.data.structured import generate_factors, compute_covariance, generate_structured_patterns
from mixed_order.metrics import seed_all, _DEVICE, compute_overlap

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()


def measure_drift_and_signal(N, D, p, beta, lam, P_max, n_trials=12, seed=42):
    """
    Measure total signal and drift amplitude across load P.
    """
    seed_all(seed)
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)

    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    model.generate_masks()

    patterns = generate_structured_patterns(P_max, F, device=_DEVICE)

    p_sweep = np.unique(np.linspace(1, P_max, 15).astype(int))
    signals = []
    drifts = []

    # Pre-mask C for drift calculation
    C_masked = C.to(_DEVICE) * model.topology.c
    C_masked.fill_diagonal_(0.0)

    for P in tqdm(p_sweep, desc="P sweep", leave=False):
        # Uncentered storage
        model.store_multiple_p(patterns, [P], centered=False)

        # Batch trials
        mu_indices = torch.arange(n_trials, device=_DEVICE) % P
        targets = patterns[mu_indices]
        states = targets.unsqueeze(0) # (1, n_trials, N)

        # 1. Total Signal E[h_i * xi_i]
        h_full = model.local_field(states).squeeze(0) # (n_trials, N)
        sig = (h_full * targets).mean().item()

        # 2. Drift Amplitude E[|drift_i|]
        # drift_field = (P/pN) * C_masked @ xi
        drift_field = (P / (p * N)) * torch.matmul(targets, C_masked.T)
        drift_amp = torch.abs(drift_field).mean().item()

        signals.append(sig)
        drifts.append(drift_amp)

    return p_sweep / N, np.array(signals), np.array(drifts)


def learning_boundary_experiment(N=1000, p=0.35, beta=0.5, D=256, n_trials=12, n_seeds=5):
    print(f"Running Learning Boundary Experiment: N={N}, D={D}")

    lambda_vals = np.linspace(0.0, 6.0, 9)
    P_max = 120 # Higher P_max to ensure crossing

    alpha_crossings = []
    alpha_cis = []

    # Shared factors for all lambda to keep g2 consistent
    seed_all(42)
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)
    g2 = ( (C.clone().fill_diagonal_(0.0)**2).sum() / N ).item()
    print(f"  System g2: {g2:.4f}")

    for lam in tqdm(lambda_vals, desc="  Lambda Sweep"):
        seed_crossings = []
        for seed in tqdm(range(n_seeds), desc="seeds", leave=False):
            alpha_sweep, signals, drifts = measure_drift_and_signal(
                N, D, p, beta, lam, P_max, n_trials=n_trials, seed=seed
            )

            # Find crossing: first alpha where signal <= drift
            # We use interpolation for a smoother boundary
            diff = signals - drifts
            crossing = None
            for i in range(len(diff)-1):
                if diff[i] > 0 and diff[i+1] <= 0:
                    # Linear interpolation
                    frac = diff[i] / (diff[i] - diff[i+1])
                    crossing = alpha_sweep[i] + frac * (alpha_sweep[i+1] - alpha_sweep[i])
                    break

            if crossing is None:
                # Right-censored: crossing is beyond P_max
                seed_crossings.append(alpha_sweep[-1])
            else:
                seed_crossings.append(crossing)

        alpha_crossings.append(np.mean(seed_crossings))
        # Simple std dev for error bars
        alpha_cis.append(np.std(seed_crossings))

    alpha_crossings = np.array(alpha_crossings)
    alpha_cis = np.array(alpha_cis)

    # Theory overlay: alpha* = c * (1 + lambda/2) / sqrt(g2)
    def scaling_law(lam, c):
        return c * (1 + lam / 2.0) / np.sqrt(g2)

    popt, _ = curve_fit(scaling_law, lambda_vals, alpha_crossings)
    c_fitted = popt[0]
    theory_overlay = scaling_law(lambda_vals, c_fitted)

    # Save raw results
    np.savez(os.path.join(RESULT_DIR, "learning_boundary_results.npz"),
             N=N, p=p, beta=beta, D=D, n_trials=n_trials, n_seeds=n_seeds,
             lambda_vals=lambda_vals, alpha_crossings=alpha_crossings,
             alpha_cis=alpha_cis, c_fitted=c_fitted, g2=g2)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    cp = get_color_palette()

    ax.errorbar(lambda_vals, alpha_crossings, yerr=alpha_cis, fmt='o', color=cp['mixed'], capsize=5, label="Measured Crossing")
    ax.plot(lambda_vals, theory_overlay, color=cp['theory'], lw=2, ls='--',
            label=rf"Scaling: ${c_fitted:.2f} \cdot (1+\lambda/2)/\sqrt{{g_2}}$")

    ax.set_xlabel(r"Coupling ratio $\lambda = b/a$")
    ax.set_ylabel(r"Learning boundary $\alpha^*$")
    ax.set_title(rf"Storage-to-Learning Transition ($N={N}, g_2={g2:.2f}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULT_DIR, "learning_boundary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved plot to {path}")
