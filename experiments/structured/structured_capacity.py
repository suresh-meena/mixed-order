import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style
from mixed_order.theory import replica_capacity_structured, optimal_lambda_structured, compute_q_from_budget
from mixed_order.metrics import find_empirical_pc, _DEVICE
from mixed_order.data.structured import generate_factors, compute_covariance, estimate_g2

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()

def test_structured_capacity(N=100, D=50, beta=0.5, p=0.35, n_trials=4, n_seeds=3):
    print(f"Testing structured capacity: N={N}, D={D}, p={p}")
    
    # 1. Generate factors and compute covariance
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)
    g2 = estimate_g2(C)
    print(f"  Estimated g2: {g2:.4f}")
    
    # 2. Theoretical predictions
    _, qt = compute_q_from_budget(p, N, beta)
    lam_opt = optimal_lambda_structured(p, beta, g2)
    print(f"  Optimal lambda (structured): {lam_opt:.4f}")
    
    lam_sweep = np.logspace(-1, 1.5, 15)
    th_iid, th_struct, th_centered = [], [], []
    emp_uncen, emp_cen = [], []
    
    # 3. Empirical sweeps
    with tqdm(lam_sweep, desc="  Lambda sweep", ncols=90) as pbar:
        for lam in pbar:
            # Theory
            th_iid.append(replica_capacity_structured(p, N, beta, g2=0.0, lam=lam))
            # Without centering, the deterministic drift lowers capacity significantly. 
            # We don't have a formula for this implemented, but we can compute empirical.
            th_centered.append(replica_capacity_structured(p, N, beta, g2=g2, lam=lam))
            
            # Empirical Uncentered
            pc_uncen = find_empirical_pc(N, p, beta, lam, n_trials, n_seeds, pbar=pbar, F=F, centered=False)
            emp_uncen.append(pc_uncen)
            
            # Empirical Centered
            pc_cen = find_empirical_pc(N, p, beta, lam, n_trials, n_seeds, pbar=pbar, F=F, C=C, centered=True)
            emp_cen.append(pc_cen)
            
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(lam_sweep, th_iid, 'k--', label="Theory (IID)")
    ax.plot(lam_sweep, th_centered, 'b-', label=f"Theory (Structured, g2={g2:.3f})")
    
    ax.plot(lam_sweep, emp_uncen, 'ro-', label="Empirical (Uncentered)")
    ax.plot(lam_sweep, emp_cen, 'bs-', label="Empirical (Centered)")
    
    ax.axvline(lam_opt, color='g', linestyle=':', label="Optimal $\\lambda$")
    
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Critical Capacity $P_c$')
    ax.set_title(f'Structured Patterns (N={N}, D={D}, p={p})')
    ax.legend()
    
    path = os.path.join(RESULT_DIR, "structured_capacity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved plot to {path}")

if __name__ == "__main__":
    test_structured_capacity()
