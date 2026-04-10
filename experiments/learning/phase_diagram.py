import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style
from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.data.structured import generate_factors, compute_covariance, generate_structured_patterns
from mixed_order.metrics import seed_all, _DEVICE
from mixed_order.theory import compute_q_from_budget

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()

def measure_drift(N=100, D=50, p=0.35, beta=0.5, lam=2.0, P_max=50, n_trials=5, seed=42):
    """
    Measure average signal and drift amplitude as a function of P (load).
    Signal is E[h_i * xi_i], drift is the spurious local field from covariance C.
    """
    seed_all(seed)
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)
    
    q, _ = compute_q_from_budget(p, N, beta)
    model_uncentered = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    model_uncentered.generate_masks()
    
    patterns = generate_structured_patterns(P_max, F, device=_DEVICE)
    
    p_sweep = np.arange(1, P_max+1, max(1, P_max//20))
    signals_pw = []
    drifts_pw = []
    signals_3b = []
    
    for P in tqdm(p_sweep, desc=f"  Measuring Drift (lam={lam})", leave=False):
        model_uncentered.store_multiple_p(patterns, [P], centered=False)
        
        mu_indices = torch.arange(n_trials) % P
        targets = patterns[mu_indices].float() # (n_trials, N)
        states = targets.unsqueeze(0) # (1, n_trials, N)
        
        J = model_uncentered.storage.J[0] # (N, N)
        h_pw = torch.matmul(targets, J.T) # (n_trials, N)
        
        C_masked = C.to(_DEVICE) * model_uncentered.topology.c
        C_masked.fill_diagonal_(0.0)
        drift_field = (P / (p * N)) * torch.matmul(targets, C_masked.T) # (n_trials, N)
        
        sig_pw = (h_pw * targets).mean(dim=1).tolist()
        drift_amp = torch.abs(drift_field).mean(dim=1).tolist()
        
        if model_uncentered.topology.n_tri > 0:
            h_full = model_uncentered.local_field(states).squeeze(0) # (n_trials, N)
            h_3b = h_full - h_pw
            sig_3b = (h_3b * targets).mean(dim=1).tolist()
        else:
            sig_3b = [0.0] * n_trials
            
        signals_pw.append(np.mean(sig_pw))
        drifts_pw.append(np.mean(drift_amp))
        signals_3b.append(np.mean(sig_3b))
        
    return p_sweep, np.array(signals_pw), np.array(drifts_pw), np.array(signals_3b)

def phase_diagram():
    print("Computing signal vs drift phase diagram...")
    N, D = 200, 50
    P_max = 80
    p = 0.35
    beta = 0.5
    lam_vals = [0.0, 2.0, 10.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    
    for ax, lam in zip(axes, lam_vals):
        p_sweep, sig_pw, drifts_pw, sig_3b = measure_drift(N, D, p, beta, lam, P_max=P_max, n_trials=4)
        
        total_sig = sig_pw + sig_3b
        alpha_sweep = p_sweep / N
        
        ax.plot(alpha_sweep, total_sig, 'k-', lw=2, label="Total Signal")
        ax.plot(alpha_sweep, sig_pw, 'b--', label="Pairwise Signal")
        ax.plot(alpha_sweep, sig_3b, 'g-.', label="3-Body Signal")
        ax.plot(alpha_sweep, drifts_pw, 'r-', lw=2, label="|Drift|")
        
        # Heuristic crossing point
        diff = total_sig - drifts_pw
        crossings = np.where(diff < 0)[0]
        if len(crossings) > 0:
            idx = crossings[0]
            alpha_star = alpha_sweep[idx]
            ax.axvline(alpha_star, color='r', linestyle=':', label=rf"$\alpha^* \approx {alpha_star:.2f}$")
            
        ax.set_title(rf"$\lambda = {lam}$")
        ax.set_xlabel(r"Load $\alpha = P/N$")
        ax.set_ylabel("Amplitude")
        if ax == axes[0]:
            ax.legend()
            
    path = os.path.join(RESULT_DIR, "phase_diagram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved phase diagram to {path}")

if __name__ == "__main__":
    phase_diagram()
