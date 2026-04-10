import pytest
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.config import NetworkConfig
from mixed_order.topology import Topology
from mixed_order.data.structured import generate_factors, compute_covariance, generate_structured_patterns
from mixed_order.metrics import seed_all

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_topology_properties():
    N = 50
    p = 0.5
    beta = 0.5
    lam = 2.0
    
    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=_DEVICE)
    topology = Topology(config)
    topology.generate_masks()
    
    # 1. Symmetry
    c = topology.c
    assert torch.allclose(c, c.T), "Pairwise mask c is not symmetric"
    
    # 2. Zero diagonal
    diag = torch.diagonal(c)
    assert torch.all(diag == 0), "Pairwise mask c has non-zero diagonal"
    
    # 3. Pairwise budget
    M_emp = c.sum().item() / 2
    M_th = 0.5 * p * N**2
    # It should be close (binomial sampling variance is p(1-p))
    assert abs(M_emp - M_th) < 3 * np.sqrt(M_th), "Pairwise budget significantly deviates from expectation"
    
    # 4. Triangle counts
    n_tri_emp = topology.n_tri
    q_tilde = 6 * beta - 3 * p
    q = max(q_tilde / N, 0.0)
    M3_th = (q * N**3) / 6.0
    if q > 0:
        assert abs(n_tri_emp - M3_th) < 3 * np.sqrt(M3_th), "3-body budget significantly deviates from expectation"

def test_storage_centering():
    N = 100
    D = 50
    P = 10
    p = 1.0 # Fully connected to avoid masking variance in the test
    beta = 0.5
    lam = 0.0 # Just test 2-body
    
    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=_DEVICE)
    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    model.generate_masks()
    
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)
    patterns = generate_structured_patterns(P, F, device=_DEVICE)
    
    # Uncentered
    model.store_multiple_p(patterns, [P], centered=False)
    J_uncentered = model.storage.J[0]
    
    # Centered
    model.store_multiple_p(patterns, [P], centered=True, C=C)
    J_centered = model.storage.J[0]
    
    # The expected uncentered J_ij is approximately (P / pN) * C_ij
    # So the mean of J_centered should be closer to 0 than J_uncentered (relative to C)
    mean_uncentered_drift = torch.mean(J_uncentered * C).item()
    mean_centered_drift = torch.mean(J_centered * C).item()
    
    assert abs(mean_centered_drift) < abs(mean_uncentered_drift), "Centering did not reduce covariance drift"
    assert abs(mean_centered_drift) < 1e-2, "Centered J has too much residual correlation with C"

def test_structured_data_moments():
    N = 100
    D = 50
    P = 1000 # Large P to measure empirical covariance
    F = generate_factors(N, D, device=_DEVICE)
    C_th = compute_covariance(F)
    
    patterns = generate_structured_patterns(P, F, device=_DEVICE)
    C_emp = (patterns.T @ patterns) / P
    
    # Check if empirical covariance matches theoretical C
    error = torch.mean(torch.abs(C_emp - C_th)).item()
    assert error < 0.05, f"Empirical covariance deviates from theoretical (MAE={error:.4f})"

import time
def test_baseline_capacity():
    t0 = time.time()
    N = 100
    p = 0.5
    beta = 0.5
    lam = 2.0
    p_list = [5, 10, 15]
    n_trials = 4
    
    seed_all(42)
    patterns = torch.randint(0, 2, (max(p_list), N), device=_DEVICE) * 2 - 1
    
    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    model.generate_masks()
    model.store_multiple_p(patterns, p_list)
    
    B = len(p_list)
    inits = torch.empty(B, n_trials, N, device=_DEVICE)
    targets = torch.empty(B, n_trials, N, device=_DEVICE)
    
    for b, P in enumerate(p_list):
        for trial in range(n_trials):
            mu = trial % P
            target = patterns[mu].float()
            flip_idx = torch.randperm(N)[:int(0.1 * N)]
            init = target.clone()
            init[flip_idx] *= -1
            inits[b, trial] = init
            targets[b, trial] = target
            
    final_states = model.run(inits, max_steps=20)
    overlaps = (final_states * targets).sum(dim=2) / N
    mean_overlaps = overlaps.mean(dim=1)
    
    # Capacity 5 and 10 should be perfect (overlap = 1.0)
    assert mean_overlaps[0].item() >= 0.99
    assert mean_overlaps[1].item() >= 0.99
    print(f"Test took {time.time() - t0:.4f} seconds")
if __name__ == "__main__":
    test_baseline_capacity()
