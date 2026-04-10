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


@pytest.fixture(autouse=True)
def deterministic_seed():
    # Make tests reproducible across runs/machines
    seed_all(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Best-effort: if not supported, continue without failing
        pass
    yield


def test_topology_properties():
    N = 50
    p = 0.5
    beta = 0.5
    lam = 2.0
    
    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=_DEVICE)
    topology = Topology(config)
    topology.generate_masks()
    
    c = topology.c
    assert torch.allclose(c, c.T, atol=1e-6), "Pairwise mask c is not symmetric"
    
    diag = torch.diagonal(c)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6), "Pairwise mask c has non-zero diagonal"
    
    # Pairwise budget (count undirected edges)
    n_pairs = N * (N - 1) / 2
    M_emp = c.sum().item() / 2
    M_th = p * n_pairs
    std = np.sqrt(M_th * (1 - p))
    tol = max(4.0 * std, 1.0)
    assert abs(M_emp - M_th) < tol, f"Pairwise budget deviates (emp={M_emp}, th={M_th}, tol={tol})"
    
    # Triangle counts
    n_tri_emp = topology.n_tri
    q_tilde = 6 * beta - 3 * p
    q = max(q_tilde / N, 0.0)
    M3_th = (q * N**3) / 6.0
    if M3_th > 0:
        tol_tri = max(4.0 * np.sqrt(M3_th), 5.0)
        assert abs(n_tri_emp - M3_th) < tol_tri, f"3-body budget deviates (emp={n_tri_emp}, th={M3_th}, tol={tol_tri})"
    else:
        assert n_tri_emp == 0


def test_storage_centering():
    N = 100
    D = 50
    P = 10
    p = 1.0
    beta = 0.5
    lam = 0.0
    
    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=_DEVICE)
    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    model.generate_masks()
    
    F = generate_factors(N, D, device=_DEVICE)
    C = compute_covariance(F)
    patterns = generate_structured_patterns(P, F, device=_DEVICE)
    
    model.store_multiple_p(patterns, [P], centered=False)
    J_uncentered = model.storage.J[0]
    
    model.store_multiple_p(patterns, [P], centered=True, C=C)
    J_centered = model.storage.J[0]
    
    mean_uncentered_drift = torch.mean(J_uncentered * C).item()
    mean_centered_drift = torch.mean(J_centered * C).item()
    
    assert abs(mean_centered_drift) < abs(mean_uncentered_drift) + 1e-12, "Centering did not reduce covariance drift"
    denom = max(abs(mean_uncentered_drift), 1e-8)
    assert abs(mean_centered_drift) < max(1e-2, 0.25 * denom), "Centered J has too much residual correlation with C"


def test_structured_data_moments():
    N = 100
    D = 50
    P = 1000
    F = generate_factors(N, D, device=_DEVICE)
    C_th = compute_covariance(F)
    
    patterns = generate_structured_patterns(P, F, device=_DEVICE)
    C_emp = (patterns.T @ patterns) / P
    
    error = torch.mean(torch.abs(C_emp - C_th)).item()
    assert error < 0.05, f"Empirical covariance deviates from theoretical (MAE={error:.4f})"


def test_baseline_capacity():
    N = 100
    p = 0.5
    beta = 0.5
    lam = 2.0
    p_list = [5, 10, 15]
    n_trials = 4
    
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
    
    assert mean_overlaps[0].item() >= 0.98
    assert mean_overlaps[1].item() >= 0.98
