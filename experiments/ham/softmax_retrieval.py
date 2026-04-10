import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style
from mixed_order.config import NetworkConfig
from mixed_order.topology import Topology
from mixed_order.ham import SparseMixedOrderHAM
from mixed_order.data.iid import generate_iid_patterns
from mixed_order.metrics import seed_all, _DEVICE

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()

def test_softmax_retrieval(N=200, P=20, p=0.35, beta=0.5, lam=2.0, seed=42):
    print(f"Testing HAM softmax retrieval: N={N}, P={P}, lambda={lam}")
    seed_all(seed)
    
    # Setup core config and topology
    config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=_DEVICE)
    topology = Topology(config)
    topology.generate_masks()
    
    # Setup HAM layer
    ham = SparseMixedOrderHAM(config, topology)
    
    # Store random patterns
    patterns = generate_iid_patterns(P, N, device=_DEVICE).float()
    ham.store(patterns)
    
    # Generate corrupted initial states
    # Flip 20% of spins for each pattern
    n_flip = int(0.2 * N)
    inits = patterns.clone()
    
    # Corrupt
    for b in range(P):
        flip_idx = torch.randperm(N)[:n_flip]
        inits[b, flip_idx] *= -1
        
    print("Initial overlap with targets:")
    init_overlaps = (inits * patterns).mean(dim=1).cpu().numpy()
    print(f"  Mean: {init_overlaps.mean():.3f}")
    
    # Run continuous retrieval at finite temperature
    beta_temp = 5.0
    finals = ham.retrieve(inits, beta_temp=beta_temp, max_steps=20)
    
    print("Final overlap with targets (continuous state projected to binary):")
    finals_binary = torch.sign(finals)
    # Handle zeros
    finals_binary[finals_binary == 0] = 1.0
    
    final_overlaps = (finals_binary * patterns).mean(dim=1).cpu().numpy()
    print(f"  Mean: {final_overlaps.mean():.3f}")
    
    # Plotting retrieval trajectory (Optional, just checking success here)
    if final_overlaps.mean() > 0.99:
        print("Retrieval successful.")
    else:
        print("Retrieval failed.")

if __name__ == "__main__":
    test_softmax_retrieval()
