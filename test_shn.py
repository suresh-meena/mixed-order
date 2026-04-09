import torch
import time
from shn_model import SimplicialHopfieldNetwork

def test_shn():
    N = 1000  # Testing with N=1000
    P = 10
    p = 0.5
    beta = 0.5
    lam = 2.0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate random patterns
    patterns = torch.randint(0, 2, (P, N), device=device) * 2 - 1
    
    # Initialize model
    model = SimplicialHopfieldNetwork(N, p, beta, lam, device=device)
    
    # Generate masks
    t0 = time.time()
    model.generate_masks()
    print(f"Mask generation time: {time.time() - t0:.4f}s")
    print(f"Number of triangles: {model.n_tri}")
    
    # Store patterns
    t0 = time.time()
    model.store_patterns(patterns)
    print(f"Pattern storage time: {time.time() - t0:.4f}s")
    
    # Test retrieval
    # Corrupt a pattern
    test_pattern = patterns[0].clone()
    flip_idx = torch.randperm(N)[:int(0.1 * N)]
    test_pattern[flip_idx] *= -1
    
    t0 = time.time()
    final_state = model.run(test_pattern, max_steps=20)
    print(f"Retrieval time (single): {time.time() - t0:.4f}s")
    
    overlap = torch.mean((final_state == patterns[0]).float()).item()
    print(f"Overlap with original pattern: {overlap:.4f}")

if __name__ == "__main__":
    test_shn()
