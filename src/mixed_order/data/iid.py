import torch
import numpy as np

def seed_all(seed: int):
    """Seed both PyTorch and NumPy global RNGs from a single integer."""
    torch.manual_seed(seed)
    np.random.seed(int(seed) & 0xFFFF_FFFF)

def generate_iid_patterns(P: int, N: int, device: str = "cpu") -> torch.Tensor:
    """Generate independent, identically distributed Rademacher (+1, -1) patterns."""
    return torch.randint(0, 2, (P, N), device=device) * 2 - 1

def generate_noise_inits(targets: torch.Tensor, n_flip: int) -> torch.Tensor:
    """
    Generate initial states by flipping spins from a specific set of target patterns.
    
    Args:
        targets: (..., N) tensor of target patterns
        n_flip: number of spins to flip
        
    Returns:
        inits: (..., N) tensor with n_flip spins flipped per pattern
    """
    inits = targets.clone()
    shape = inits.shape
    N = shape[-1]
    inits_flat = inits.view(-1, N)
    for i in range(inits_flat.shape[0]):
        flip_idx = torch.randperm(N, device=targets.device)[:n_flip]
        inits_flat[i, flip_idx] *= -1
    return inits.view(shape)
