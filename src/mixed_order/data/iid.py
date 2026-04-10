import torch
from mixed_order.utils import seed_all

def generate_iid_patterns(P: int, N: int, device: str = "cpu", generator: torch.Generator = None) -> torch.Tensor:
    """Generate independent, identically distributed Rademacher (+1, -1) patterns."""
    return torch.randint(0, 2, (P, N), device=device, generator=generator) * 2 - 1

def generate_noise_inits(targets: torch.Tensor, n_flip: int, generator: torch.Generator = None) -> torch.Tensor:
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

    if n_flip <= 0:
        return inits
    if n_flip >= N:
        return -inits

    inits_flat = inits.view(-1, N)
    scores = torch.rand((inits_flat.shape[0], N), device=targets.device, generator=generator)
    flip_idx = torch.topk(scores, n_flip, dim=-1, largest=False).indices
    flipped = -inits_flat.gather(1, flip_idx)
    inits_flat.scatter_(1, flip_idx, flipped)
    return inits.view(shape)
