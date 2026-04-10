import torch
import math

def generate_factors(N: int, D: int, device: str = "cpu") -> torch.Tensor:
    """Generate fixed factor matrix F with normalized rows."""
    F = torch.randn(N, D, device=device)
    F = F / F.norm(dim=1, keepdim=True)
    return F

def generate_structured_patterns(P: int, F: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Generate patterns using Gaussian-sign teacher: xi_i = sign(F_i * z / sqrt(D))"""
    N, D = F.shape
    z = torch.randn(P, D, device=device)
    # F: (N, D), z: (P, D) -> z @ F.T is (P, N)
    h = z @ F.T / math.sqrt(D)
    
    # Handle 0s by defaulting to 1 (like Rademacher)
    patterns = torch.sign(h)
    patterns[patterns == 0] = 1.0
    return patterns

def compute_covariance(F: torch.Tensor) -> torch.Tensor:
    """Compute exact covariance matrix C_ij = (2/pi) * arcsin(F_i * F_j)"""
    inner_prods = F @ F.T
    # clamp to avoid NaNs from floating point issues
    inner_prods = torch.clamp(inner_prods, -1.0, 1.0)
    return (2.0 / math.pi) * torch.arcsin(inner_prods)

def estimate_g2(C: torch.Tensor) -> float:
    """Estimate g2 = (1/N) * sum_{j!=l} C_jl^2"""
    N = C.shape[0]
    C_no_diag = C.clone()
    C_no_diag.fill_diagonal_(0.0)
    return (C_no_diag ** 2).sum().item() / N
