import torch
import numpy as np
from typing import Optional, List, Tuple, Union

from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.theory import compute_q_from_budget
from mixed_order.data.structured import generate_structured_patterns

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_all(seed: int):
    """Seed both PyTorch and NumPy global RNGs from a single integer."""
    torch.manual_seed(seed)
    np.random.seed(int(seed) & 0xFFFF_FFFF)

def compute_overlap(finals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute overlap m = (1/N) * sum(x_i * xi_i).
    Shape: (..., N) -> (...)
    """
    N = finals.shape[-1]
    return (finals * targets).sum(dim=-1) / N

def compute_success(overlaps: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """
    Return boolean tensor of success based on overlap threshold.
    """
    return overlaps >= threshold

def aggregate_success(success: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute success rate along specified dimension.
    """
    return success.float().mean(dim=dim)

def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_boot: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    """
    if len(values) < 2:
        return np.mean(values), np.mean(values)
    resamples = np.random.choice(values, size=(n_boot, len(values)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower
    return float(np.quantile(means, lower)), float(np.quantile(means, upper))

def run_batched_retrieval(N: int, p_list: List[int], p: float, beta: float = 0.5, 
                          n_trials: int = 5, max_steps: int = 100, seed: int = 42, 
                          lam: float = 2.0, device: str = _DEVICE, 
                          topology_cache: Optional[dict] = None,
                          F: Optional[torch.Tensor] = None, 
                          C: Optional[torch.Tensor] = None, 
                          centered: bool = False,
                          noise_level: float = 0.10):
    """
    Run retrieval for a batch of capacities (P values) simultaneously.
    Returns: Tensor of shape (len(p_list), n_trials) containing overlaps.
    """
    q, _ = compute_q_from_budget(p, N, beta)
    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=device)

    # 1. Topology
    if topology_cache is not None:
        cache_key = (int(N), float(p), float(beta), int(seed))
        cached = topology_cache.get(cache_key)
        if cached is None:
            seed_all(seed)
            model.generate_masks()
            topology_cache[cache_key] = (model.c, model.tri_i, model.tri_j, model.tri_k, model.n_tri)
        else:
            model.c, model.tri_i, model.tri_j, model.tri_k, model.n_tri = cached
    else:
        seed_all(seed)
        model.generate_masks()

    # 2. Patterns & Storage
    # Seed patterns based on setting and seed as per plan
    # (data setting, seed) should be consistent. Here seed is the experiment seed.
    seed_all(seed ^ 0xABCDEF) 
    P_max = max(p_list)
    if F is not None:
        patterns = generate_structured_patterns(P_max, F, device=device)
    else:
        patterns = torch.randint(0, 2, (P_max, N), device=device).float() * 2 - 1
        
    model.store_multiple_p(patterns, p_list, centered=centered, C=C)

    # 3. Initial States
    B = len(p_list)
    inits = torch.empty(B, n_trials, N, device=device)
    targets = torch.empty(B, n_trials, N, device=device)
    
    n_flip = max(1, int(noise_level * N))
    
    # We use trial-specific noise seeds derived from base seed
    for trial in range(n_trials):
        seed_all(seed * 10007 + trial)
        for b, P in enumerate(p_list):
            mu = trial % P
            target = patterns[mu].float()
            flip_idx = torch.randperm(N, device=device)[:n_flip]
            init = target.clone()
            init[flip_idx] *= -1
            inits[b, trial] = init
            targets[b, trial] = target

    finals = model.run(inits, max_steps=max_steps) # (B, n_trials, N)
    overlaps = compute_overlap(finals, targets)   # (B, n_trials)
    return overlaps

def find_empirical_pc_by_success(N: int, p: float, beta: float, lam: float, 
                                 n_trials: int, n_seeds: int,
                                 success_threshold: float = 0.9,
                                 overlap_threshold: float = 0.95,
                                 noise_level: float = 0.10,
                                 pbar=None, device: str = _DEVICE,
                                 F: Optional[torch.Tensor] = None, 
                                 C: Optional[torch.Tensor] = None, 
                                 centered: bool = False):
    """
    Find critical capacity Pc based on success rate.
    Returns: Pc (float) and optionally full sweep data if requested.
    """
    # Define search candidates
    max_p = int(min(N, max(64, int(0.8 * N))))
    step  = max(1, N // 20)
    p_candidates = np.unique(np.concatenate([
        np.arange(1, 15),
        np.arange(15, max_p + 1, step),
    ])).astype(int)

    topology_cache = {}
    
    # success_counts: (len(p_candidates),)
    # total_trials = n_seeds * n_trials
    
    all_overlaps = [] # list of (n_seeds, B, n_trials)
    
    for seed in range(n_seeds):
        overlaps = run_batched_retrieval(
            N, p_candidates.tolist(), p, beta,
            n_trials=n_trials, seed=seed, lam=lam,
            device=device, topology_cache=topology_cache,
            F=F, C=C, centered=centered,
            noise_level=noise_level
        )
        all_overlaps.append(overlaps.cpu().numpy())
        if pbar is not None:
            pbar.set_postfix(p=f"{p:.2f}", seed=f"{seed+1}/{n_seeds}")

    # (n_seeds, B, n_trials)
    all_overlaps = np.stack(all_overlaps)
    # success: (n_seeds, B, n_trials)
    success = all_overlaps >= overlap_threshold
    # success_rate_per_seed: (n_seeds, B)
    success_rate_per_seed = success.mean(axis=2)
    # mean_success_rate: (B,)
    mean_success_rate = success_rate_per_seed.mean(axis=0)
    
    # Find boundary in coarse sweep
    best_idx = -1
    for i, rate in enumerate(mean_success_rate):
        if rate >= success_threshold:
            best_idx = i
        else:
            break
            
    if best_idx == -1: return 0.0
    if best_idx == len(p_candidates) - 1: return float(p_candidates[-1])
    
    # Fine sweep
    p_fine = np.arange(p_candidates[best_idx] + 1, p_candidates[best_idx+1]).astype(int)
    if len(p_fine) == 0:
        return float(p_candidates[best_idx])

    all_overlaps_fine = []
    for seed in range(n_seeds):
        overlaps = run_batched_retrieval(
            N, p_fine.tolist(), p, beta,
            n_trials=n_trials, seed=seed, lam=lam,
            device=device, topology_cache=topology_cache,
            F=F, C=C, centered=centered,
            noise_level=noise_level
        )
        all_overlaps_fine.append(overlaps.cpu().numpy())

    # (n_seeds, B_fine, n_trials)
    all_overlaps_fine = np.stack(all_overlaps_fine)
    mean_success_fine = (all_overlaps_fine >= overlap_threshold).mean(axis=(0, 2))
    
    all_p = np.concatenate([[p_candidates[best_idx]], p_fine, [p_candidates[best_idx+1]]])
    all_rates = np.concatenate([[mean_success_rate[best_idx]], mean_success_fine, [mean_success_rate[best_idx+1]]])
    
    for i in range(len(all_rates) - 1):
        if all_rates[i] >= success_threshold and all_rates[i+1] < success_threshold:
            # Linear interpolation for fractional Pc
            denom = all_rates[i] - all_rates[i+1]
            frac = (all_rates[i] - success_threshold) / denom
            return all_p[i] + frac * (all_p[i+1] - all_p[i])
            
    return float(all_p[best_idx])
