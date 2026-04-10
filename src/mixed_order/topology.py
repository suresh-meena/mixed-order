import torch
import numpy as np
from numba import njit

@njit
def _sample_triples_numba(N, q, seed=42):
    """
    Fast, memory-efficient sampling of triples (i < j < k) with Bern(q).
    Uses a single-pass sequential loop with Numba JIT and a deterministic 
    two-pass strategy to minimize memory overhead.
    """
    # Pass 1: Count
    state = seed
    total = 0
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                if (state / 2147483647.0) < q:
                    total += 1
                    
    tri_i = np.empty(total, dtype=np.int64)
    tri_j = np.empty(total, dtype=np.int64)
    tri_k = np.empty(total, dtype=np.int64)
    
    # Pass 2: Fill
    state = seed
    curr = 0
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                if (state / 2147483647.0) < q:
                    tri_i[curr] = i
                    tri_j[curr] = j
                    tri_k[curr] = k
                    curr += 1
    return tri_i, tri_j, tri_k

class Topology:
    def __init__(self, config):
        self.config = config
        self.c = None
        self.tri_i = None
        self.tri_j = None
        self.tri_k = None
        self.all_tri_idx = None
        self.n_tri = 0

    def generate_masks(self, generator=None):
        N = self.config.N
        device = self.config.device
        
        if generator is None:
            c_upper = (torch.rand(N, N, device=device) < self.config.p).float()
        else:
            c_upper = (torch.rand(N, N, generator=generator, device=device) < self.config.p).float()
        
        self.c = torch.triu(c_upper, diagonal=1)
        self.c = self.c + self.c.T

        if self.config.q > 0:
            if generator is not None:
                # Extract a pseudo-random integer from the torch generator
                seed = int(torch.randint(0, 0x7FFFFFFF, (), dtype=torch.int64, generator=generator, device=device).item())
            else:
                # Extract a pseudo-random integer from global torch RNG
                seed = int(torch.randint(0, 0x7FFFFFFF, (), dtype=torch.int64, device=device).item())
                
            tri_i_np, tri_j_np, tri_k_np = _sample_triples_numba(N, self.config.q, seed)
            n_tri = len(tri_i_np)

            if n_tri > 0:
                self.n_tri = n_tri
                self.tri_i = torch.as_tensor(tri_i_np, dtype=torch.long, device=device)
                self.tri_j = torch.as_tensor(tri_j_np, dtype=torch.long, device=device)
                self.tri_k = torch.as_tensor(tri_k_np, dtype=torch.long, device=device)
            else:
                self.n_tri = 0
                self.tri_i = self.tri_j = self.tri_k = None
        else:
            self.n_tri = 0
            self.tri_i = self.tri_j = self.tri_k = None
