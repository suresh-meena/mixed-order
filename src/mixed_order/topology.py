import torch
import numpy as np
from numba import njit
import random
from mixed_order.utils import pin_and_move

@njit
def _unrank_triples_numba(N, ranks, prefix_counts):
    """
    Convert ranks in [0, C(N,3)) into triples (i < j < k)
    using combinadic-style unranking with precomputed prefix counts.
    """
    m = ranks.shape[0]
    tri_i = np.empty(m, dtype=np.int64)
    tri_j = np.empty(m, dtype=np.int64)
    tri_k = np.empty(m, dtype=np.int64)

    for idx in range(m):
        r = ranks[idx]

        # Find i via prefix counts of blocks with fixed i.
        lo = 0
        hi = N - 3
        while lo <= hi:
            mid = (lo + hi) // 2
            if prefix_counts[mid] <= r:
                lo = mid + 1
            else:
                hi = mid - 1
        i = lo

        prev = 0 if i == 0 else prefix_counts[i - 1]
        rem = r - prev

        # Unrank (j, k) within fixed i.
        n2 = N - i - 1
        a = 0
        cnt = n2 - 1
        while rem >= cnt and cnt > 0:
            rem -= cnt
            a += 1
            cnt -= 1

        j = i + 1 + a
        k = j + 1 + rem

        tri_i[idx] = i
        tri_j[idx] = j
        tri_k[idx] = k

    return tri_i, tri_j, tri_k


def _sample_triples_numba(N, q, seed=42):
    """
    Sample triples by rank, avoiding O(N^3) enumeration.

    Distribution:
    1) Set M ~= round(C(N,3) * q)
    2) Draw M unique ranks uniformly from [0, C(N,3))
    3) Unrank ranks -> (i, j, k)
    """
    total = N * (N - 1) * (N - 2) // 6
    if total <= 0 or q <= 0.0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    m = int(round(total * q))
    if m <= 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # random.sample(range(total), m) avoids materializing full range.
    py_rng = random.Random(seed)
    ranks = np.array(py_rng.sample(range(total), m), dtype=np.int64)
    ranks.sort()

    prefix_counts = np.empty(N - 2, dtype=np.int64)
    acc = 0
    for i in range(N - 2):
        acc += (N - i - 1) * (N - i - 2) // 2
        prefix_counts[i] = acc

    return _unrank_triples_numba(N, ranks, prefix_counts)

class Topology:
    def __init__(self, config):
        self.config = config
        self.c = None
        self.tri_i = None
        self.tri_j = None
        self.tri_k = None
        self.all_tri_idx = None
        self.n_tri = 0

    def generate_masks(self, generator=None, device_override=None):
        N = self.config.N
        device = self.config.device if device_override is None else device_override
        
        if generator is None:
            c_upper = (torch.rand(N, N, device=device) < self.config.p).to(dtype=self.config.dtype)
        else:
            c_upper = (torch.rand(N, N, generator=generator, device=device) < self.config.p).to(dtype=self.config.dtype)
        
        self.c = torch.triu(c_upper, diagonal=1)
        self.c = self.c + self.c.T

        if self.config.q > 0 and abs(self.config.lam) > 0.0:
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
                # Convert numpy arrays to tensors, pin and move to `device`.
                # `pin_and_move` will pin CPU tensors and perform async move
                # when `device` is non-CPU.
                self.tri_i = pin_and_move(tri_i_np, device, non_blocking=True, dtype=torch.long)
                self.tri_j = pin_and_move(tri_j_np, device, non_blocking=True, dtype=torch.long)
                self.tri_k = pin_and_move(tri_k_np, device, non_blocking=True, dtype=torch.long)
            else:
                self.n_tri = 0
                self.tri_i = self.tri_j = self.tri_k = None
        else:
            self.n_tri = 0
            self.tri_i = self.tri_j = self.tri_k = None
