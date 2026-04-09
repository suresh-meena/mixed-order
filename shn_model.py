import torch
import math
import warnings
import numpy as np

# Suppress torch sparse-CSR "beta state" notice (harmless, feature is stable enough)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta", category=UserWarning)

class SimplicialHopfieldNetwork:
    def __init__(self, N, p, beta=0.5, lam=2.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Simplicial Hopfield Network with GPU acceleration and synchronous updates.
        Uses sparse representations to scale to large N (e.g., N=1000) under low VRAM.
        
        Args:
            N (int): Number of neurons.
            p (float): Edge probability (2-body sparsity).
            beta (float): Budget parameter M/N^2.
            lam (float): Normalization ratio lambda = b/a.
            device (str): Device to run on ('cuda' or 'cpu').
        """
        self.N = N
        self.p = p
        self.beta = beta
        self.lam = lam
        self.device = torch.device(device)
        
        # Compute q from budget
        M = beta * N**2
        M2 = 0.5 * p * N**2
        M3 = M - M2
        self.q = max(6.0 * M3 / (N**3), 0.0)
        
        # Initialize weights
        self.J = None  # 2-body weights (dense N x N, since N=1000 is only 4MB)
        
        # 3-body weights stored as sparse edge lists
        self.tri_i = None
        self.tri_j = None
        self.tri_k = None
        self.W_vals = None
        self.n_tri = 0
        
        self.tri_adj_indices = None
        self.tri_adj_indptr = None
        
        # Fused scatter index (precomputed in generate_masks)
        self.all_tri_idx = None

        # Masks
        self.c = None  # 2-body mask
        
    def generate_masks(self):
        """Generate sparse interaction masks c_{ij} and d_{ijk}.

        Uses numpy for fast Binomial sampling on CPU, then transfers the
        resulting index arrays to the target device in three bulk copies.
        All subsequent ops (adjacency build, matmul, scatter_add) run on device.
        This is faster than torch.distributions.Binomial for both CPU and CUDA.
        """
        N, device = self.N, self.device

        # 2-body mask c_{ij} ~ Bern(p)
        c_upper = (torch.rand(N, N, device=device) < self.p).float()
        self.c = torch.triu(c_upper, diagonal=1)
        self.c = self.c + self.c.T

        # 3-body mask d_{ijk} ~ i.i.d. Bern(q) for each triple (i<j<k)
        # For each pair (i<j), sample n_active ~ Binomial(k_avail, q) to avoid
        # materialising all N³/6 triples.  Peak memory O(N²) ≈ 50 MB for N=1000.
        #
        # numpy's binomial is vectorised over all ~N²/2 pairs at once and is
        # much faster than torch.distributions.Binomial on both CPU and CUDA.
        # After sampling, three bulk .to(device) transfers move the index arrays
        # to the GPU; from that point on everything stays on-device.
        if self.q > 0:
            q = self.q

            ii_np, jj_np = np.triu_indices(N, k=1)
            k_avail_np   = (N - jj_np - 1).astype(np.int64)

            # Vectorised Binomial draw for all ~N²/2 pairs at once
            n_active_np = np.random.binomial(k_avail_np, q)

            ti, tj, tk = [], [], []

            # n_active == 1 (common case): fully vectorised
            mask1 = n_active_np == 1
            if mask1.any():
                j1 = jj_np[mask1]
                k1 = (np.random.random(int(mask1.sum())) * (N - 1 - j1)).astype(np.int64) + j1 + 1
                ti.append(ii_np[mask1].astype(np.int64))
                tj.append(j1)
                tk.append(k1)

            # n_active >= 2 (rare): loop
            for idx in np.where(n_active_np >= 2)[0]:
                i_val, j_val, n_k = int(ii_np[idx]), int(jj_np[idx]), int(n_active_np[idx])
                k_vals = np.random.choice(N - j_val - 1, size=n_k, replace=False).astype(np.int64) + j_val + 1
                ti.append(np.full(n_k, i_val, dtype=np.int64))
                tj.append(np.full(n_k, j_val, dtype=np.int64))
                tk.append(k_vals)

            if ti:
                # Three bulk transfers to device — one per array
                tri_i_parts = [torch.as_tensor(np.concatenate(ti), dtype=torch.long, device=device)]
                tri_j_parts = [torch.as_tensor(np.concatenate(tj), dtype=torch.long, device=device)]
                tri_k_parts = [torch.as_tensor(np.concatenate(tk), dtype=torch.long, device=device)]
            else:
                tri_i_parts = tri_j_parts = tri_k_parts = []

            n_tri = sum(len(x) for x in tri_i_parts)

            if n_tri > 0:
                self.n_tri = n_tri
                self.tri_i = torch.cat(tri_i_parts)
                self.tri_j = torch.cat(tri_j_parts)
                self.tri_k = torch.cat(tri_k_parts)

                # ── Vectorised adjacency list (CSR-style, fully on GPU)
                all_t   = torch.arange(n_tri, device=device)
                neurons = torch.cat([self.tri_i, self.tri_j, self.tri_k])  # (3*n_tri,)
                tris    = torch.cat([all_t,      all_t,      all_t     ])  # (3*n_tri,)

                order = torch.argsort(neurons, stable=True)
                self.tri_adj_indices = tris[order]

                counts = torch.bincount(neurons[order], minlength=N)
                self.tri_adj_indptr = torch.zeros(N + 1, dtype=torch.long, device=device)
                self.tri_adj_indptr[1:] = torch.cumsum(counts, dim=0)

                # ── Fused scatter index (3*n_tri,): precomputed once so
                #    local_field can do a single scatter_add_ per call.
                #    Layout: [tri_i entries | tri_j entries | tri_k entries]
                self.all_tri_idx = torch.cat([self.tri_i, self.tri_j, self.tri_k])
            else:
                self.n_tri = 0
                self.tri_i = self.tri_j = self.tri_k = None
                self.tri_adj_indices = self.tri_adj_indptr = None
                self.all_tri_idx = None
        else:
            self.n_tri = 0
            self.tri_i = self.tri_j = self.tri_k = None
            self.tri_adj_indices = self.tri_adj_indptr = None
            self.all_tri_idx = None
            
    def store_patterns(self, patterns):
        """
        Store patterns using Hebbian learning rules.
        
        Args:
            patterns (torch.Tensor): Shape (P, N) containing {-1, 1}.
        """
        if patterns.shape[1] != self.N:
            patterns = patterns.T  # Ensure shape is (P, N)
            
        P = patterns.shape[0]
        patterns = patterns.float().to(self.device)
        
        # 2-body Hebbian weights: J_{ij} = (1 / pN) sum_mu xi_i^mu xi_j^mu
        if self.p > 0:
            J_dense = torch.matmul(patterns.T, patterns) / (self.p * self.N)
            J_dense.fill_diagonal_(0.0)
            J_dense = J_dense * self.c  # Apply sparsity mask
            # Store as sparse CSR to halve VRAM and speed up matvec
            self.J = J_dense.to_sparse_csr()
        else:
            self.J = torch.zeros(self.N, self.N, device=self.device).to_sparse_csr()
            
        # 3-body Hebbian weights: W_{ijk} = (lam / qN^2) sum_mu xi_i^mu xi_j^mu xi_k^mu
        if self.n_tri > 0:
            norm = self.lam / (self.q * self.N**2)
            # Chunk over triangles to cap peak VRAM at ~(P * CHUNK * 3 * 4) bytes.
            # For N=1000, n_tri can reach ~10M; without chunking that is 3×(P×10M) floats.
            CHUNK = 100_000
            if self.n_tri <= CHUNK:
                p_i = patterns[:, self.tri_i]
                p_j = patterns[:, self.tri_j]
                p_k = patterns[:, self.tri_k]
                self.W_vals = torch.sum(p_i * p_j * p_k, dim=0) * norm
            else:
                self.W_vals = torch.empty(self.n_tri, device=self.device)
                for start in range(0, self.n_tri, CHUNK):
                    end = min(start + CHUNK, self.n_tri)
                    p_i = patterns[:, self.tri_i[start:end]]
                    p_j = patterns[:, self.tri_j[start:end]]
                    p_k = patterns[:, self.tri_k[start:end]]
                    self.W_vals[start:end] = torch.sum(p_i * p_j * p_k, dim=0) * norm
        else:
            self.W_vals = None
            
    def local_field(self, state):
        """Compute the local field h_i for all neurons.

        Supports unbatched (N,) or batched (B, N) states.
        The 3-body contribution uses a single fused scatter_add_ over all three
        vertex roles, reducing GPU kernel launches from 3 to 1.
        """
        is_batched = state.dim() > 1

        if is_batched:
            h = torch.mm(self.J, state.T).T          # (B, N)
        else:
            h = torch.mv(self.J, state)               # (N,)

        if self.n_tri > 0:
            n = self.n_tri
            if is_batched:
                B  = state.shape[0]
                si = state[:, self.tri_i]   # (B, n_tri)
                sj = state[:, self.tri_j]
                sk = state[:, self.tri_k]
                W  = self.W_vals.unsqueeze(0)   # (1, n_tri)

                # All three contributions stacked: (B, 3*n_tri)
                all_cont = torch.cat([W * sj * sk,
                                      W * si * sk,
                                      W * si * sj], dim=1)
                # Broadcast index: (B, 3*n_tri)
                idx = self.all_tri_idx.unsqueeze(0).expand(B, -1)
                h.scatter_add_(1, idx, all_cont)
            else:
                si = state[self.tri_i]          # (n_tri,)
                sj = state[self.tri_j]
                sk = state[self.tri_k]
                W  = self.W_vals

                # Fused: single scatter over all three vertex roles
                all_cont = torch.cat([W * sj * sk,
                                      W * si * sk,
                                      W * si * sj])     # (3*n_tri,)
                h.scatter_add_(0, self.all_tri_idx, all_cont)

        return h
        
    def run(self, initial_state, max_steps=100, tol=1e-5):
        """
        Run synchronous zero-temperature updates until convergence or max_steps.
        Supports both unbatched (N,) and batched (B, N) initial states.

        Convergence is declared when no neuron changes (checked with a single
        integer comparison to avoid multiple GPU syncs).

        Returns:
            Tensor with the same shape as ``initial_state``.
        """
        state = initial_state.float().to(self.device).clone()
        is_batched = state.dim() > 1

        ones  = torch.ones(1, device=self.device)
        neg   = -ones

        for _ in range(max_steps):
            h         = self.local_field(state)
            new_state = torch.where(h >= 0.0, ones, neg).expand_as(h)
            # Fast convergence check: single .item() call
            if (new_state != state).sum().item() == 0:
                break
            state = new_state

        return state
