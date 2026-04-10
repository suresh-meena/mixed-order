import torch
from mixed_order.utils import pin_and_move

class Storage:
    def __init__(self, config, topology):
        self.config = config
        self.topology = topology
        self.J = None
        self.W_vals = None

    def store_multiple_p(self, patterns, p_list, centered=False, C=None):
        """
        Hebbian storage for multiple capacities simultaneously.
        
        Args:
            patterns (Tensor): Shape (P_max, N).
            p_list (list[int]): List of capacities to test.
            centered (bool): If True, apply covariance centering for structured data.
            C (Tensor): Pairwise covariance matrix (N, N), required if centered=True.
        """
        N = self.config.N
        device = self.config.device
        p_list = sorted(p_list)
        B = len(p_list)
        # Move patterns to target device; if patterns live in pinned CPU memory
        # this can be done asynchronously via `pin_and_move`.
        patterns = pin_and_move(patterns, device, dtype=self.config.dtype)
        C_device = None if C is None else pin_and_move(C, device)

        # 2-body: choose dense or sparse accumulation depending on topology density
        use_sparse = False
        edge_i = edge_j = None
        if self.topology.c is not None:
            upper = torch.triu(self.topology.c, diagonal=1)
            edge_idx = torch.nonzero(upper, as_tuple=False)
            n_edges = edge_idx.shape[0]
            total_pairs = N * (N - 1) // 2
            frac = float(n_edges) / float(total_pairs) if total_pairs > 0 else 0.0
            use_sparse = frac <= getattr(self.config, 'sparse_threshold', 0.10) and n_edges > 0

        if not use_sparse:
            # Dense accumulation (original path)
            self.J = torch.empty(B, N, N, device=device)
            J_acc = torch.zeros(N, N, device=device)

            current_P = 0
            for b, P in enumerate(p_list):
                if self.config.p > 0:
                    # Accumulate new patterns up to P
                    if P > current_P:
                        # e.g., chunk size is P - current_P
                        p_chunk = patterns[current_P:P]
                        # We can use einsum or mm for the chunk: (chunk, N) -> (N, N)
                        J_acc += torch.matmul(p_chunk.T, p_chunk)
                        current_P = P
                        
                    J_b = J_acc.clone()
                    if centered and C_device is not None:
                        J_b = J_b - P * C_device
                    J_b = J_b / (self.config.p * N)
                else:
                    J_b = torch.zeros(N, N, device=device)
                J_b.fill_diagonal_(0.0)
                self.J[b] = J_b * self.topology.c
        else:
            # Sparse accumulation: store only upper-triangular edges (i < j)
            edge_i = pin_and_move(edge_idx[:, 0], device, dtype=torch.long)
            edge_j = pin_and_move(edge_idx[:, 1], device, dtype=torch.long)
            n_edges = edge_i.numel()

            # Prepare sparse containers
            self.J = None
            self.J_edges_i = edge_i
            self.J_edges_j = edge_j
            self.J_vals = torch.empty(B, n_edges, device=device, dtype=self.config.dtype)

            J_acc_edges = torch.zeros(n_edges, device=device, dtype=self.config.dtype)
            current_P = 0

            # Precompute C edge values if centering requested
            C_edge = None
            if centered and C_device is not None:
                C_edge = C_device[edge_i, edge_j]

            for b, P in enumerate(p_list):
                if self.config.p > 0:
                    if P > current_P:
                        p_chunk = patterns[current_P:P]
                        # gather columns for all edges
                        p_i = p_chunk[:, edge_i]
                        p_j = p_chunk[:, edge_j]
                        # accumulate sum_mu xi_i * xi_j for each edge
                        J_acc_edges += (p_i * p_j).sum(dim=0)
                        current_P = P

                    if centered and C_edge is not None:
                        J_vals_b = (J_acc_edges - P * C_edge) / (self.config.p * N)
                    else:
                        J_vals_b = J_acc_edges / (self.config.p * N)
                else:
                    J_vals_b = torch.zeros(n_edges, device=device, dtype=self.config.dtype)

                # store per-capacity edge values
                self.J_vals[b] = J_vals_b

        # 3-body: Iterative accumulation over triangles
        if self.topology.n_tri > 0:
            self.W_vals = torch.empty(B, self.topology.n_tri, device=device)
            W_acc = torch.zeros(self.topology.n_tri, device=device)
            
            tri_i = self.topology.tri_i
            tri_j = self.topology.tri_j
            tri_k = self.topology.tri_k
            
            current_P_W = 0
            norm = self.config.lam / (self.config.q * N**2)
            
            for b, P in enumerate(p_list):
                if P > current_P_W:
                    p_chunk = patterns[current_P_W:P]
                    p_tri = p_chunk[:, tri_i] * p_chunk[:, tri_j] * p_chunk[:, tri_k]
                    W_acc += p_tri.sum(dim=0)
                    current_P_W = P
                self.W_vals[b] = W_acc * norm
        else:
            self.W_vals = None
