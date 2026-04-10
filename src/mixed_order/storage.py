import torch

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
        patterns = patterns.float().to(device)

        # 2-body: Iterative accumulation to avoid O(P_max * N^2) memory
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
                if centered and C is not None:
                    J_b = J_b - P * C.to(device)
                J_b = J_b / (self.config.p * N)
            else:
                J_b = torch.zeros(N, N, device=device)
            J_b.fill_diagonal_(0.0)
            self.J[b] = J_b * self.topology.c

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
