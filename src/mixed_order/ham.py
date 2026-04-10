import torch
import torch.nn.functional as F
import math

class SparseMixedOrderHAM:
    """
    Hierarchical Associative Memory (HAM) layer using the mixed-order sparse 
    polynomial feature map. This is decoupled from the binary core engine.
    """
    def __init__(self, config, topology):
        self.config = config
        self.topology = topology
        self.patterns = None
        self.P = 0
        
    def store(self, patterns):
        """Store the patterns (keys). patterns shape: (P, N)"""
        self.patterns = patterns.float().to(self.config.device)
        self.P = self.patterns.shape[0]
        
    def score(self, x):
        """
        Compute s_mu(x) = u_mu^T Phi_mix(x) for all mu.
        This represents the inner product in the sparse feature space.
        
        Args:
            x: (B, N) continuous state tensor
        Returns: 
            (B, P) scores
        """
        if self.patterns is None:
            raise ValueError("Patterns must be stored before scoring.")
            
        B, N = x.shape
        P = self.P
        device = self.config.device
        
        x_flat = x.unsqueeze(1).expand(B, P, N) # (B, P, N)
        p_flat = self.patterns.unsqueeze(0).expand(B, P, N) # (B, P, N)
        
        # Local overlaps m_i^\mu = xi_i^\mu * x_i
        m = x_flat * p_flat
        
        # 1. Pairwise score: sum_{i,j} c_ij m_i m_j
        C = self.topology.c.to(device).float()
        
        # Reshape m to (B*P, N) for bmm/mv operations, or use einsum
        # m: (B, P, N)
        m_2d = m.reshape(B*P, N)
        # m_C_2d: (B*P, N)
        m_C_2d = torch.matmul(m_2d, C)
        m_C = m_C_2d.reshape(B, P, N)
        
        # We need the inner product m * m_C
        s2 = (m * m_C).sum(dim=2) # (B, P)
        
        # Remove self-interactions (diagonal was already 0 in C)
        norm2 = 1.0 / (self.config.p * N)
        s2 = s2 * norm2
        
        # 2. 3-body score: sum_{i,j,k} m_i m_j m_k over triangles
        if self.topology.n_tri > 0:
            m_i = m[:, :, self.topology.tri_i] # (B, P, n_tri)
            m_j = m[:, :, self.topology.tri_j]
            m_k = m[:, :, self.topology.tri_k]
            
            s3 = (m_i * m_j * m_k).sum(dim=2) # (B, P)
            norm3 = self.config.lam / (self.config.q * (N**2))
            s3 = s3 * norm3
        else:
            s3 = torch.zeros(B, P, device=device)
            
        return s2 + s3
        
    def retrieve(self, x, beta_temp=1.0, max_steps=10):
        """
        Softmax retrieval demo (continuous state).
        This implements the attention-like step from the log-sum-exp energy.
        
        Args:
            x: (B, N) initial continuous state
            beta_temp: inverse temperature for softmax
        Returns:
            (B, N) fixed point state
        """
        state = x.clone().float().to(self.config.device)
        
        for _ in range(max_steps):
            scores = self.score(state) # (B, P)
            attn = F.softmax(beta_temp * scores, dim=1) # (B, P)
            
            # state_new = sum_mu a_mu xi_mu
            state_new = torch.matmul(attn, self.patterns) # (B, N)
            
            diff = torch.norm(state_new - state, p=2, dim=1).max().item()
            state = state_new
            if diff < 1e-4:
                break
                
        return state
