import torch

class Dynamics:
    def __init__(self, config, topology, storage):
        self.config = config
        self.topology = topology
        self.storage = storage

    @torch.inference_mode()
    def local_field(self, state):
        """
        Compute local field for batched states and batched weights.
        
        Args:
            state (Tensor): Shape (B, T, N) where B is capacity-batch, T is trials.
        Returns:
            Tensor: Shape (B, T, N).
        """
        B, T, N = state.shape
        # 2-body: (B, T, N) @ (B, N, N) -> (B, T, N)
        h = torch.bmm(state, self.storage.J.transpose(1, 2))

        if self.topology.n_tri > 0:
            # Pre-fetch components
            si = state[:, :, self.topology.tri_i]  # (B, T, n_tri)
            sj = state[:, :, self.topology.tri_j]
            sk = state[:, :, self.topology.tri_k]
            W  = self.storage.W_vals.unsqueeze(1) # (B, 1, n_tri)

            # Accumulate 3rd-order fields into h sequentially to avoid O(B*T*3*n_tri) memory
            idx_i = self.topology.tri_i.view(1, 1, -1).expand(B, T, -1)
            idx_j = self.topology.tri_j.view(1, 1, -1).expand(B, T, -1)
            idx_k = self.topology.tri_k.view(1, 1, -1).expand(B, T, -1)
            
            h.scatter_add_(2, idx_i, W * sj * sk)
            h.scatter_add_(2, idx_j, W * si * sk)
            h.scatter_add_(2, idx_k, W * si * sj)
        
        return h

    @torch.inference_mode()
    def run(self, initial_state, max_steps=100):
        """
        Run synchronous updates for a batched state.
        initial_state: (B, T, N) or (T, N). 
        If (T, N), it is expanded to (B, T, N) where B is self.storage.J.shape[0].
        """
        if initial_state.dim() == 2:
            B = self.storage.J.shape[0]
            state = initial_state.unsqueeze(0).expand(B, -1, -1).clone().float().to(self.config.device)
        else:
            state = initial_state.clone().float().to(self.config.device)
            
        B, T, N = state.shape
        ones = torch.ones(1, device=self.config.device)
        neg = -ones

        for _ in range(max_steps):
            h = self.local_field(state)
            new_state = torch.where(h >= 0.0, ones, neg)
            if (new_state != state).sum().item() == 0:
                break
            state = new_state
            
        return state
