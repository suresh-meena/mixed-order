import torch
from mixed_order.utils import pin_and_move

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

        # 2-body: support dense J or sparse edge-based representation
        if getattr(self.storage, 'J', None) is not None:
            # Dense path: (B, T, N) @ (B, N, N) -> (B, T, N)
            h = torch.bmm(state, self.storage.J.transpose(1, 2))
        elif getattr(self.storage, 'J_vals', None) is not None and getattr(self.storage, 'J_edges_i', None) is not None:
            # Sparse path: edges list with per-capacity values
            edge_i = self.storage.J_edges_i
            edge_j = self.storage.J_edges_j
            J_vals = self.storage.J_vals  # (B, n_edges)

            flat_count = B * T
            state_flat = state.reshape(flat_count, N)

            # expand edge indices to match flattened batch for gather/scatter
            idx_i = edge_i.view(1, -1).expand(flat_count, -1)
            idx_j = edge_j.view(1, -1).expand(flat_count, -1)

            si = torch.gather(state_flat, 1, idx_i)
            sj = torch.gather(state_flat, 1, idx_j)

            # expand J_vals to (B, T, n_edges) then flatten to (flat_count, n_edges)
            J_expanded = J_vals.unsqueeze(1).expand(-1, T, -1).reshape(flat_count, -1)

            contrib_i = J_expanded * sj
            contrib_j = J_expanded * si

            h_flat = torch.zeros(flat_count, N, device=state.device, dtype=state.dtype)
            h_flat.scatter_add_(1, idx_i, contrib_i)
            h_flat.scatter_add_(1, idx_j, contrib_j)

            h = h_flat.reshape(B, T, N)
        else:
            # Fallback: no pairwise interactions
            h = torch.zeros(B, T, N, device=state.device, dtype=state.dtype)

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
            # Determine batch size B from storage (dense J) or sparse J_vals
            if getattr(self.storage, 'J', None) is not None:
                B = self.storage.J.shape[0]
            elif getattr(self.storage, 'J_vals', None) is not None:
                B = self.storage.J_vals.shape[0]
            else:
                B = 1
            state = pin_and_move(initial_state.unsqueeze(0).expand(B, -1, -1).clone(), self.config.device, dtype=self.config.dtype)
        else:
            state = pin_and_move(initial_state.clone(), self.config.device, dtype=self.config.dtype)
            
        B, T, N = state.shape
        ones = torch.ones(1, device=self.config.device, dtype=self.config.dtype)
        neg = -ones

        # Check convergence only periodically to avoid frequent host<->device syncs
        for step in range(max_steps):
            h = self.local_field(state)
            new_state = torch.where(h >= 0.0, ones, neg)
            if (step & 3) == 0:
                if (new_state != state).sum().item() == 0:
                    state = new_state
                    break
            state = new_state
            
        return state
