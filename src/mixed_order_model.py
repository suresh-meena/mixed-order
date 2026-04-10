import torch
import warnings
from mixed_order.config import NetworkConfig
from mixed_order.topology import Topology
from mixed_order.storage import Storage
from mixed_order.dynamics import Dynamics

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta", category=UserWarning)

class MixedOrderHopfieldNetwork:
    def __init__(self, N, p, beta=0.5, lam=2.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = NetworkConfig(N=N, p=p, beta=beta, lam=lam, device=device)
        self.topology = Topology(self.config)
        self.storage = Storage(self.config, self.topology)
        self.dynamics = Dynamics(self.config, self.topology, self.storage)

    # Property accessors for backward compatibility
    @property
    def N(self): return self.config.N
    @property
    def p(self): return self.config.p
    @property
    def beta(self): return self.config.beta
    @property
    def lam(self): return self.config.lam
    @property
    def device(self): return torch.device(self.config.device)
    @property
    def q(self): return self.config.q
    
    @property
    def J(self): return self.storage.J
    @J.setter
    def J(self, val): self.storage.J = val
    
    @property
    def c(self): return self.topology.c
    @c.setter
    def c(self, val): self.topology.c = val
    
    @property
    def tri_i(self): return self.topology.tri_i
    @tri_i.setter
    def tri_i(self, val): self.topology.tri_i = val
    
    @property
    def tri_j(self): return self.topology.tri_j
    @tri_j.setter
    def tri_j(self, val): self.topology.tri_j = val
    
    @property
    def tri_k(self): return self.topology.tri_k
    @tri_k.setter
    def tri_k(self, val): self.topology.tri_k = val
    
    @property
    def n_tri(self): return self.topology.n_tri
    @n_tri.setter
    def n_tri(self, val): self.topology.n_tri = val
    
    @property
    def W_vals(self): return self.storage.W_vals
    @W_vals.setter
    def W_vals(self, val): self.storage.W_vals = val

    def generate_masks(self, generator=None, device_override=None):
        self.topology.generate_masks(generator=generator, device_override=device_override)

    def store_patterns(self, patterns):
        self.storage.store_multiple_p(patterns, [patterns.shape[0]])

    def store_multiple_p(self, patterns, p_list, centered=False, C=None):
        self.storage.store_multiple_p(patterns, p_list, centered=centered, C=C)

    def local_field(self, state):
        return self.dynamics.local_field(state)

    def run(self, initial_state, max_steps=100):
        return self.dynamics.run(initial_state, max_steps)
