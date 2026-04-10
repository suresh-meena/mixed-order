import torch
from dataclasses import dataclass, field

@dataclass
class NetworkConfig:
    N: int
    p: float
    beta: float = 0.5
    lam: float = 2.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    q: float = field(init=False)
    
    def __post_init__(self):
        M = self.beta * self.N**2
        M2 = 0.5 * self.p * self.N**2
        M3 = M - M2
        self.q = max(6.0 * M3 / (self.N**3), 0.0)
