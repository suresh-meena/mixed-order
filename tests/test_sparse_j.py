import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mixed_order_model import MixedOrderHopfieldNetwork
from mixed_order.config import NetworkConfig
from mixed_order.metrics import seed_all

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def deterministic_seed():
    seed_all(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    yield


def test_sparse_j_equivalence():
    N = 24
    p = 0.05
    beta = 0.5
    lam = 0.0
    P = 6

    model = MixedOrderHopfieldNetwork(N, p, beta, lam, device=_DEVICE)
    # Relax the sparse threshold to ensure sparse path is chosen for this test
    model.config.sparse_threshold = 0.99

    model.generate_masks()

    patterns = (torch.randint(0, 2, (P, N), device=_DEVICE) * 2 - 1).float()
    model.store_multiple_p(patterns, [P], centered=False)

    # Ensure sparse representation was created
    assert getattr(model.storage, 'J_vals', None) is not None
    assert getattr(model.storage, 'J', None) is None

    # Small test state
    state = (torch.randint(0, 2, (1, 1, N), device=_DEVICE) * 2 - 1).float()

    h_sparse = model.local_field(state)

    # Reconstruct full dense J from sparse edge lists and values
    edge_i = model.storage.J_edges_i.cpu().long().numpy()
    edge_j = model.storage.J_edges_j.cpu().long().numpy()
    vals = model.storage.J_vals[0].cpu().numpy()  # B == 1

    J_full = torch.zeros(N, N, device=_DEVICE, dtype=h_sparse.dtype)
    for (ii, jj), v in zip(zip(edge_i, edge_j), vals):
        J_full[ii, jj] = float(v)
        J_full[jj, ii] = float(v)

    # Force dense path on the same storage object
    model.storage.J = J_full.unsqueeze(0)
    model.storage.J_vals = None
    model.storage.J_edges_i = None
    model.storage.J_edges_j = None

    h_dense = model.local_field(state)

    assert torch.allclose(h_sparse, h_dense, atol=1e-6), "Sparse and dense local fields disagree"
