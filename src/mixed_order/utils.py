import numpy as np
import torch
from typing import Any, Optional


def seed_all(seed: int) -> None:
    """Seed both PyTorch and NumPy global RNGs from a single integer."""
    torch.manual_seed(int(seed) & 0xFFFF_FFFF)
    np.random.seed(int(seed) & 0xFFFF_FFFF)


def make_generator(seed: int, device: str) -> torch.Generator:
    """Create a torch.Generator on `device` seeded from `seed`."""
    generator_device = "cuda" if device == "cuda" else "cpu"
    gen = torch.Generator(device=generator_device)
    gen.manual_seed(int(seed) & 0xFFFF_FFFF)
    return gen


def pin_and_move(x: Any, device: str, non_blocking: bool = True, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Ensure a tensor/ndarray is in pinned CPU memory (if on CPU) and move to `device`.

    - If `x` is a numpy array, convert to tensor first.
    - If `x` is a CPU tensor, pin memory if not already pinned.
    - Move to `device` with `non_blocking` when applicable.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x

    if dtype is not None:
        t = t.to(dtype=dtype)

    # If tensor, handle CPU pinning first so callers that want CPU-pinned
    # tensors (device='cpu') will receive a pinned tensor.
    if isinstance(t, torch.Tensor):
        # If on CPU, ensure pinned memory and return early when target is CPU
        if t.device.type == "cpu":
            if not t.is_pinned():
                try:
                    t = t.pin_memory()
                except Exception:
                    # pin_memory may fail for some types; fall back silently
                    pass
            if device == "cpu":
                return t

        # If already on the desired non-CPU device, return as-is
        if t.device.type == device:
            return t

    # Otherwise move to the target device (async when supported)
    return t.to(device, non_blocking=non_blocking)


def choose_chunk_size(available_bytes: int, element_size: int, safety_fraction: float = 0.5) -> int:
    """
    Simple heuristic to choose a chunk length (number of elements) that fits in
    `safety_fraction` of `available_bytes`, given `element_size` bytes per element.
    """
    target = int(available_bytes * safety_fraction)
    if element_size <= 0:
        return 1024
    return max(1, target // element_size)
