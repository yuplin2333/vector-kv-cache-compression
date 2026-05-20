"""
Randomness control helpers.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.
    """
    assert isinstance(seed, int), "seed must be int"
    assert seed >= 0, "seed must be non-negative"

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
