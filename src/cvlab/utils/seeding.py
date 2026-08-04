"""Global RNG seeding. Decision 4: one seed controls every stochastic element
of a run (subset sampling, weight init, batch shuffling order)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) RNGs, and force deterministic
    Torch kernels. Call once at the start of a run, before any sampling,
    model construction, or data loading."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
