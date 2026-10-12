"""Reproducibility helper. / Hàm hỗ trợ tái lập kết quả."""
import random

import numpy as np


def set_seed(seed: int = 42) -> int:
    """Seed Python, NumPy and PyTorch (if installed) and return the seed.

    VI: Cố định seed cho Python, NumPy và PyTorch (nếu có cài), trả về seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
