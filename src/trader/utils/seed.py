import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int = 42):
    """
    Sets global seeds for Python, NumPy, and PyTorch (if available).
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
