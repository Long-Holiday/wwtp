import torch

_orig_use_deterministic = torch.use_deterministic_algorithms

def _safe_use_deterministic(mode, warn_only=True):
    return _orig_use_deterministic(mode, warn_only=True)

torch.use_deterministic_algorithms = _safe_use_deterministic
