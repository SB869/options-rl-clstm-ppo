# src/trader/models/action_heads.py
from __future__ import annotations
import torch
import torch.nn as nn


class TanhHead(nn.Module):
    """
    Continuous policy head: outputs mean in [-1, 1]^A via tanh,
    with a learned, state-independent log_std per action dimension.
    """

    def __init__(self, in_dim: int, action_dim: int, init_log_std: float = -0.5):
        super().__init__()
        self.mu = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

        # Initialization (orthogonal, mild gain)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, H)
        Returns (mu, std) each of shape (B, A)
        """
        # Compute mean bounded to [-1, 1]
        mu = torch.tanh(self.mu(x))

        # Safe log_std -> std conversion in fp32, then cast back to match mu
        ls = self.log_std.float()
        ls = torch.nan_to_num(ls, nan=-0.5, posinf=0.0, neginf=-3.0)
        ls = torch.clamp(ls, min=-3.0, max=1.0)  # std in ~[0.05, 2.7]
        std = torch.exp(ls).to(dtype=mu.dtype, device=mu.device).expand_as(mu)

        return mu, std
