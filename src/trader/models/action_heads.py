import torch
import torch.nn as nn


class TanhHead(nn.Module):
    """
    Continuous policy head: outputs mean in [-1,1]^A via tanh,
    with a learned, state-independent log_std per action dimension.
    """

    def __init__(self, in_dim: int, action_dim: int, init_log_std: float = -0.5):
        super().__init__()
        self.mu = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, x):
        # x: (B, H)
        mu = torch.tanh(self.mu(x))            # (B, A) bounded to [-1,1]
        std = self.log_std.exp().expand_as(mu) # (B, A)
        return mu, std
