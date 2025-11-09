# src/trader/models/policy_lstm.py
from __future__ import annotations
import torch
import torch.nn as nn

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_embed_dim: int, action_dim: int, hidden: int = 128, init_log_std: float = -1.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_embed_dim, hidden_size=hidden, batch_first=True)
        self.mu = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

        # Learnable log std
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

        # Inits
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=1.0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=1.0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, z, state=None):
        # z: (B,T,E)
        y, state = self.lstm(z, state) if state is not None else self.lstm(z)  # (B,T,H)
        h = y[:, -1, :]                       # (B,H)
        mu = self.mu(h)                       # (B,A)

        # Compute std in fp32 for numerical stability, then match dtype/device of mu
        ls = self.log_std.float()
        ls = torch.nan_to_num(ls, nan=-1.0, posinf=0.0, neginf=-3.0)
        ls = torch.clamp(ls, min=-3.0, max=0.0)          # std in ~[0.05, 1.0]
        std = torch.exp(ls).to(dtype=mu.dtype, device=mu.device).expand_as(mu)

        v = self.v(h)                          # (B,1)
        return mu, std, v, state
