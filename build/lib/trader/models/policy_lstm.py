import torch
import torch.nn as nn
from trader.models.action_heads import TanhHead


class RecurrentActorCritic(nn.Module):
    """
    Paper's second LSTM: recurrent policy/value head.
    Takes the embedded sequence from FeatureLSTM and applies an LSTM,
    then produces:
      - policy (mu, std) via TanhHead
      - value V(s)

    Inputs:
      z_seq: (B, T, E)   sequence of embeddings
      h: optional ((h, c)) hidden state for the policy LSTM

    Returns:
      mu:   (B, A)
      std:  (B, A)
      v:    (B, 1)
      h:    new hidden state tuple
    """

    def __init__(self, obs_embed_dim: int, action_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=obs_embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.pi = TanhHead(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, z_seq, h=None):
        # z_seq: (B, T, E)
        out, h = self.lstm(z_seq, h) if h is not None else self.lstm(z_seq)
        h_t = out[:, -1, :]            # (B, H) last timestep
        mu, std = self.pi(h_t)         # (B, A), (B, A)
        v = self.v(h_t)                # (B, 1)
        return mu, std, v, h
