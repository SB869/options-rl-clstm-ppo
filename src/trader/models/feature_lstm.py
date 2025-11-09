# src/trader/models/feature_lstm.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

class FeatureLSTM(nn.Module):
    """
    Sequence feature extractor (batch_first).
    Input:  x_seq  (B, T, D)
            hx     optional (h, c) where each is (num_layers, B, H)
    Output: out    (B, T, H)
            (h, c) detached tuple for safe recurrence chaining
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Orthogonal init for stability
        for layer in range(num_layers):
            for suffix in ("ih", "hh"):
                w = getattr(self.lstm, f"weight_{suffix}_l{layer}")
                nn.init.orthogonal_(w, gain=1.0)
            for suffix in ("ih", "hh"):
                b = getattr(self.lstm, f"bias_{suffix}_l{layer}")
                nn.init.zeros_(b)

    def forward(
        self,
        x_seq: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x_seq: (B, T, D)
        hx:    optional tuple (h, c) each of shape (num_layers*(2 if bidi else 1), B, H)
        """
        # Ensure LSTM input is the same dtype as module weights (important under AMP)
        x_seq = x_seq.to(dtype=self.lstm.weight_ih_l0.dtype, device=self.lstm.weight_ih_l0.device)

        if hx is not None:
            h, c = hx
            # Match device/dtype to the module
            h = h.to(dtype=self.lstm.weight_ih_l0.dtype, device=self.lstm.weight_ih_l0.device)
            c = c.to(dtype=self.lstm.weight_ih_l0.dtype, device=self.lstm.weight_ih_l0.device)
            out, (h, c) = self.lstm(x_seq, (h, c))
        else:
            out, (h, c) = self.lstm(x_seq)

        # Detach hidden state so callers can safely carry it across steps without growing graphs
        return out, (h.detach(), c.detach())
