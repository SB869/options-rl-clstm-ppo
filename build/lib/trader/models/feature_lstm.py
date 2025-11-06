import torch
import torch.nn as nn


class FeatureLSTM(nn.Module):
    """
    Paper's first LSTM: sequence feature extractor.
    Input:  (B, T, D)
    Output: (B, T, E), plus (h, c) for recurrence chaining if desired.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x_seq, hx=None):
        # x_seq: (B, T, D)
        out, (h, c) = self.lstm(x_seq, hx) if hx is not None else self.lstm(x_seq)
        return out, (h, c)
