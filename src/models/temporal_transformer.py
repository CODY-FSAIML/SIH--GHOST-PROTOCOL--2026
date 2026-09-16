"""
Temporal Transformer Forecaster (Phase 1)
=========================================
Neural sequence model mapping H past NetworkState vectors [S_(t-H+1), ..., S_t]
to future attack probability P(Y_(t+1) = 1).

Architectural Invariants:
1. Input projection: 16 (macro features) -> d_model (default 64)
2. Positional Encoding: Sinusoidal or learned positional embeddings
3. Transformer Encoder:
   - num_layers: 2
   - nhead: 4
   - d_model: 64
   - dim_feedforward: 128
   - dropout: 0.1
   - batch_first: True
4. Final Representation: Uses final/current timestep representation (index H-1),
   representing the current state conditioned on historical temporal trajectory.
5. Prediction Head:
   - Linear(64, 32) -> ReLU -> Dropout -> Linear(32, 1)
   - Outputs unconstrained LOGITS (sigmoid is NOT applied inside the model).
6. Attention Extraction:
   - Attention weights can be retained from multihead attention layers for future auditing.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalTransformerForecaster(nn.Module):
    """
    Temporal Transformer encoder predicting attack occurrence at t+1.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # 1. Linear projection from raw macro features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # 3. Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 4. Classification prediction head (outputs raw unconstrained logit)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)

        Returns:
            logits: Output tensor of shape (batch_size,) representing logit P(Y_t+1 = 1).
                    Sigmoid is NOT applied.
        """
        if x.dim() != 3 or x.size(2) != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )

        # Project to d_model and add positional encoding
        h = self.input_proj(x)                  # (B, H, d_model)
        h = self.pos_encoder(h)                 # (B, H, d_model)

        # Pass through Transformer encoder
        encoded = self.transformer_encoder(h, mask=mask)  # (B, H, d_model)

        # Take representation at the final timestep H-1 (anchor window t)
        current_state_repr = encoded[:, -1, :]  # (B, d_model)

        # Pass through prediction head to obtain scalar logit
        logits = self.head(current_state_repr).squeeze(-1)  # (B,)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: returns calibrated sigmoid probabilities in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
