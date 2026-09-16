"""Learned latent world model for temporal network-state transitions.

This module learns normalized transitions from the existing 32-dimensional
[NetworkState, first-difference velocity] representation. Its attack head
returns a predicted attack-risk score; it is not calibrated unless a separate
calibration procedure is performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..data.temporal_targets import TemporalWindow
from ..features.network_state import MacroNetworkState
from ..models.temporal_transformer import TemporalTransformerForecaster


@dataclass(frozen=True)
class LearnedWorldModelConfig:
    input_dim: int = 32
    state_dim: int = 16
    history_len: int = 10
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_len: int = 100
    latent_dim: int = 64
    transition_hidden_dim: int = 128
    decoder_hidden_dim: int = 128


class LearnedLatentWorldModel(nn.Module):
    """Transformer encoder plus latent transition, decoder, and risk heads."""

    mode = "learned"
    risk_score_semantics = "predicted attack risk score; calibration not established"

    def __init__(self, config: LearnedWorldModelConfig | None = None):
        super().__init__()
        self.config = config or LearnedWorldModelConfig()
        if self.config.input_dim != self.config.state_dim * 2:
            raise ValueError("input_dim must equal state_dim * 2 for state plus velocity input.")
        self.encoder = TemporalTransformerForecaster(
            input_dim=self.config.input_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
        )
        if self.config.latent_dim != self.config.d_model:
            self.latent_projection = nn.Linear(self.config.d_model, self.config.latent_dim)
        else:
            self.latent_projection = nn.Identity()
        self.transition = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.transition_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.transition_hidden_dim, self.config.latent_dim),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.decoder_hidden_dim, self.config.input_dim),
        )
        self.attack_risk_head = nn.Linear(self.config.latent_dim, 1)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3 or history.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected history [batch, {self.config.history_len}, {self.config.input_dim}], got {tuple(history.shape)}"
            )
        encoded = self.encoder.transformer_encoder(
            self.encoder.pos_encoder(self.encoder.input_proj(history))
        )
        return self.latent_projection(encoded[:, -1, :])

    def predict_next_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transition(latent)

    def decode_next(self, next_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.state_decoder(next_latent), self.attack_risk_head(next_latent).squeeze(-1)

    def forward(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encode_history(history)
        next_latent = self.predict_next_latent(latent)
        predicted_state, attack_risk_logit = self.decode_next(next_latent)
        return {
            "latent": latent,
            "next_latent": next_latent,
            "predicted_state": predicted_state,
            "attack_risk_logit": attack_risk_logit,
            "attack_risk_score": torch.sigmoid(attack_risk_logit),
        }

    @torch.no_grad()
    def rollout(self, history: torch.Tensor, steps: int) -> Dict[str, torch.Tensor]:
        """Roll predicted state representations forward for K future steps."""
        if steps < 1:
            raise ValueError("steps must be at least one")
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.size(1) != self.config.history_len:
            raise ValueError("History length does not match model configuration.")
        self.eval()
        current_history = history.clone()
        predicted_states: List[torch.Tensor] = []
        latent_states: List[torch.Tensor] = []
        risk_logits: List[torch.Tensor] = []
        risk_scores: List[torch.Tensor] = []
        for _ in range(steps):
            output = self(current_history)
            predicted_states.append(output["predicted_state"])
            latent_states.append(output["next_latent"])
            risk_logits.append(output["attack_risk_logit"])
            risk_scores.append(output["attack_risk_score"])
            current_history = torch.cat(
                [current_history[:, 1:, :], output["predicted_state"].unsqueeze(1)], dim=1
            )
        return {
            "predicted_states": torch.stack(predicted_states, dim=1),
            "latent_states": torch.stack(latent_states, dim=1),
            "attack_risk_logits": torch.stack(risk_logits, dim=1),
            "attack_risk_scores": torch.stack(risk_scores, dim=1),
            "risk_score_semantics": self.risk_score_semantics,
        }

    @classmethod
    def from_temporal_checkpoint(cls, checkpoint_path: str | Path, config: LearnedWorldModelConfig | None = None):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        temporal_config = checkpoint.get("model_config", {})
        model_config = config or LearnedWorldModelConfig(
            input_dim=int(temporal_config.get("input_dim", 32)),
            d_model=int(temporal_config.get("d_model", 64)),
            nhead=int(temporal_config.get("nhead", 4)),
            num_layers=int(temporal_config.get("num_layers", 2)),
            dim_feedforward=int(temporal_config.get("dim_feedforward", 128)),
            dropout=float(temporal_config.get("dropout", 0.1)),
            max_len=int(temporal_config.get("max_len", 100)),
        )
        model = cls(model_config)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("input_proj.")
            or key.startswith("pos_encoder.")
            or key.startswith("transformer_encoder.")
        }
        model.encoder.load_state_dict(encoder_state, strict=False)
        return model

    def checkpoint_payload(self, scaler: StandardScaler, training_metadata: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "config": asdict(self.config),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "scaler_state": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            "training_metadata": dict(training_metadata),
            "risk_score_semantics": self.risk_score_semantics,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> Tuple["LearnedLatentWorldModel", StandardScaler, Dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = cls(LearnedWorldModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_state"]["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(checkpoint["scaler_state"]["scale"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        return model, scaler, checkpoint


def state_velocity_representation(state: np.ndarray, previous_state: Optional[np.ndarray]) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    velocity = np.zeros_like(state) if previous_state is None else state - np.asarray(previous_state, dtype=np.float32)
    return np.concatenate([state, velocity]).astype(np.float32)


def build_transition_dataset(
    windows: Sequence[TemporalWindow],
    states_lookup: Mapping[int, MacroNetworkState],
    history_len: int = 10,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[Dict[str, Any]]]:
    """Build chronological history -> next-state samples for one capture segment."""
    if len(windows) < history_len + 1:
        empty = np.empty((0, history_len, 32), dtype=np.float32)
        return empty, np.empty((0, 32), dtype=np.float32), np.empty((0,), dtype=np.int64), scaler, []
    histories, targets, labels, metadata = [], [], [], []
    for anchor in range(history_len - 1, len(windows) - 1):
        history_windows = windows[anchor - history_len + 1 : anchor + 1]
        vectors = [np.asarray(states_lookup[item.window_idx].vector, dtype=np.float32) for item in history_windows]
        history_repr = np.stack(
            [state_velocity_representation(vector, vectors[index - 1] if index else None) for index, vector in enumerate(vectors)],
            axis=0,
        )
        next_window = windows[anchor + 1]
        next_state = np.asarray(states_lookup[next_window.window_idx].vector, dtype=np.float32)
        target_repr = state_velocity_representation(next_state, vectors[-1])
        histories.append(history_repr)
        targets.append(target_repr)
        labels.append(0 if next_window.is_empty else int(next_window.y))
        metadata.append({
            "anchor_window_idx": history_windows[-1].window_idx,
            "target_window_idx": next_window.window_idx,
            "is_onset_t1": bool(history_windows[-1].y == 0 and labels[-1] == 1),
        })
    X = np.asarray(histories, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if scaler is not None or fit_scaler:
        if fit_scaler:
            scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
        Y = scaler.transform(Y).astype(np.float32)
    return X, Y, y, scaler, metadata
