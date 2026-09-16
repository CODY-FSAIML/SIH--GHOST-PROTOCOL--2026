"""Fusion model for pretrained temporal and structural representations."""

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .graph_gnn import GraphSAGEModel
from .temporal_transformer import TemporalTransformerForecaster


class TemporalGraphFusionModel(nn.Module):
    """Fuse frozen Transformer and GraphSAGE representations with a trainable MLP."""

    def __init__(
        self,
        temporal_model: TemporalTransformerForecaster,
        graph_model: GraphSAGEModel,
        fusion_hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_model = temporal_model
        self.graph_model = graph_model
        self.temporal_representation_dim = temporal_model.d_model
        self.graph_representation_dim = graph_model.convs[-1].out_channels
        self.fusion_input_dim = self.temporal_representation_dim + self.graph_representation_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        for parameter in self.temporal_model.parameters():
            parameter.requires_grad = False
        for parameter in self.graph_model.parameters():
            parameter.requires_grad = False
        self.temporal_model.eval()
        self.graph_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.temporal_model.eval()
        self.graph_model.eval()
        return self

    def temporal_representation(self, temporal_x: torch.Tensor) -> torch.Tensor:
        h = self.temporal_model.input_proj(temporal_x)
        h = self.temporal_model.pos_encoder(h)
        encoded = self.temporal_model.transformer_encoder(h)
        return encoded[:, -1, :]

    def graph_representation(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> torch.Tensor:
        h = graph_x
        for conv in self.graph_model.convs:
            h = conv(h, graph_edge_index)
            h = self.graph_model.relu(h)
        return global_mean_pool(h, graph_batch)

    def forward(
        self,
        temporal_x: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            temporal_repr = self.temporal_representation(temporal_x)
            graph_repr = self.graph_representation(graph_x, graph_edge_index, graph_batch)
        fused = torch.cat([temporal_repr, graph_repr], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

    def representations(
        self,
        temporal_x: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return (
                self.temporal_representation(temporal_x),
                self.graph_representation(graph_x, graph_edge_index, graph_batch),
            )

    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())
