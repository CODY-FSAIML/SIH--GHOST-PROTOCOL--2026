import torch

from src.models.fusion_model import TemporalGraphFusionModel
from src.models.graph_gnn import GraphSAGEModel
from src.models.temporal_transformer import TemporalTransformerForecaster


def test_fusion_dimensions_and_trainable_head():
    temporal = TemporalTransformerForecaster(input_dim=32, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
    graph = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    model = TemporalGraphFusionModel(temporal, graph, fusion_hidden_dim=32, dropout=0.1)

    assert model.temporal_representation_dim == 64
    assert model.graph_representation_dim == 32
    assert model.fusion_input_dim == 96
    assert TemporalGraphFusionModel.count_trainable_parameters(model) == (96 * 32 + 32 + 32 + 1)

    temporal_x = torch.randn(2, 10, 32)
    graph_x = torch.randn(5, 6)
    graph_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    graph_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    logits = model(temporal_x, graph_x, graph_edge_index, graph_batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_backbones_are_frozen():
    temporal = TemporalTransformerForecaster(input_dim=32)
    graph = GraphSAGEModel()
    model = TemporalGraphFusionModel(temporal, graph)
    assert all(not parameter.requires_grad for parameter in model.temporal_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.graph_model.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion_head.parameters())
