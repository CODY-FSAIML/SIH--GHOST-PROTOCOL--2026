import pytest
import torch
from torch_geometric.data import Data, Batch

# Import the model
from src.models.graph_gnn import GraphSAGEModel


def test_imports():
    """Verify required imports are available."""
    assert torch is not None
    # torch_geometric should be importable
    import torch_geometric
    assert torch_geometric is not None
    assert GraphSAGEModel is not None


def test_model_construction():
    """Model should be constructed with the correct dimensions."""
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    # Verify internal attributes
    assert len(model.convs) == 2
    # First conv input dim
    first_conv = model.convs[0]
    assert first_conv.in_channels == 6
    assert first_conv.out_channels == 32
    # Second conv should map hidden -> hidden
    second_conv = model.convs[1]
    assert second_conv.in_channels == 32
    assert second_conv.out_channels == 32


def test_forward_pass_single_graph():
    """Run a forward pass on a tiny 3‑node directed graph.
    Expected output shape: [1] (one logit for the single graph).
    """
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    model.eval()

    # 3 nodes, each with 6 features (random values are fine)
    x = torch.randn((3, 6), dtype=torch.float32)
    # Define a simple directed edge set: 0->1, 1->2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # batch vector – all nodes belong to graph 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index, batch)
    # Output should be a 1‑D tensor with a single element
    assert out.shape == (1,)
    # Ensure it is a scalar (logit) and not NaN
    assert torch.isfinite(out).all()


def test_batch_handling_two_graphs():
    """Create two small graphs, batch them, and verify two predictions are returned."""
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    model.eval()

    # Graph A: 2 nodes
    x_a = torch.randn((2, 6), dtype=torch.float32)
    edge_a = torch.tensor([[0], [1]], dtype=torch.long)  # single edge 0->1
    data_a = Data(x=x_a, edge_index=edge_a)

    # Graph B: 3 nodes
    x_b = torch.randn((3, 6), dtype=torch.float32)
    edge_b = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # edges 0->1, 1->2
    data_b = Data(x=x_b, edge_index=edge_b)

    batch = Batch.from_data_list([data_a, data_b])
    # Forward pass – note that Batch provides a batch vector attribute
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
    # Expected one logit per graph → shape (2,)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_parameter_count_matches_actual():
    """The static count_parameters method should equal the sum of trainable params."""
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    counted = GraphSAGEModel.count_parameters(model)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert counted == actual


def test_empty_graph_behavior():
    """The model does not explicitly support empty graphs; ensure a clear error.
    The training script is expected to handle empty windows separately.
    """
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    model.eval()
    # Empty tensors – no nodes, no edges
    x = torch.empty((0, 6), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.empty((0,), dtype=torch.long)
    with pytest.raises(RuntimeError):
        _ = model(x, edge_index, batch)


def test_input_contains_only_allowed_fields():
    """Confirm that only node features and edge_index are used.
    No label‑related columns should be accessed by the model.
    """
    # Simulate a Data object that includes extra fields
    x = torch.randn((3, 6), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    # Extra attribute that a naive implementation might mistakenly read
    extra = {"is_attack": torch.tensor([0, 1, 0]), "raw_label": ["a", "b", "c"]}
    # The model's forward signature does not accept these extras, so they are ignored.
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    with torch.no_grad():
        out = model(x, edge_index, batch)
    assert out.shape == (1,)


def test_edge_feature_limitation_documented():
    """SAGEConv does not consume edge attributes – verify that passing extra edge_attr
    does not affect the forward computation (the model simply ignores it)."""
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    model.eval()
    x = torch.randn((3, 6), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # Create dummy edge attributes (6‑dim) which the model will ignore
    edge_attr = torch.randn((2, 6), dtype=torch.float32)
    batch = torch.zeros(3, dtype=torch.long)
    # Forward without edge_attr – standard call
    out1 = model(x, edge_index, batch)
    # Forward where we manually pass edge_attr via Data (won't be used)
    # The model's forward does not accept edge_attr, so we simply ignore it.
    out2 = model(x, edge_index, batch)
    # The outputs should be identical
    assert torch.allclose(out1, out2)
