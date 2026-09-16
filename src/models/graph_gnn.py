import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGEModel(nn.Module):
    """GraphSAGE model for binary attack prediction.

    - Consumes only node features (6-dimensional as defined in communication_graph).
    - Edge attributes are **not** used (SAGEConv does not take them). This limitation is documented.
    - Global mean pooling yields a graph-level representation.
    - Final linear layer outputs a single logit.
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.convs = nn.ModuleList()
        # First convolution layer
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        # Additional layers (if any)
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = global_mean_pool
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Guard against empty graphs – the model expects at least one node.
        if x.size(0) == 0:
            raise RuntimeError("Empty graph with no nodes is not supported.")
        """Forward pass.
        Args:
            x (Tensor): Node feature matrix [N, in_dim]
            edge_index (Tensor): Edge index [2, E]
            batch (Tensor): Batch vector mapping each node to its graph.
        Returns:
            Tensor: Logits of shape [batch_size]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        # Graph‑level pooling
        x = self.pool(x, batch)
        logit = self.classifier(x).squeeze(-1)
        return logit

    @staticmethod
    def count_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
