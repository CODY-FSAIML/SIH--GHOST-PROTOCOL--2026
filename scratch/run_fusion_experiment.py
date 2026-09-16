#!/usr/bin/env python
"""Run the first pretrained Transformer + GraphSAGE fusion experiment."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_temporal_dynamics_ablation import (  # noqa: E402
    SCENARIO_42_PATH,
    SCENARIO_50_PATH,
    load_raw_capture,
    split_capture_windows,
)
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.graph.communication_graph import build_communication_graphs  # noqa: E402
from src.models.fusion_model import TemporalGraphFusionModel  # noqa: E402
from src.models.graph_gnn import GraphSAGEModel  # noqa: E402
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402
from src.models.temporal_dataset import compute_pos_weight  # noqa: E402

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
TEMPORAL_CHECKPOINT = PROJECT_ROOT / "models" / "temporal_h10_state_velocity_best.pt"
TEMPORAL_SCALER = PROJECT_ROOT / "models" / "temporal_h10_state_velocity_scaler.pkl"
GNN_CHECKPOINT = PROJECT_ROOT / "scratch" / "gnn_only_best.pt"
FUSION_CHECKPOINT = PROJECT_ROOT / "models" / "temporal_graph_fusion_best.pt"


def load_pretrained_backbones() -> Tuple[TemporalTransformerForecaster, GraphSAGEModel]:
    temporal_payload = torch.load(TEMPORAL_CHECKPOINT, map_location="cpu", weights_only=False)
    temporal_config = temporal_payload["model_config"]
    temporal_model = TemporalTransformerForecaster(**temporal_config)
    temporal_model.load_state_dict(temporal_payload["state_dict"])

    graph_model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    graph_state = torch.load(GNN_CHECKPOINT, map_location="cpu", weights_only=False)
    graph_model.load_state_dict(graph_state)
    return temporal_model, graph_model


def graph_data_for_windows(norm_df, windows: List[Any], labels: np.ndarray, temporal_x: np.ndarray) -> List[Data]:
    graphs = build_communication_graphs(norm_df, windows)
    data_list = []
    for graph, label, sequence in zip(graphs, labels, temporal_x):
        node_features = graph["node_features"]
        edge_index = graph["edge_index"]
        if node_features.shape[0] == 0:
            node_features = np.zeros((1, 6), dtype=np.float32)
            edge_index = np.empty((2, 0), dtype=np.int64)
        data = Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            temporal_x=torch.from_numpy(sequence).float().unsqueeze(0),
            y=torch.tensor(int(label), dtype=torch.float32),
        )
        data_list.append(data)
    return data_list


def build_split_data(norm_df, windows, states_lookup, history_len=10, scaler=None, fit_scaler=False):
    X, y, scaler, meta = build_temporal_dynamics_dataset(
        windows,
        states_lookup,
        history_len=history_len,
        include_velocity=True,
        scaler=scaler,
        fit_scaler=fit_scaler,
    )
    anchor_windows = [windows[item["anchor_window_idx"]] for item in meta]
    return X, y, meta, anchor_windows, scaler


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.temporal_x, batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, criterion):
    model.eval()
    losses, logits, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_logits = model(batch.temporal_x, batch.x, batch.edge_index, batch.batch)
            losses.append(float(criterion(batch_logits, batch.y).item()))
            logits.append(batch_logits.cpu())
            labels.append(batch.y.cpu())
    return (
        float(np.mean(losses)) if losses else 0.0,
        torch.cat(logits).numpy() if logits else np.empty(0),
        torch.cat(labels).numpy() if labels else np.empty(0),
    )


def metrics(logits: np.ndarray, labels: np.ndarray, meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = 1.0 / (1.0 + np.exp(-logits))
    predictions = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    onset_indices = [i for i, item in enumerate(meta) if item.get("is_onset_t1", False)]
    detected = sum(predictions[i] == 1 for i in onset_indices)
    both_classes = len(np.unique(labels)) >= 2
    return {
        "sample_count": int(len(labels)),
        "pos_count": int(np.sum(labels == 1)),
        "neg_count": int(np.sum(labels == 0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else None,
        "total_onsets": len(onset_indices),
        "onsets_detected": int(detected),
        "onset_recall": float(detected / len(onset_indices)) if onset_indices else 0.0,
    }


def main():
    started = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not TEMPORAL_CHECKPOINT.exists() or not TEMPORAL_SCALER.exists() or not GNN_CHECKPOINT.exists():
        raise FileNotFoundError("Required verified temporal, scaler, or GNN checkpoint is missing")

    temporal_model, graph_model = load_pretrained_backbones()
    fusion_model = TemporalGraphFusionModel(temporal_model, graph_model, fusion_hidden_dim=32, dropout=0.1)
    trainable_parameters = TemporalGraphFusionModel.count_trainable_parameters(fusion_model)
    total_parameters = TemporalGraphFusionModel.count_parameters(fusion_model)

    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")
    w42_train, w42_val, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)

    train_x, train_y, train_meta, train_anchor_windows, scaler = build_split_data(
        cap42["norm_df"], w42_train, cap42["states_lookup"], fit_scaler=True
    )
    val_x, val_y, val_meta, val_anchor_windows, _ = build_split_data(
        cap42["norm_df"], w42_val, cap42["states_lookup"], scaler=scaler
    )
    # The scaler is loaded from the verified checkpoint artifact and must match the ablation fit.
    import joblib
    saved_scaler = joblib.load(TEMPORAL_SCALER)
    if not np.allclose(saved_scaler.mean_, scaler.mean_) or not np.allclose(saved_scaler.scale_, scaler.scale_):
        raise RuntimeError("Scenario 42 scaler does not match the saved verified temporal scaler")
    train_x, _, _, _ = build_temporal_dynamics_dataset(w42_train, cap42["states_lookup"], history_len=10, include_velocity=True, scaler=saved_scaler)
    val_x, _, _, _ = build_temporal_dynamics_dataset(w42_val, cap42["states_lookup"], history_len=10, include_velocity=True, scaler=saved_scaler)

    train_data = graph_data_for_windows(cap42["norm_df"], train_anchor_windows, train_y, train_x)
    val_data = graph_data_for_windows(cap42["norm_df"], val_anchor_windows, val_y, val_x)
    train_loader = GeoDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = compute_pos_weight(train_y)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        fusion_model.fusion_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    training_start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(fusion_model, train_loader, criterion, optimizer)
        val_loss, _, _ = evaluate(fusion_model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.cpu().clone() for key, value in fusion_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val loss {best_loss:.6f})")
                break
    training_time = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("No best fusion state was produced")
    fusion_model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": fusion_model.state_dict(),
            "temporal_checkpoint": str(TEMPORAL_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "gnn_checkpoint": str(GNN_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "fusion_hidden_dim": 32,
            "dropout": 0.1,
            "temporal_representation_dim": 64,
            "graph_representation_dim": 32,
            "trainable_parameter_count": trainable_parameters,
            "total_parameter_count": total_parameters,
        },
        FUSION_CHECKPOINT,
    )

    # Scenario 50 is constructed and evaluated only after fitting the fusion head.
    test_x, test_y, test_meta, test_anchor_windows, _ = build_split_data(
        cap50["norm_df"], cap50["windows"], cap50["states_lookup"]
    )
    test_x, _, _, _ = build_temporal_dynamics_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=True, scaler=saved_scaler
    )
    test_data = graph_data_for_windows(cap50["norm_df"], test_anchor_windows, test_y, test_x)
    test_loader = GeoDataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    _, test_logits, test_labels = evaluate(fusion_model, test_loader, criterion)
    test_metrics = metrics(test_logits, test_labels, test_meta)
    preprocessing_runtime = time.perf_counter() - started - training_time
    total_runtime = time.perf_counter() - started

    print(json.dumps({
        "temporal_representation_dim": 64,
        "gnn_representation_dim": 32,
        "fusion_dimension": 96,
        "fusion_architecture": "Linear(96,32) -> ReLU -> Dropout(0.1) -> Linear(32,1)",
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "train_samples": len(train_y),
        "validation_samples": len(val_y),
        "test_samples": len(test_y),
        "epochs": epoch,
        "preprocessing_runtime_seconds": preprocessing_runtime,
        "training_runtime_seconds": training_time,
        "total_runtime_seconds": total_runtime,
        "checkpoint": str(FUSION_CHECKPOINT),
        "pos_weight": pos_weight,
        "metrics": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
