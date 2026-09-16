#!/usr/bin/env python
"""Validation-only threshold analysis for saved neural forecasting checkpoints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.run_temporal_dynamics_ablation import load_raw_capture, split_capture_windows  # noqa: E402
from src.data.temporal_targets import TemporalWindow  # noqa: E402
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.forecasting.learned_world_model import LearnedLatentWorldModel, build_transition_dataset  # noqa: E402
from src.models.fusion_model import TemporalGraphFusionModel  # noqa: E402
from src.models.graph_gnn import GraphSAGEModel  # noqa: E402
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402
from src.graph.communication_graph import build_communication_graphs  # noqa: E402

SCENARIO_42 = ROOT / "data/raw/ctu13/capture20110810.binetflow.txt"
SCENARIO_50 = ROOT / "data/raw/ctu13/capture20110817.binetflow"
TEMPORAL_CKPT = ROOT / "models/temporal_h10_state_velocity_best.pt"
TEMPORAL_SCALER = ROOT / "models/temporal_h10_state_velocity_scaler.pkl"
WORLD_CKPT = ROOT / "models/latent_world_model_h10_best.pt"
FUSION_CKPT = ROOT / "models/temporal_graph_fusion_best.pt"
GNN_CKPT = ROOT / "scratch/gnn_only_best.pt"
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel()]
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_predictions": int(np.sum(predictions == 1)),
    }


def rank_thresholds(scores: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates = [classification_metrics(scores, labels, threshold) for threshold in THRESHOLDS]
    # np.linspace is ascending, so ties deliberately retain the lowest threshold.
    selected = max(candidates, key=lambda row: row["f1"])
    return selected, candidates


def add_test_metrics(selected: Dict[str, Any], scores: np.ndarray, labels: np.ndarray, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(selected)
    result["test"] = classification_metrics(scores, labels, selected["threshold"])
    result["test"]["roc_auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) >= 2 else None
    result["test"]["pr_auc"] = float(average_precision_score(labels, scores)) if len(np.unique(labels)) >= 2 else None
    onset_indices = [index for index, item in enumerate(metadata) if item.get("is_onset_t1", False)]
    predictions = (scores >= selected["threshold"]).astype(int)
    detected = sum(int(predictions[index] == 1) for index in onset_indices)
    result["test"]["onsets"] = len(onset_indices)
    result["test"]["onsets_detected"] = detected
    result["test"]["onset_recall"] = float(detected / len(onset_indices)) if onset_indices else None
    result["score_semantics"] = "model score; not a calibrated probability"
    return result


def add_fixed_threshold_metrics(result: Dict[str, Any], scores: np.ndarray, labels: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
    fixed = classification_metrics(scores, labels, 0.5)
    fixed["roc_auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) >= 2 else None
    fixed["pr_auc"] = float(average_precision_score(labels, scores)) if len(np.unique(labels)) >= 2 else None
    onset_indices = [index for index, item in enumerate(metadata) if item.get("is_onset_t1", False)]
    predictions = (scores >= 0.5).astype(int)
    detected = sum(int(predictions[index] == 1) for index in onset_indices)
    fixed["onsets"] = len(onset_indices)
    fixed["onsets_detected"] = detected
    fixed["onset_recall"] = float(detected / len(onset_indices)) if onset_indices else None
    result["fixed_0_5_test"] = fixed


def temporal_scores(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def load_temporal_data(cap42, cap50):
    payload = torch.load(TEMPORAL_CKPT, map_location="cpu", weights_only=False)
    model = TemporalTransformerForecaster(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = joblib.load(TEMPORAL_SCALER)
    train_windows, val_windows, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    X_val, y_val, _, meta_val = build_temporal_dynamics_dataset(val_windows, cap42["states_lookup"], history_len=10, include_velocity=True, scaler=scaler)
    X_test, y_test, _, meta_test = build_temporal_dynamics_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=True, scaler=scaler)
    return temporal_scores(model, X_val), y_val, meta_val, temporal_scores(model, X_test), y_test, meta_test


def load_world_data(cap42, cap50):
    model, scaler, _ = LearnedLatentWorldModel.from_checkpoint(WORLD_CKPT)
    _, val_windows, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    X_val, _, y_val, _, meta_val = build_transition_dataset(val_windows, cap42["states_lookup"], history_len=10, scaler=scaler)
    X_test, _, y_test, _, meta_test = build_transition_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, scaler=scaler)
    with torch.no_grad():
        val_scores = model(torch.as_tensor(X_val, dtype=torch.float32))["attack_risk_score"].numpy()
        test_scores = model(torch.as_tensor(X_test, dtype=torch.float32))["attack_risk_score"].numpy()
    return val_scores, y_val, meta_val, test_scores, y_test, meta_test


def fusion_scores(model, norm_df, windows, X, metadata):
    selected_windows = [windows[item["anchor_window_idx"]] for item in metadata]
    graphs = build_communication_graphs(norm_df, selected_windows)
    records = []
    for sequence, graph_item in zip(X, graphs):
        nodes = graph_item["node_features"]
        edges = graph_item["edge_index"]
        if len(nodes) == 0:
            nodes = np.zeros((1, 6), dtype=np.float32)
            edges = np.empty((2, 0), dtype=np.int64)
        records.append(Data(x=torch.from_numpy(nodes).float(), edge_index=torch.from_numpy(edges).long(), temporal_x=torch.from_numpy(sequence).float().unsqueeze(0)))
    scores = []
    with torch.no_grad():
        for batch in GeoDataLoader(records, batch_size=32, shuffle=False):
            scores.extend(torch.sigmoid(model(batch.temporal_x, batch.x, batch.edge_index, batch.batch)).numpy().tolist())
    return np.asarray(scores)


def load_fusion_data(cap42, cap50):
    temporal_payload = torch.load(TEMPORAL_CKPT, map_location="cpu", weights_only=False)
    temporal = TemporalTransformerForecaster(**temporal_payload["model_config"])
    temporal.load_state_dict(temporal_payload["state_dict"])
    graph = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    graph.load_state_dict(torch.load(GNN_CKPT, map_location="cpu", weights_only=False))
    model = TemporalGraphFusionModel(temporal, graph, fusion_hidden_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    scaler = joblib.load(TEMPORAL_SCALER)
    _, val_windows, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    X_val, y_val, _, meta_val = build_temporal_dynamics_dataset(val_windows, cap42["states_lookup"], history_len=10, include_velocity=True, scaler=scaler)
    X_test, y_test, _, meta_test = build_temporal_dynamics_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=True, scaler=scaler)
    val_scores = fusion_scores(model, cap42["norm_df"], val_windows, X_val, meta_val)
    test_scores = fusion_scores(model, cap50["norm_df"], cap50["windows"], X_test, meta_test)
    return val_scores, y_val, meta_val, test_scores, y_test, meta_test


def main():
    started = time.perf_counter()
    cap42 = load_raw_capture(str(SCENARIO_42), "Scenario 42")
    cap50 = load_raw_capture(str(SCENARIO_50), "Scenario 50")
    model_loaders = {
        "Transformer H10 State + Velocity": load_temporal_data,
        "Learned latent World Model": load_world_data,
        "Transformer + GraphSAGE fusion": load_fusion_data,
    }
    results = []
    for name, loader in model_loaders.items():
        val_scores, val_labels, val_meta, test_scores, test_labels, test_meta = loader(cap42, cap50)
        selected, grid = rank_thresholds(val_scores, val_labels)
        result = add_test_metrics(selected, test_scores, test_labels, test_meta)
        add_fixed_threshold_metrics(result, test_scores, test_labels, test_meta)
        result.update({"model": name, "validation": selected, "validation_threshold_grid": grid, "validation_samples": len(val_labels), "test_samples": len(test_labels)})
        results.append(result)

    payload = {
        "methodology": {
            "train_development": "Scenario 42 chronological validation data only for threshold selection",
            "held_out_test": "Scenario 50 full forecastable timeline",
            "candidate_thresholds": [float(value) for value in THRESHOLDS],
            "selection_metric": "validation F1; ties retain lowest threshold",
            "test_labels_used_for_selection": False,
            "thresholds_are_calibration": False,
            "score_semantics": "model scores; not calibrated probabilities",
            "runtime_seconds": time.perf_counter() - started,
        },
        "results": results,
    }
    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "validation_threshold_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        rows.append({
            "model": result["model"], "threshold": result["threshold"],
            "validation_f1": result["validation"]["f1"], "validation_precision": result["validation"]["precision"],
            "validation_recall": result["validation"]["recall"], "validation_fpr": result["validation"]["fpr"],
            "test_precision": result["test"]["precision"], "test_recall": result["test"]["recall"],
            "test_f1": result["test"]["f1"], "test_fpr": result["test"]["fpr"],
            "test_roc_auc": result["test"]["roc_auc"], "test_pr_auc": result["test"]["pr_auc"],
            "test_tn": result["test"]["confusion_matrix"]["tn"], "test_fp": result["test"]["confusion_matrix"]["fp"],
            "test_fn": result["test"]["confusion_matrix"]["fn"], "test_tp": result["test"]["confusion_matrix"]["tp"],
            "test_onset_recall": result["test"]["onset_recall"], "test_onsets": result["test"]["onsets"],
            "fixed_0_5_test_f1": result["fixed_0_5_test"]["f1"],
            "fixed_0_5_test_fpr": result["fixed_0_5_test"]["fpr"],
            "fixed_0_5_test_recall": result["fixed_0_5_test"]["recall"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "validation_threshold_analysis.csv", index=False)

    markdown = ["| Model | Threshold | Validation F1 | Test F1 | Test FPR | Test Recall | Test PR-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['model']} | {row['threshold']:.2f} | {row['validation_f1']:.4f} | {row['test_f1']:.4f} | {row['test_fpr']:.4f} | {row['test_recall']:.4f} | {row['test_pr_auc']:.4f} |")
    (output_dir / "validation_threshold_analysis.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("VALIDATION-ONLY THRESHOLD ANALYSIS")
    for result in results:
        print(f"\n{result['model']}")
        print(f"  selected_threshold={result['threshold']:.2f}")
        print(f"  validation precision={result['validation']['precision']:.4f} recall={result['validation']['recall']:.4f} f1={result['validation']['f1']:.4f} fpr={result['validation']['fpr']:.4f}")
        print(f"  test precision={result['test']['precision']:.4f} recall={result['test']['recall']:.4f} f1={result['test']['f1']:.4f} fpr={result['test']['fpr']:.4f}")
        print(f"  test roc_auc={result['test']['roc_auc']:.4f} pr_auc={result['test']['pr_auc']:.4f} confusion={result['test']['confusion_matrix']}")
        print(f"  onset recall={result['test']['onsets_detected']}/{result['test']['onsets']} ({result['test']['onset_recall']})")
        print(f"  fixed-0.5 comparison: f1={result['fixed_0_5_test']['f1']:.4f} recall={result['fixed_0_5_test']['recall']:.4f} fpr={result['fixed_0_5_test']['fpr']:.4f}")
    print(f"\nJSON: {output_dir / 'validation_threshold_analysis.json'}")
    print(f"CSV: {output_dir / 'validation_threshold_analysis.csv'}")
    print(f"Markdown: {output_dir / 'validation_threshold_analysis.md'}")
    print(f"Runtime seconds: {payload['methodology']['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
