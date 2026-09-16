#!/usr/bin/env python
"""Reproducible held-out forecasting benchmark: Scenario 42 -> Scenario 50."""

from __future__ import annotations

import csv
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
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.forecasting.learned_world_model import LearnedLatentWorldModel, build_transition_dataset  # noqa: E402
from src.models.fusion_model import TemporalGraphFusionModel  # noqa: E402
from src.models.graph_gnn import GraphSAGEModel  # noqa: E402
from src.models.logistic_baseline import LogisticRegressionBaseline, build_baseline_dataset  # noqa: E402
from src.models.temporal_dataset import build_temporal_sequence_dataset  # noqa: E402
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402
from src.data.temporal_targets import build_forecasting_samples  # noqa: E402
from src.graph.communication_graph import build_communication_graphs  # noqa: E402

SCENARIO_42 = ROOT / "data/raw/ctu13/capture20110810.binetflow.txt"
SCENARIO_50 = ROOT / "data/raw/ctu13/capture20110817.binetflow"
TEMPORAL_CKPT = ROOT / "models/temporal_h10_state_velocity_best.pt"
TEMPORAL_SCALER = ROOT / "models/temporal_h10_state_velocity_scaler.pkl"
WORLD_CKPT = ROOT / "models/latent_world_model_h10_best.pt"
FUSION_CKPT = ROOT / "models/temporal_graph_fusion_best.pt"
GNN_CKPT = ROOT / "scratch/gnn_only_best.pt"
THRESHOLD_DEFAULT = 0.5


def _metrics(name: str, scores: np.ndarray, labels: np.ndarray, metadata: List[Dict[str, Any]], threshold: float, status: str = "evaluated") -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    both = len(np.unique(labels)) >= 2
    onset_indices = [index for index, item in enumerate(metadata) if item.get("is_onset_t1", False)]
    onset_detected = sum(int(predictions[index] == 1) for index in onset_indices)
    return {
        "model": name,
        "status": status,
        "threshold": threshold,
        "threshold_policy": "0.5 default or validation-selected; never selected on Scenario 50",
        "samples": int(len(labels)),
        "positive_labels": int(np.sum(labels == 1)),
        "negative_labels": int(np.sum(labels == 0)),
        "attack_window_prevalence": float(np.mean(labels == 1)) if len(labels) else None,
        "positive_predictions": int(np.sum(predictions == 1)),
        "negative_predictions": int(np.sum(predictions == 0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)) if len(labels) else None,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if len(labels) else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)) if len(labels) else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "pr_auc": float(average_precision_score(labels, scores)) if both else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "onsets": len(onset_indices),
        "onsets_detected": int(onset_detected),
        "onset_recall": float(onset_detected / len(onset_indices)) if onset_indices else None,
        "score_semantics": "model score; not a calibrated probability",
    }


def _state_only_unavailable() -> Dict[str, Any]:
    return {"model": "Transformer H10 State-only", "status": "unavailable", "reason": "No saved compatible state-only Transformer checkpoint exists; evaluation did not retrain it."}


def _load_temporal_test(cap50):
    payload = torch.load(TEMPORAL_CKPT, map_location="cpu", weights_only=False)
    model = TemporalTransformerForecaster(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = joblib.load(TEMPORAL_SCALER)
    X, y, _, meta = build_temporal_dynamics_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=True, scaler=scaler)
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()
    return scores, y, meta


def _load_world_test(cap50):
    model, scaler, _ = LearnedLatentWorldModel.from_checkpoint(WORLD_CKPT)
    X, _, y, _, meta = build_transition_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, scaler=scaler)
    with torch.no_grad():
        scores = model(torch.as_tensor(X, dtype=torch.float32))["attack_risk_score"].numpy()
    return scores, y, meta


def _load_fusion_test(cap50):
    temporal_payload = torch.load(TEMPORAL_CKPT, map_location="cpu", weights_only=False)
    temporal = TemporalTransformerForecaster(**temporal_payload["model_config"])
    temporal.load_state_dict(temporal_payload["state_dict"])
    graph = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    graph.load_state_dict(torch.load(GNN_CKPT, map_location="cpu", weights_only=False))
    fusion = TemporalGraphFusionModel(temporal, graph, fusion_hidden_dim=32, dropout=0.1)
    fusion.load_state_dict(torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)["state_dict"])
    fusion.eval()
    scaler = joblib.load(TEMPORAL_SCALER)
    X, _, y, _, meta = build_transition_dataset(cap50["windows"], cap50["states_lookup"], history_len=10, scaler=scaler)
    # build_transition_dataset and temporal dynamics use the same 32D H10 representation.
    graphs = build_communication_graphs(cap50["norm_df"], [cap50["windows"][item["anchor_window_idx"]] for item in meta])
    data = []
    for sequence, label, graph_item in zip(X, y, graphs):
        node_features = graph_item["node_features"]
        edge_index = graph_item["edge_index"]
        if len(node_features) == 0:
            node_features = np.zeros((1, 6), dtype=np.float32)
            edge_index = np.empty((2, 0), dtype=np.int64)
        data.append(Data(x=torch.from_numpy(node_features).float(), edge_index=torch.from_numpy(edge_index).long(), temporal_x=torch.from_numpy(sequence).float().unsqueeze(0), y=torch.tensor(float(label))))
    loader = GeoDataLoader(data, batch_size=32, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.extend(torch.sigmoid(fusion(batch.temporal_x, batch.x, batch.edge_index, batch.batch)).numpy().tolist())
    return np.asarray(scores), y, meta


def _load_logistic(cap42, cap50):
    train_windows, val_windows, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    # Preserve the existing baseline's H=3 ForecastingSample construction and train-only scaler.
    train_samples = build_forecasting_samples(train_windows, history_len=3, horizons=(1,))
    val_samples = build_forecasting_samples(val_windows, history_len=3, horizons=(1,))
    test_samples = build_forecasting_samples(cap50["windows"], history_len=3, horizons=(1,))
    train_lookup = {index: state for index, state in enumerate([cap42["states_lookup"][window.window_idx] for window in train_windows])}
    val_lookup = {index: state for index, state in enumerate([cap42["states_lookup"][window.window_idx] for window in val_windows])}
    X_train, y_train, _, train_meta = build_baseline_dataset(train_samples, train_lookup)
    X_val, y_val, _, val_meta = build_baseline_dataset(val_samples, val_lookup)
    # Scenario 50 starts at window zero, matching the existing sample/index convention.
    X_test, y_test, _, test_meta = build_baseline_dataset(test_samples, cap50["states_lookup"])
    model = LogisticRegressionBaseline(class_weight="balanced", random_state=42).fit(X_train, y_train)
    threshold, valid, threshold_note = model.calibrate_threshold(X_val, y_val, metric="f1")
    scores = model.predict_proba(X_test)
    result = _metrics("Logistic Regression baseline (H3 State)", scores, y_test, test_meta, threshold)
    result["validation_threshold_calibration"] = {"valid": valid, "note": threshold_note, "validation_samples": len(y_val)}
    return result


def main():
    started = time.perf_counter()
    cap42 = load_raw_capture(str(SCENARIO_42), "Scenario 42")
    cap50 = load_raw_capture(str(SCENARIO_50), "Scenario 50")
    results = [_load_logistic(cap42, cap50)]

    temporal_scores, temporal_labels, temporal_meta = _load_temporal_test(cap50)
    results.append(_metrics("Transformer H10 State + Velocity", temporal_scores, temporal_labels, temporal_meta, THRESHOLD_DEFAULT))
    results.append(_state_only_unavailable())

    world_scores, world_labels, world_meta = _load_world_test(cap50)
    results.append(_metrics("Learned latent World Model", world_scores, world_labels, world_meta, THRESHOLD_DEFAULT))

    fusion_scores, fusion_labels, fusion_meta = _load_fusion_test(cap50)
    results.append(_metrics("Transformer + GraphSAGE fusion", fusion_scores, fusion_labels, fusion_meta, THRESHOLD_DEFAULT))

    evaluation = {
        "methodology": {
            "train_development": "CTU-13 Scenario 42 chronological train/validation",
            "held_out_test": "CTU-13 Scenario 50 full forecastable timeline",
            "window_size_seconds": 60,
            "target": "Y(t+1)",
            "threshold_policy": "Validation-only threshold selection for Logistic Regression; fixed 0.5 for saved neural/world/fusion checkpoints",
            "test_tuning": False,
            "risk_score_semantics": "Scores are not calibrated probabilities",
            "runtime_seconds": time.perf_counter() - started,
        },
        "results": results,
    }
    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "final_forecasting_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"confusion_matrix", "validation_threshold_calibration"}}
        if "confusion_matrix" in result:
            row.update({f"cm_{key}": value for key, value in result["confusion_matrix"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "final_forecasting_evaluation.csv", index=False)

    markdown = ["| Model | Status | Threshold | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | Confusion | Pos pred | Prevalence | Onset recall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for result in results:
        if result["status"] != "evaluated":
            markdown.append(f"| {result['model']} | unavailable | - | - | - | - | - | - | - | {result['reason']} | - | - | - |")
            continue
        cm = result["confusion_matrix"]
        markdown.append(f"| {result['model']} | evaluated | {result['threshold']:.2f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {result['false_positive_rate']:.4f} | {result['roc_auc']:.4f} | {result['pr_auc']:.4f} | TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']} | {result['positive_predictions']} | {result['attack_window_prevalence']:.4f} | {result['onsets_detected']}/{result['onsets']} |")
    (output_dir / "final_forecasting_evaluation.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("FINAL FORECASTING EVALUATION")
    print(f"Methodology: Scenario 42 train/development -> Scenario 50 held-out test; threshold policy fixed before test")
    for result in results:
        print(f"\n{result['model']} [{result['status']}]")
        if result["status"] != "evaluated":
            print(f"  {result['reason']}")
            continue
        print(f"  threshold={result['threshold']:.3f} samples={result['samples']} prevalence={result['attack_window_prevalence']:.4f} positive_predictions={result['positive_predictions']}")
        print(f"  precision={result['precision']:.4f} recall={result['recall']:.4f} f1={result['f1']:.4f} fpr={result['false_positive_rate']:.4f}")
        print(f"  roc_auc={result['roc_auc']} pr_auc={result['pr_auc']} confusion={result['confusion_matrix']}")
        print(f"  onset_recall={result['onsets_detected']}/{result['onsets']} ({result['onset_recall']})")
    print(f"\nJSON: {output_dir / 'final_forecasting_evaluation.json'}")
    print(f"CSV: {output_dir / 'final_forecasting_evaluation.csv'}")
    print(f"Markdown: {output_dir / 'final_forecasting_evaluation.md'}")
    print(f"Runtime seconds: {evaluation['methodology']['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
