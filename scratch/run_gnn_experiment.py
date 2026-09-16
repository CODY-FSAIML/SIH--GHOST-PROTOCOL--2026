#!/usr/bin/env python
"""Run the GraphSAGE GNN experiment (SIH project).

Implements the approved GNN‑only experiment:
- GraphSAGE with 2 layers, hidden_dim=32
- Global mean pooling, binary logit output
- Edge attributes are ignored (SAGEConv does not consume them)
- Chronological train/validation split on Scenario 42
- Scenario 50 is held‑out test (may contain empty windows)
- Early stopping on validation loss (patience=5)
- Logs training loss, validation loss, precision, recall per epoch
- Saves best checkpoint to ``scratch/gnn_only_best.pt``
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.adapters.ctu13_adapter import CTU13Adapter, _state_category
from src.graph.communication_graph import build_communication_graph, build_communication_graphs
from src.models.graph_gnn import GraphSAGEModel
from src.models.temporal_dataset import compute_pos_weight, build_temporal_sequence_dataset, chronological_split

# ---------------------------------------------------------------------------
# Helper: load raw capture and build windows (same as transformer script)
# ---------------------------------------------------------------------------

def load_raw_capture(filepath: str, name: str) -> Dict[str, Any]:
    print(f"Loading {name} from {filepath}…")
    t0 = time.time()
    cols = [
        "StartTime",
        "Dur",
        "Proto",
        "SrcAddr",
        "DstAddr",
        "TotPkts",
        "TotBytes",
        "SrcBytes",
        "State",
        "Label",
    ]
    df_raw = pd.read_csv(filepath, usecols=cols, low_memory=False)
    print(f"  Loaded {len(df_raw):,} flows in {time.time() - t0:.2f}s")

    dt = pd.to_datetime(df_raw["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
    ts = dt.astype("datetime64[ns]").astype(np.int64) / 1e9
    labels = df_raw["Label"].astype(str)
    is_attack = labels.str.contains("Botnet", case=True, regex=False).astype(int)
    state_cats = [_state_category(str(s)) for s in df_raw["State"]]

    from src.features.network_state import MacroNetworkStateBuilder
    builder = MacroNetworkStateBuilder(window_size_sec=60.0)
    norm_df = pd.DataFrame({
        "timestamp": ts,
        "duration": pd.to_numeric(df_raw["Dur"], errors="coerce").fillna(0.0),
        "protocol": df_raw["Proto"].astype(str).str.lower().str.strip(),
        "src_ip": df_raw["SrcAddr"].astype(str).str.strip(),
        "dst_ip": df_raw["DstAddr"].astype(str).str.strip(),
        "total_packets": pd.to_numeric(df_raw["TotPkts"], errors="coerce").fillna(0),
        "total_bytes": pd.to_numeric(df_raw["TotBytes"], errors="coerce").fillna(0),
        "src_bytes": pd.to_numeric(df_raw["SrcBytes"], errors="coerce").fillna(0),
        "state_category": state_cats,
        "raw_label": labels,
        "is_attack": is_attack,
    })
    states, _, _ = builder.build_states(norm_df)
    states_lookup = {s.window_idx: s for s in states}
    from src.data.temporal_targets import build_temporal_windows
    windows = build_temporal_windows(norm_df, window_size_sec=60.0)
    return {"name": name, "norm_df": norm_df, "states_lookup": states_lookup, "windows": windows}

# ---------------------------------------------------------------------------
# Convert windows to torch_geometric Data objects (graph + label)
# ---------------------------------------------------------------------------

def windows_to_graph_data(
    windows: List[Any],
    states_lookup: Dict[int, Any],
    norm_df: pd.DataFrame,
    scenario_name: str = "",
) -> Tuple[List[torch.utils.data.Dataset], List[int]]:
    """Transform each window (except the last) into a graph and a target label.

    Empty windows are represented by a single dummy node with zero features
    so that batching works. The target label is the attack flag of the next
    window (Y_{t+1}).

    This function scans the entire normalized DataFrame for each window
    (via ``norm_df.iloc[flow_indices]``), which is a known performance bottleneck
    because the DataFrame contains millions of rows.
    """
    from torch_geometric.data import Data
    data_list: List[Data] = []
    labels: List[int] = []
    total = len(windows) - 1
    start_time = time.time()
    for t in range(total):
        cur_win = windows[t]
        target_w = windows[t + 1]
        # Retrieve the rows belonging to the current window using stored indices
        df_window = norm_df.iloc[list(cur_win.flow_indices)] if hasattr(cur_win, "flow_indices") else pd.DataFrame()
        graph = build_communication_graph(df_window)
        node_feat = graph["node_features"]  # (N, 6)
        edge_index = graph["edge_index"]
        if node_feat.shape[0] == 0:
            node_feat = np.zeros((1, 6), dtype=np.float32)
            edge_index = np.empty((2, 0), dtype=np.int64)
        data = Data(x=torch.from_numpy(node_feat).float(), edge_index=torch.from_numpy(edge_index).long())
        data.y = torch.tensor(int(target_w.is_empty == False and target_w.y == 1), dtype=torch.long)
        data_list.append(data)
        labels.append(int(target_w.is_empty == False and target_w.y == 1))
        # Progress logging every 50 windows (or the last one)
        if (t + 1) % 50 == 0 or (t + 1) == total:
            elapsed = time.time() - start_time
            avg_per = elapsed / (t + 1)
            remaining = (total - (t + 1)) * avg_per
            print(f"[{scenario_name}] Building graph {t + 1}/{total} – elapsed {elapsed:.1f}s, avg {avg_per:.3f}s/win, est. remaining {remaining/60:.1f}min")
    return data_list, labels

# ---------------------------------------------------------------------------
# Optimized version: pre‑extract window slices to avoid repeated iloc calls.
# ---------------------------------------------------------------------------
def windows_to_graph_data_optimized(
    windows: List[Any],
    states_lookup: Dict[int, Any],
    norm_df: pd.DataFrame,
    scenario_name: str = "",
) -> Tuple[List[torch.utils.data.Dataset], List[int]]:
    """Optimized conversion that extracts each window's rows once.

    The original implementation performed ``norm_df.iloc[flow_indices]`` for every
    window, which caused O(W·M) work. Here we pre‑compute a list of DataFrames (or
    NumPy arrays) for all windows up‑front, then iterate without further
    slicing.
    """
    from torch_geometric.data import Data
    optimized_graphs = build_communication_graphs(norm_df, list(windows[:-1]))
    data_list: List[Data] = []
    labels: List[int] = []
    total = len(windows) - 1
    start_time = time.time()
    for t in range(total):
        target_w = windows[t + 1]
        graph = optimized_graphs[t]
        node_feat = graph["node_features"]
        edge_index = graph["edge_index"]
        if node_feat.shape[0] == 0:
            node_feat = np.zeros((1, 6), dtype=np.float32)
            edge_index = np.empty((2, 0), dtype=np.int64)
        data = Data(x=torch.from_numpy(node_feat).float(), edge_index=torch.from_numpy(edge_index).long())
        data.y = torch.tensor(int(target_w.is_empty == False and target_w.y == 1), dtype=torch.long)
        data_list.append(data)
        labels.append(int(target_w.is_empty == False and target_w.y == 1))
        if (t + 1) % 50 == 0 or (t + 1) == total:
            elapsed = time.time() - start_time
            avg_per = elapsed / (t + 1)
            remaining = (total - (t + 1)) * avg_per
            print(f"[{scenario_name}] (opt) Building graph {t + 1}/{total} – elapsed {elapsed:.1f}s, avg {avg_per:.3f}s/win, est. remaining {remaining/60:.1f}min")
    return data_list, labels

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def evaluate(model, loader, criterion, device):
    model.eval()
    losses, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())
            losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    logits = torch.cat(all_logits) if all_logits else torch.tensor([])
    labels = torch.cat(all_labels) if all_labels else torch.tensor([])
    return avg_loss, logits, labels

# ---------------------------------------------------------------------------
# Metric calculation (mirrors transformer script)
# ---------------------------------------------------------------------------

def compute_metrics(probs: np.ndarray, labels: np.ndarray, sample_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    has_both = len(np.unique(labels)) >= 2
    roc_auc = roc_auc_score(labels, probs) if has_both else None
    pr_auc = average_precision_score(labels, probs) if has_both else None
    onset_indices = [i for i, m in enumerate(sample_meta) if m.get("is_onset_t1", False)]
    total_onsets = len(onset_indices)
    onsets_detected = sum(preds[i] == 1 for i in onset_indices)
    onsets_missed = total_onsets - onsets_detected
    onset_recall = onsets_detected / total_onsets if total_onsets else 0.0
    benign_indices = [i for i, m in enumerate(sample_meta) if (not m.get("is_onset_t1", False)) and labels[i] == 0]
    false_alarms_benign = sum(preds[i] == 1 for i in benign_indices)
    onset_details = []
    for idx in onset_indices:
        m = sample_meta[idx]
        onset_details.append({
            "anchor_window": m["anchor_window_idx"],
            "target_window": m["target_window_idx"],
            "prob": float(probs[idx]),
            "predicted": int(preds[idx]),
            "detected": bool(preds[idx] == 1),
        })
    return {
        "sample_count": int(len(labels)),
        "pos_count": int(np.sum(labels == 1)),
        "neg_count": int(np.sum(labels == 0)),
        "pred_pos": int(np.sum(preds == 1)),
        "pred_neg": int(np.sum(preds == 0)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "total_onsets": total_onsets,
        "onsets_detected": onsets_detected,
        "onsets_missed": onsets_missed,
        "onset_recall": onset_recall,
        "false_alarms_benign": false_alarms_benign,
        "onset_details": onset_details,
    }

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    experiment_start = time.perf_counter()
    torch.manual_seed(42)
    np.random.seed(42)

    SCENARIO_42_PATH = r"c:\\Users\\sridevi\\OneDrive\\Desktop\\sih\\data\\raw\\ctu13\\capture20110810.binetflow.txt"
    SCENARIO_50_PATH = r"c:\\Users\\sridevi\\OneDrive\\Desktop\\sih\\data\\raw\\ctu13\\capture20110817.binetflow"

    print("[1/5] Loading Scenario 42...")
    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    print("[3/5] Loading Scenario 50...")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")

    w42_train, w42_val, w42_test = chronological_split(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    w50_all = cap50["windows"]

    print("[2/5] Building Scenario 42 graphs...")
    train_data, train_labels = windows_to_graph_data_optimized(w42_train, cap42["states_lookup"], cap42["norm_df"], scenario_name="Scenario 42")
    val_data, val_labels = windows_to_graph_data_optimized(w42_val, cap42["states_lookup"], cap42["norm_df"], scenario_name="Scenario 42")
    print("[4/5] Building Scenario 50 graphs...")
    test_data, test_labels = windows_to_graph_data_optimized(w50_all, cap50["states_lookup"], cap50["norm_df"], scenario_name="Scenario 50") 
    preprocessing_time = time.perf_counter() - experiment_start
    print(f"Preprocessing time: {preprocessing_time:.2f}s")

    from torch_geometric.loader import DataLoader as GeoDataLoader
    batch_size = 32
    train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2).to(device)

    pos_weight = compute_pos_weight(np.array(train_labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0
    max_epochs = 50
    history = {"train_loss": [], "val_loss": []}

    print("[5/5] Training/evaluating GNN...")
    training_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_logits, val_labels_tensor = evaluate(model, val_loader, criterion, device)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_labels_np = val_labels_tensor.cpu().numpy()
        val_prec = precision_score(val_labels_np, (val_probs >= 0.5).astype(int), zero_division=0)
        val_rec = recall_score(val_labels_np, (val_probs >= 0.5).astype(int), zero_division=0)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_prec={val_prec:.4f} | val_rec={val_rec:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})")
                break

    training_time = time.perf_counter() - training_start
    print(f"Training time: {training_time:.2f}s")

    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = Path(__file__).parent / "gnn_only_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"Best checkpoint saved to {ckpt_path}")

    # Final evaluation on validation and test sets
    def run_eval(loader, windows, states_lookup, name):
        loss, logits, labels_tensor = evaluate(model, loader, criterion, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels_tensor.cpu().numpy()
        _, _, _, sample_meta = build_temporal_sequence_dataset(windows, states_lookup, history_len=1, fit_scaler=False)
        metrics = compute_metrics(probs, labels_np, sample_meta)
        metrics["description"] = name
        metrics["validation_loss"] = loss
        return metrics

    val_metrics = run_eval(val_loader, w42_val, cap42["states_lookup"], "Scenario 42 Validation")
    print("\n=== Validation Results ===")
    for k, v in val_metrics.items():
        if k not in ["onset_details", "sample_meta"]:
            print(f"{k}: {v}")

    test_metrics = run_eval(test_loader, w50_all, cap50["states_lookup"], "Scenario 50 Test (held‑out)")
    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        if k not in ["onset_details", "sample_meta"]:
            print(f"{k}: {v}")
    print(f"Total runtime: {time.perf_counter() - experiment_start:.2f}s")

if __name__ == "__main__":
    main()
