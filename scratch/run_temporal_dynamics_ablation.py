"""
Temporal Dynamics Ablation Experiment Runner (Phase 2A)
======================================================
Compares:
1. Primary Experiment (H=3):
   - Transformer H=3 STATE ONLY (16 features)
   vs
   - Transformer H=3 STATE + VELOCITY (32 features)
   Train: Scenario 42 Train
   Validate: Scenario 42 Val
   Test: Scenario 50 Full Timeline

2. Secondary Experiment (H=10):
   - Transformer H=10 STATE ONLY (16 features)
   vs
   - Transformer H=10 STATE + VELOCITY (32 features)
   Train: Scenario 42 Train
   Validate: Scenario 42 Val
   Test: Scenario 50 Full Timeline

Fairness Invariants:
- Identical random seed (42)
- Identical optimizer (AdamW, lr=1e-3, weight_decay=1e-4)
- Identical batch size (32), epochs (30), patience (7)
- Identical class weighting (pos_weight calculated on train)
- Identical threshold (0.500)
- Identical train/test samples
- Strict preprocessing isolation (scaler fit ONLY on train)
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, r"c:\Users\sridevi\OneDrive\Desktop\sih")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.features.network_state import MacroNetworkStateBuilder
from src.data.adapters.ctu13_adapter import _state_category
from src.data.temporal_targets import build_temporal_windows
from src.models.temporal_dataset import TemporalSequenceDataset, compute_pos_weight
from src.models.temporal_transformer import TemporalTransformerForecaster
from src.features.temporal_dynamics import build_temporal_dynamics_dataset

SCENARIO_42_PATH = r"c:\Users\sridevi\OneDrive\Desktop\sih\data\raw\ctu13\capture20110810.binetflow.txt"
SCENARIO_50_PATH = r"c:\Users\sridevi\OneDrive\Desktop\sih\data\raw\ctu13\capture20110817.binetflow"


def load_raw_capture(filepath: str, name: str):
    print(f"Loading {name} from {filepath}...")
    t0 = time.time()
    cols = ["StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "TotPkts", "TotBytes", "SrcBytes", "State", "Label"]
    df_raw = pd.read_csv(filepath, usecols=cols, low_memory=False)

    dt = pd.to_datetime(df_raw["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
    ts = dt.astype("datetime64[ns]").astype(np.int64) / 1e9
    labels = df_raw["Label"].astype(str)
    is_attack = labels.str.contains("Botnet", case=True, regex=False).astype(int)
    state_cats = [_state_category(str(s)) for s in df_raw["State"]]

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

    builder = MacroNetworkStateBuilder(window_size_sec=60.0)
    states, _, _ = builder.build_states(norm_df)
    states_lookup = {s.window_idx: s for s in states}
    windows = build_temporal_windows(norm_df, window_size_sec=60.0)

    print(f"  {name} ready: {len(df_raw):,} flows, {len(windows)} windows in {time.time()-t0:.2f}s")
    return {
        "name": name,
        "norm_df": norm_df,
        "states_lookup": states_lookup,
        "windows": windows,
    }


def split_capture_windows(windows: List[Any], train_ratio=0.6, val_ratio=0.2, purge_gap=2):
    N = len(windows)
    n_train = int(np.floor(N * train_ratio))
    train_windows = windows[:n_train]
    gap1_end = n_train + purge_gap
    n_val = int(np.floor(N * val_ratio))
    val_windows = windows[gap1_end : gap1_end + n_val]
    gap2_end = gap1_end + n_val + purge_gap
    test_windows = windows[gap2_end:]
    return train_windows, val_windows, test_windows


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
) -> Tuple[TemporalTransformerForecaster, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TemporalSequenceDataset(X_train, y_train)
    val_ds = TemporalSequenceDataset(X_val, y_val) if len(X_val) > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    pos_weight = compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    model = TemporalTransformerForecaster(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Val check
        if val_ds and len(val_ds) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_ds.X)
                val_loss = criterion(val_logits, val_ds.y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"epochs": epoch, "best_val_loss": best_val_loss, "pos_weight": pos_weight}


def evaluate(
    model: TemporalTransformerForecaster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_meta: List[Dict[str, Any]],
    description: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(X_test, dtype=torch.float32)
        probs = model.predict_proba(x_t).cpu().numpy()

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    has_both = len(np.unique(y_test)) >= 2
    roc_auc = float(roc_auc_score(y_test, probs)) if has_both else None
    pr_auc = float(average_precision_score(y_test, probs)) if has_both else None

    onset_indices = [i for i, m in enumerate(sample_meta) if m.get("is_onset_t1", False)]
    total_onsets = len(onset_indices)
    onsets_detected = sum(preds[i] == 1 for i in onset_indices)
    onsets_missed = total_onsets - onsets_detected
    onset_recall = float(onsets_detected / total_onsets) if total_onsets > 0 else 0.0

    benign_indices = [i for i, m in enumerate(sample_meta) if (not m.get("is_onset_t1", False)) and y_test[i] == 0]
    false_alarms = sum(preds[i] == 1 for i in benign_indices)

    onset_details = []
    for idx in onset_indices:
        m = sample_meta[idx]
        onset_details.append({
            "anchor": m["anchor_window_idx"],
            "target": m["target_window_idx"],
            "prob": float(probs[idx]),
            "predicted": int(preds[idx]),
            "detected": bool(preds[idx] == 1),
        })

    return {
        "description": description,
        "sample_count": len(y_test),
        "pos_count": int(np.sum(y_test == 1)),
        "neg_count": int(np.sum(y_test == 0)),
        "pred_pos": int(np.sum(preds == 1)),
        "pred_neg": int(np.sum(preds == 0)),
        "threshold": threshold,
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
        "false_alarms": false_alarms,
        "onset_details": onset_details,
    }


def print_comparison_row(r1, r2, name1, name2):
    print(f"\n{'='*75}")
    print(f"ABLATION COMPARISON: {name1} vs {name2}")
    print(f"{'='*75}")
    print(f"{'Metric':25s} | {name1:20s} | {name2:20s} | {'Delta':10s}")
    print(f"{'-'*75}")
    metrics = [
        ("Precision", "precision", False),
        ("Recall", "recall", False),
        ("F1-Score", "f1", False),
        ("FPR (lower better)", "fpr", True),
        ("ROC-AUC", "roc_auc", False),
        ("PR-AUC", "pr_auc", False),
        ("Onset Recall", "onset_recall", False),
        ("False Alarms (lower)", "false_alarms", True),
    ]
    for label, key, lower_better in metrics:
        v1 = r1[key]
        v2 = r2[key]
        if isinstance(v1, float):
            diff = v2 - v1
            print(f"{label:25s} | {v1:20.4f} | {v2:20.4f} | {diff:+10.4f}")
        else:
            diff = v2 - v1
            print(f"{label:25s} | {v1:<20} | {v2:<20} | {diff:+10}")

    print("\nConfusion Matrices:")
    cm1 = r1['confusion_matrix']
    cm2 = r2['confusion_matrix']
    print(f"  {name1}: TN={cm1['tn']}, FP={cm1['fp']}, FN={cm1['fn']}, TP={cm1['tp']}")
    print(f"  {name2}: TN={cm2['tn']}, FP={cm2['fp']}, FN={cm2['fn']}, TP={cm2['tp']}")

    print("\nOnset Detection Breakdown:")
    for d1, d2 in zip(r1['onset_details'], r2['onset_details']):
        s1 = "DETECTED" if d1['detected'] else "MISSED"
        s2 = "DETECTED" if d2['detected'] else "MISSED"
        print(f"  W{d1['anchor']}->W{d1['target']}: {name1}={s1} (P={d1['prob']:.4f}) | {name2}={s2} (P={d2['prob']:.4f})")


def run_ablation():
    # 1. Load data
    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")

    w42_tr, w42_val, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)

    # =====================================================================
    # PRIMARY EXPERIMENT: H=3 (STATE ONLY vs STATE + VELOCITY)
    # =====================================================================
    print("\n" + "#"*70)
    print("PRIMARY EXPERIMENT: H=3 ABLATION (STATE ONLY vs STATE + VELOCITY)")
    print("#"*70)

    # 1.1 State Only H=3
    X42_tr_s3, y42_tr_s3, scaler_s3, _ = build_temporal_dynamics_dataset(
        w42_tr, cap42["states_lookup"], history_len=3, include_velocity=False, fit_scaler=True
    )
    X42_val_s3, y42_val_s3, _, _ = build_temporal_dynamics_dataset(
        w42_val, cap42["states_lookup"], history_len=3, include_velocity=False, scaler=scaler_s3
    )
    X50_all_s3, y50_all_s3, _, meta50_s3 = build_temporal_dynamics_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=3, include_velocity=False, scaler=scaler_s3
    )

    model_s3, _ = train_model(X42_tr_s3, y42_tr_s3, X42_val_s3, y42_val_s3, input_dim=16)
    res_s3 = evaluate(model_s3, X50_all_s3, y50_all_s3, meta50_s3, "Transformer H=3 STATE ONLY")

    # 1.2 State + Velocity H=3
    X42_tr_v3, y42_tr_v3, scaler_v3, _ = build_temporal_dynamics_dataset(
        w42_tr, cap42["states_lookup"], history_len=3, include_velocity=True, fit_scaler=True
    )
    X42_val_v3, y42_val_v3, _, _ = build_temporal_dynamics_dataset(
        w42_val, cap42["states_lookup"], history_len=3, include_velocity=True, scaler=scaler_v3
    )
    X50_all_v3, y50_all_v3, _, meta50_v3 = build_temporal_dynamics_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=3, include_velocity=True, scaler=scaler_v3
    )

    model_v3, _ = train_model(X42_tr_v3, y42_tr_v3, X42_val_v3, y42_val_v3, input_dim=32)
    res_v3 = evaluate(model_v3, X50_all_v3, y50_all_v3, meta50_v3, "Transformer H=3 STATE + VELOCITY")

    print_comparison_row(res_s3, res_v3, "State Only H=3", "State+Velocity H=3")

    # =====================================================================
    # SECONDARY EXPERIMENT: H=10 (STATE ONLY vs STATE + VELOCITY)
    # =====================================================================
    print("\n" + "#"*70)
    print("SECONDARY EXPERIMENT: H=10 ABLATION (STATE ONLY vs STATE + VELOCITY)")
    print("#"*70)

    # 2.1 State Only H=10
    X42_tr_s10, y42_tr_s10, scaler_s10, _ = build_temporal_dynamics_dataset(
        w42_tr, cap42["states_lookup"], history_len=10, include_velocity=False, fit_scaler=True
    )
    X42_val_s10, y42_val_s10, _, _ = build_temporal_dynamics_dataset(
        w42_val, cap42["states_lookup"], history_len=10, include_velocity=False, scaler=scaler_s10
    )
    X50_all_s10, y50_all_s10, _, meta50_s10 = build_temporal_dynamics_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=False, scaler=scaler_s10
    )

    model_s10, _ = train_model(X42_tr_s10, y42_tr_s10, X42_val_s10, y42_val_s10, input_dim=16)
    res_s10 = evaluate(model_s10, X50_all_s10, y50_all_s10, meta50_s10, "Transformer H=10 STATE ONLY")

    # 2.2 State + Velocity H=10
    X42_tr_v10, y42_tr_v10, scaler_v10, _ = build_temporal_dynamics_dataset(
        w42_tr, cap42["states_lookup"], history_len=10, include_velocity=True, fit_scaler=True
    )
    X42_val_v10, y42_val_v10, _, _ = build_temporal_dynamics_dataset(
        w42_val, cap42["states_lookup"], history_len=10, include_velocity=True, scaler=scaler_v10
    )
    X50_all_v10, y50_all_v10, _, meta50_v10 = build_temporal_dynamics_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=10, include_velocity=True, scaler=scaler_v10
    )

    model_v10, _ = train_model(X42_tr_v10, y42_tr_v10, X42_val_v10, y42_val_v10, input_dim=32)
    res_v10 = evaluate(model_v10, X50_all_v10, y50_all_v10, meta50_v10, "Transformer H=10 STATE + VELOCITY")

    print_comparison_row(res_s10, res_v10, "State Only H=10", "State+Velocity H=10")

    return {
        "h3_state": res_s3,
        "h3_velo": res_v3,
        "h10_state": res_s10,
        "h10_velo": res_v10,
    }


if __name__ == "__main__":
    run_ablation()
