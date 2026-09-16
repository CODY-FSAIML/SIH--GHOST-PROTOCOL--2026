"""
Full Training and Benchmark Runner for Temporal Transformer Forecaster (Phase 1)
================================================================================
Executes:
1. Experiment 1: Train Scenario 42 -> Test Scenario 50 (H=10)
2. Experiment 2: Train Scenario 50 -> Test Scenario 42 (H=10)
3. Experiment 4: Ablation H=3 vs H=10 on 42 -> 50
4. Detailed Onset Analysis (8 binary onsets in Sc.50, 2 binary onsets in Sc.42)
5. Comparison against Logistic Regression H=3
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, r"c:\Users\sridevi\OneDrive\Desktop\sih")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.adapters.ctu13_adapter import _state_category
from src.features.network_state import MacroNetworkStateBuilder, MacroNetworkState
from src.data.temporal_targets import build_temporal_windows, chronological_split
from src.models.temporal_dataset import (
    TemporalSequenceDataset,
    build_temporal_sequence_dataset,
    compute_pos_weight,
)
from src.models.temporal_transformer import TemporalTransformerForecaster

SCENARIO_42_PATH = r"c:\Users\sridevi\OneDrive\Desktop\sih\data\raw\ctu13\capture20110810.binetflow.txt"
SCENARIO_50_PATH = r"c:\Users\sridevi\OneDrive\Desktop\sih\data\raw\ctu13\capture20110817.binetflow"


def load_raw_capture(filepath: str, name: str):
    print(f"Loading {name} from {filepath}...")
    t0 = time.time()
    cols = ["StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "TotPkts", "TotBytes", "SrcBytes", "State", "Label"]
    df_raw = pd.read_csv(filepath, usecols=cols, low_memory=False)
    print(f"  Loaded {len(df_raw):,} flows in {time.time() - t0:.2f}s")

    t_norm = time.time()
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
    print(f"  Normalized in {time.time() - t_norm:.2f}s")

    builder = MacroNetworkStateBuilder(window_size_sec=60.0)
    states, _, _ = builder.build_states(norm_df)
    states_lookup = {s.window_idx: s for s in states}

    windows = build_temporal_windows(norm_df, window_size_sec=60.0)
    return {
        "name": name,
        "norm_df": norm_df,
        "states_lookup": states_lookup,
        "windows": windows,
    }


def split_capture_windows(
    windows: List[Any],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    purge_gap: int = 2,
):
    """
    Partition windows chronologically into Train, Val, Test window slices.
    """
    N = len(windows)
    n_train = int(np.floor(N * train_ratio))
    train_windows = windows[:n_train]

    gap1_end = n_train + purge_gap
    n_val = int(np.floor(N * val_ratio))
    val_windows = windows[gap1_end : gap1_end + n_val]

    gap2_end = gap1_end + n_val + purge_gap
    test_windows = windows[gap2_end:]

    return train_windows, val_windows, test_windows


def train_transformer_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    history_len: int = 10,
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

    pos_weight_val = compute_pos_weight(y_train)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    model = TemporalTransformerForecaster(
        input_dim=16,
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
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))
        history["train_loss"].append(mean_train_loss)

        # Validation loss
        if val_ds and len(val_ds) > 0:
            model.eval()
            with torch.no_grad():
                x_val_t = val_ds.X
                y_val_t = val_ds.y
                val_logits = model(x_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                    break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_meta = {
        "epochs_trained": epoch,
        "best_val_loss": best_val_loss,
        "pos_weight": pos_weight_val,
        "history": history,
    }
    return model, train_meta


def evaluate_transformer(
    model: TemporalTransformerForecaster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_meta: List[Dict[str, Any]],
    description: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(X_test, dtype=torch.float32)
        probs = model.predict_proba(x_tensor).cpu().numpy()

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

    # Onsets: anchor_y == 0 and target Y_t+1 == 1
    onset_indices = [i for i, m in enumerate(sample_meta) if m.get("is_onset_t1", False)]
    total_onsets = len(onset_indices)
    onsets_detected = sum(preds[i] == 1 for i in onset_indices)
    onsets_missed = total_onsets - onsets_detected
    onset_recall = float(onsets_detected / total_onsets) if total_onsets > 0 else 0.0

    benign_indices = [i for i, m in enumerate(sample_meta) if (not m.get("is_onset_t1", False)) and y_test[i] == 0]
    false_alarms_benign = sum(preds[i] == 1 for i in benign_indices)

    onset_details = []
    for idx in onset_indices:
        m = sample_meta[idx]
        w_anchor = m["anchor_window_idx"]
        w_target = m["target_window_idx"]
        onset_details.append({
            "anchor_window": w_anchor,
            "target_window": w_target,
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
        "false_alarms_benign": false_alarms_benign,
        "onset_details": onset_details,
        "probs_summary": {
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
            "mean": float(np.mean(probs)),
            "median": float(np.median(probs)),
            "q25": float(np.quantile(probs, 0.25)),
            "q75": float(np.quantile(probs, 0.75)),
        },
    }


def print_report(res: Dict[str, Any]):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {res['description']}")
    print(f"{'='*65}")
    print(f"Samples: {res['sample_count']} (Pos: {res['pos_count']}, Neg: {res['neg_count']})")
    print(f"Predicted: Pos={res['pred_pos']}, Neg={res['pred_neg']} (Threshold: {res['threshold']:.3f})")
    print(f"Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1']:.4f} | FPR: {res['fpr']:.4f}")
    cm = res['confusion_matrix']
    print(f"Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    roc_str = f"{res['roc_auc']:.4f}" if res['roc_auc'] is not None else "N/A"
    pr_str = f"{res['pr_auc']:.4f}" if res['pr_auc'] is not None else "N/A"
    print(f"ROC-AUC: {roc_str} | PR-AUC: {pr_str}")
    ps = res['probs_summary']
    print(f"Probabilities: Min={ps['min']:.4f}, Q25={ps['q25']:.4f}, Median={ps['median']:.4f}, Mean={ps['mean']:.4f}, Q75={ps['q75']:.4f}, Max={ps['max']:.4f}")
    print(f"ONSET METRICS:")
    print(f"  Total Binary Onsets: {res['total_onsets']}")
    print(f"  Onsets Detected:     {res['onsets_detected']} ({res['onset_recall']*100:.1f}%)")
    print(f"  Onsets Missed:       {res['onsets_missed']}")
    print(f"  False Alarms in Benign States: {res['false_alarms_benign']}")
    print("  Per-Onset Predictions:")
    for d in res['onset_details']:
        status = "DETECTED [OK]" if d['detected'] else "MISSED [X]"
        print(f"    W{d['anchor_window']} -> W{d['target_window']} | P(Attack)={d['prob']:.4f} | {status}")


def run_all_experiments():
    # 1. Load raw captures
    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")

    # Split windows chronologically
    w42_tr, w42_val, w42_test = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)
    w50_tr, w50_val, w50_test = split_capture_windows(cap50["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)

    # =====================================================================
    # EXPERIMENT 1: TRAIN SCENARIO 42 (H=10) -> TEST SCENARIO 50 (FULL TIMELINE)
    # =====================================================================
    print("\n" + "#"*70)
    print("EXPERIMENT 1: TRAIN SCENARIO 42 (H=10) -> TEST SCENARIO 50 (H=10)")
    print("#"*70)

    # 1. Fit scaler strictly on Scenario 42 Train
    X42_tr_h10, y42_tr_h10, scaler_42_h10, meta42_tr_h10 = build_temporal_sequence_dataset(
        w42_tr, cap42["states_lookup"], history_len=10, fit_scaler=True
    )
    X42_val_h10, y42_val_h10, _, meta42_val_h10 = build_temporal_sequence_dataset(
        w42_val, cap42["states_lookup"], history_len=10, scaler=scaler_42_h10
    )
    # Transform Scenario 50 Full timeline with Scenario 42's scaler
    X50_all_h10, y50_all_h10, _, meta50_all_h10 = build_temporal_sequence_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=10, scaler=scaler_42_h10
    )

    print(f"Sc.42 Train shape: {X42_tr_h10.shape}, Pos: {np.sum(y42_tr_h10==1)}, Neg: {np.sum(y42_tr_h10==0)}")
    print(f"Sc.42 Val shape:   {X42_val_h10.shape}, Pos: {np.sum(y42_val_h10==1)}, Neg: {np.sum(y42_val_h10==0)}")
    print(f"Sc.50 Test shape:  {X50_all_h10.shape}, Pos: {np.sum(y50_all_h10==1)}, Neg: {np.sum(y50_all_h10==0)}")

    model_42_h10, meta_train_42 = train_transformer_model(
        X42_tr_h10, y42_tr_h10, X42_val_h10, y42_val_h10, history_len=10
    )
    res_1 = evaluate_transformer(
        model_42_h10, X50_all_h10, y50_all_h10, meta50_all_h10, "Train Sc.42 (H=10) -> Test Sc.50 Full"
    )
    print_report(res_1)

    # =====================================================================
    # EXPERIMENT 2: TRAIN SCENARIO 50 (H=10) -> TEST SCENARIO 42 (FULL TIMELINE)
    # =====================================================================
    print("\n" + "#"*70)
    print("EXPERIMENT 2: TRAIN SCENARIO 50 (H=10) -> TEST SCENARIO 42 (H=10)")
    print("#"*70)

    # Fit scaler strictly on Scenario 50 Train
    X50_tr_h10, y50_tr_h10, scaler_50_h10, meta50_tr_h10 = build_temporal_sequence_dataset(
        w50_tr, cap50["states_lookup"], history_len=10, fit_scaler=True
    )
    X50_val_h10, y50_val_h10, _, meta50_val_h10 = build_temporal_sequence_dataset(
        w50_val, cap50["states_lookup"], history_len=10, scaler=scaler_50_h10
    )
    # Transform Scenario 42 Full timeline with Scenario 50's scaler
    X42_all_h10, y42_all_h10, _, meta42_all_h10 = build_temporal_sequence_dataset(
        cap42["windows"], cap42["states_lookup"], history_len=10, scaler=scaler_50_h10
    )

    print(f"Sc.50 Train shape: {X50_tr_h10.shape}, Pos: {np.sum(y50_tr_h10==1)}, Neg: {np.sum(y50_tr_h10==0)}")
    print(f"Sc.50 Val shape:   {X50_val_h10.shape}, Pos: {np.sum(y50_val_h10==1)}, Neg: {np.sum(y50_val_h10==0)}")
    print(f"Sc.42 Test shape:  {X42_all_h10.shape}, Pos: {np.sum(y42_all_h10==1)}, Neg: {np.sum(y42_all_h10==0)}")

    model_50_h10, meta_train_50 = train_transformer_model(
        X50_tr_h10, y50_tr_h10, X50_val_h10, y50_val_h10, history_len=10
    )
    res_2 = evaluate_transformer(
        model_50_h10, X42_all_h10, y42_all_h10, meta42_all_h10, "Train Sc.50 (H=10) -> Test Sc.42 Full"
    )
    print_report(res_2)

    # =====================================================================
    # EXPERIMENT 3 (ABLATION): TRANSFORMER H=3 vs H=10 (Train 42 -> Test 50)
    # =====================================================================
    print("\n" + "#"*70)
    print("EXPERIMENT 3 (ABLATION): TRANSFORMER H=3 vs H=10 (Train Sc.42 -> Test Sc.50)")
    print("#"*70)

    X42_tr_h3, y42_tr_h3, scaler_42_h3, _ = build_temporal_sequence_dataset(
        w42_tr, cap42["states_lookup"], history_len=3, fit_scaler=True
    )
    X42_val_h3, y42_val_h3, _, _ = build_temporal_sequence_dataset(
        w42_val, cap42["states_lookup"], history_len=3, scaler=scaler_42_h3
    )
    X50_all_h3, y50_all_h3, _, meta50_all_h3 = build_temporal_sequence_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=3, scaler=scaler_42_h3
    )

    model_42_h3, _ = train_transformer_model(
        X42_tr_h3, y42_tr_h3, X42_val_h3, y42_val_h3, history_len=3
    )
    res_3_h3 = evaluate_transformer(
        model_42_h3, X50_all_h3, y50_all_h3, meta50_all_h3, "Train Sc.42 (H=3) -> Test Sc.50 Full"
    )
    print_report(res_3_h3)

    return {
        "exp1": res_1,
        "exp2": res_2,
        "exp_h3": res_3_h3,
    }


if __name__ == "__main__":
    run_all_experiments()
