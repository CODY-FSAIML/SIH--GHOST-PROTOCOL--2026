"""
Temporal Sequence Dataset for Transformer Forecaster
===================================================
Constructs strictly non-leaking temporal sequences of shape (N, H, 16)
for forecasting future attack occurrences Y_(t+1).

Anti-Leakage & Integrity Invariants:
1. History length H is customizable (e.g. H=10, H=3).
2. Input tensors contain strictly past observable NetworkState vectors:
   [S_(t-H+1), ..., S_t].
3. No future window (>= t+1) ever enters input features.
4. Sequences never cross capture boundaries (Scenario 42 and Scenario 50 are disjoint).
5. Feature scaling (StandardScaler) is strictly fit on Training data only.
6. Chronological ordering is preserved.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.network_state import MacroNetworkState, MACRO_FEATURE_NAMES
from ..data.temporal_targets import TemporalWindow, ForecastingSample


class TemporalSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping historical sequence tensors and binary targets.
    
    Tensor shapes:
        X: torch.FloatTensor of shape (N, H, 16)
        y: torch.FloatTensor of shape (N,)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_meta: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Parameters:
            X: Array of shape (N, H, 16)
            y: Array of shape (N,)
            sample_meta: Optional diagnostic metadata per sample
        """
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples.")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sample_meta = sample_meta or []

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_temporal_sequence_dataset(
    windows: Sequence[TemporalWindow],
    states_lookup: Dict[int, MacroNetworkState],
    history_len: int = 10,
    scaler: Optional[Any] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any, List[Dict[str, Any]]]:
    """
    Construct (N, H, 16) temporal sequence arrays and (N,) targets from a single capture.

    Parameters:
        windows: Chronologically ordered list of TemporalWindow objects for ONE capture.
        states_lookup: Mapping of window_idx -> MacroNetworkState.
        history_len: Number of historical time steps H (e.g. 10 or 3).
        scaler: StandardScaler instance (fitted or to be fitted).
        fit_scaler: If True, fit the scaler on the 16 features across training steps.

    Returns:
        X: (N, H, 16) float32 numpy array.
        y: (N,) int64 numpy array (target at t+1).
        scaler: Fitted or passed-through scaler.
        sample_meta: Diagnostic metadata list for each sample.
    """
    if len(windows) < history_len + 1:
        return (
            np.empty((0, history_len, len(MACRO_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            scaler,
            [],
        )

    num_windows = len(windows)
    # Valid anchor t must have at least (history_len - 1) past windows
    # and at least 1 future window (t + 1 < num_windows)
    start_t = history_len - 1
    end_t = num_windows - 1  # t+1 <= num_windows - 1

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    for t in range(start_t, end_t):
        # History indices: [t - history_len + 1, ..., t]
        hist_indices = list(range(t - history_len + 1, t + 1))
        target_idx = t + 1

        # Check all history states exist
        hist_vectors = []
        for h_idx in hist_indices:
            if h_idx not in states_lookup:
                raise KeyError(f"Window index {h_idx} missing from states_lookup.")
            hist_vectors.append(states_lookup[h_idx].vector)

        # Target window ground truth
        target_w = windows[target_idx]
        y_val = 0 if target_w.is_empty else int(target_w.y)

        # Anchor window ground truth (for diagnostic onset evaluation)
        anchor_w = windows[t]
        is_onset = bool(anchor_w.y == 0 and y_val == 1)

        seq_mat = np.stack(hist_vectors, axis=0)  # Shape: (H, 16)
        x_list.append(seq_mat)
        y_list.append(y_val)

        meta_list.append({
            "sample_idx": len(meta_list),
            "anchor_window_idx": t,
            "history_window_indices": tuple(hist_indices),
            "target_window_idx": target_idx,
            "anchor_y": anchor_w.y,
            "is_onset_t1": is_onset,
            "anchor_attack_flows": anchor_w.attack_flows,
            "target_attack_flows": target_w.attack_flows,
            "target_active_actions": target_w.active_attack_actions,
        })

    X = np.array(x_list, dtype=np.float32)  # (N, H, 16)
    y = np.array(y_list, dtype=np.int64)    # (N,)

    # Feature normalization across the 16 features
    if scaler is not None or fit_scaler:
        N, H, D = X.shape
        X_reshaped = X.reshape(-1, D)  # (N * H, 16)

        if fit_scaler:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_reshaped = scaler.fit_transform(X_reshaped)
        else:
            X_reshaped = scaler.transform(X_reshaped)

        X = X_reshaped.reshape(N, H, D).astype(np.float32)

    return X, y, scaler, meta_list


def compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Calculate pos_weight for BCEWithLogitsLoss strictly from training labels:
        pos_weight = num_negative / num_positive
    """
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))

    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def chronological_split(windows, train_ratio=0.6, val_ratio=0.2, purge_gap=0):
    """Chronologically split windows into train/val/test.

    Parameters:
        windows: list of windows in chronological order.
        train_ratio: fraction for training.
        val_ratio: fraction of remaining for validation.
        purge_gap: number of windows to purge between splits to avoid leakage.
    Returns:
        train_windows, val_windows, test_windows
    """
    total = len(windows)
    if total == 0:
        return [], [], []
    train_end = int(total * train_ratio)
    remaining = total - train_end
    val_end = train_end + int(remaining * val_ratio)
    if purge_gap > 0:
        train = windows[:max(0, train_end - purge_gap)]
        val_start = train_end + purge_gap
        val = windows[val_start:max(0, val_end - purge_gap)]
        test_start = val_end + purge_gap
        test = windows[max(0, test_start):]
    else:
        train = windows[:train_end]
        val = windows[train_end:val_end]
        test = windows[val_end:]
    return train, val, test
