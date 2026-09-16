"""
Temporal Dynamics Module (Phase 2A)
===================================
Constructs first-difference (velocity) features:
    ΔS_t = S_t - S_(t-1)
and concatenated State + Velocity representations:
    Z_t = [S_t, ΔS_t] ∈ R^32

Anti-Leakage & Numerical Invariants:
1. Strict Causality: ΔS_t uses only S_t and S_(t-1). No future window (>= t+1) is ever accessed.
2. Boundary Safety: For the first timestep in a sequence (index 0), ΔS is zeroed:
   ΔS_(t-H+1) = 0.
3. Feature Dimensions: S ∈ R^16, ΔS ∈ R^16, Z ∈ R^32.
4. Normalization Isolation: Scalers for S and ΔS are fitted strictly on Training sequence data.
   Test data is transformed using the pre-fitted training statistics with zero leakage.
5. Determinism: Output is strictly deterministic and free of NaN/Inf.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .network_state import MacroNetworkState, MACRO_FEATURE_NAMES
from ..data.temporal_targets import TemporalWindow


def compute_first_difference_sequence(
    X_state: np.ndarray,
) -> np.ndarray:
    """
    Compute first-difference temporal velocity ΔS for each sample sequence.

    Parameters:
        X_state: Array of shape (N, H, 16) representing state sequence [S_(t-H+1), ..., S_t].

    Returns:
        delta_S: Array of shape (N, H, 16) where:
            delta_S[:, 0, :] = 0.0 (first timestep has no prior intra-sequence observation)
            delta_S[:, i, :] = X_state[:, i, :] - X_state[:, i-1, :] for i >= 1.
    """
    if X_state.ndim != 3:
        raise ValueError(f"Expected 3D array of shape (N, H, 16), got ndim={X_state.ndim}")

    N, H, D = X_state.shape
    delta_S = np.zeros_like(X_state, dtype=np.float32)

    if H > 1:
        # Vectorized calculation across all samples and features: ΔS_i = S_i - S_(i-1)
        delta_S[:, 1:, :] = X_state[:, 1:, :] - X_state[:, :-1, :]

    return delta_S


def construct_state_plus_velocity_sequence(
    X_state: np.ndarray,
) -> np.ndarray:
    """
    Concatenate state and velocity along feature dimension:
        Z_t = [S_t, ΔS_t]

    Parameters:
        X_state: Array of shape (N, H, 16)

    Returns:
        Z: Array of shape (N, H, 32)
    """
    delta_S = compute_first_difference_sequence(X_state)
    Z = np.concatenate([X_state, delta_S], axis=-1).astype(np.float32)
    return Z


def build_temporal_dynamics_dataset(
    windows: Sequence[TemporalWindow],
    states_lookup: Dict[int, MacroNetworkState],
    history_len: int = 3,
    include_velocity: bool = True,
    scaler: Optional[Any] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any, List[Dict[str, Any]]]:
    """
    Construct (N, H, D) temporal sequences with or without velocity, with isolated scaling.

    Parameters:
        windows: Chronological sequence of TemporalWindows for ONE capture.
        states_lookup: Mapping of window_idx -> MacroNetworkState.
        history_len: Length of historical context H (e.g. 3 or 10).
        include_velocity: If True, outputs (N, H, 32) [S, ΔS]; if False, outputs (N, H, 16) [S].
        scaler: Pre-fitted StandardScaler or None.
        fit_scaler: If True, fit StandardScaler strictly on this dataset's features.

    Returns:
        X: (N, H, 32) if include_velocity else (N, H, 16).
        y: (N,) binary target array (Y_(t+1)).
        scaler: Fitted or passed-through scaler.
        sample_meta: Diagnostic metadata per sample.
    """
    if len(windows) < history_len + 1:
        feature_dim = 32 if include_velocity else 16
        return (
            np.empty((0, history_len, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            scaler,
            [],
        )

    num_windows = len(windows)
    start_t = history_len - 1
    end_t = num_windows - 1

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    for t in range(start_t, end_t):
        hist_indices = list(range(t - history_len + 1, t + 1))
        target_idx = t + 1

        hist_vectors = []
        for h_idx in hist_indices:
            if h_idx not in states_lookup:
                raise KeyError(f"Window index {h_idx} missing from states_lookup.")
            hist_vectors.append(states_lookup[h_idx].vector)

        target_w = windows[target_idx]
        y_val = 0 if target_w.is_empty else int(target_w.y)

        anchor_w = windows[t]
        is_onset = bool(anchor_w.y == 0 and y_val == 1)

        seq_mat = np.stack(hist_vectors, axis=0).astype(np.float32)  # (H, 16)
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

    X_state = np.array(x_list, dtype=np.float32)  # (N, H, 16)
    y = np.array(y_list, dtype=np.int64)          # (N,)

    # If include_velocity, construct Z = [S, ΔS] BEFORE scaling, or scale S and ΔS
    # Here we construct Z_t = [S_t, ΔS_t]
    if include_velocity:
        X_out = construct_state_plus_velocity_sequence(X_state)  # (N, H, 32)
    else:
        X_out = X_state                                          # (N, H, 16)

    # Isolated Feature Normalization
    if scaler is not None or fit_scaler:
        N, H, D = X_out.shape
        X_flat = X_out.reshape(-1, D)

        if fit_scaler:
            scaler = StandardScaler()
            X_flat = scaler.fit_transform(X_flat)
        else:
            X_flat = scaler.transform(X_flat)

        X_out = X_flat.reshape(N, H, D).astype(np.float32)

    return X_out, y, scaler, meta_list
