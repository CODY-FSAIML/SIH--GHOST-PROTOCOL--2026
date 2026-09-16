"""
Temporal Forecasting Targets and Chronological Splitting Module
==============================================================
Constructs 60-second non-overlapping temporal windows and multi-step
forecasting targets (t+1, t+2, t+3) from normalized network flow records.

Anti-Leakage Guarantees:
- Input history windows only expose window indexing and chronological boundaries.
- Attack ground truth (is_attack, attack_flow_count, attack_ratio, raw_label)
  is strictly isolated to target and evaluation metadata.
- Chronological splitting with purge/embargo gap buffers guarantees no
  temporal overlap across train, validation, and test partitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalWindow:
    """
    Metadata and target ground truth for a discrete, non-overlapping temporal window.
    """
    window_idx: int
    start_time: float               # POSIX seconds, inclusive
    end_time: float                 # POSIX seconds, exclusive
    total_flows: int
    attack_flows: int
    attack_ratio: float
    y: int                          # Binary window label: 1 if attack_flows >= 1 else 0
    is_empty: bool
    active_attack_actions: Tuple[str, ...] = field(default_factory=tuple)
    flow_indices: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ForecastingSample:
    """
    A single forecasting sample pairing historical window references with future targets.
    
    CRITICAL ANTI-LEAKAGE DESIGN:
    - history_window_indices contains ONLY indices [t-H+1, ..., t] of observed windows.
    - Features derived from history MUST NOT access target_metadata or Y.
    - Y and target_metadata are strictly for training loss and evaluation.
    """
    sample_idx: int
    anchor_window_idx: int                  # Window t
    history_window_indices: Tuple[int, ...] # e.g. (t-2, t-1, t) for H=3
    target_window_indices: Tuple[int, ...]  # e.g. (t+1, t+2, t+3)
    Y: Dict[int, int]                       # {1: y_(t+1), 2: y_(t+2), 3: y_(t+3)}
    target_empty_flags: Dict[int, bool]     # {1: is_empty_(t+1), ...}
    target_metadata: Dict[str, Any]         # Diagnostic target info (NEVER model input)


@dataclass(frozen=True)
class ChronologicalSplit:
    """
    Container for chronologically partitioned forecasting samples and window ranges.
    """
    train_samples: List[ForecastingSample]
    val_samples: List[ForecastingSample]
    test_samples: List[ForecastingSample]
    train_window_range: Tuple[int, int]     # (start_window_idx, end_window_idx inclusive)
    val_window_range: Tuple[int, int]
    test_window_range: Tuple[int, int]
    gap1_window_range: Optional[Tuple[int, int]]
    gap2_window_range: Optional[Tuple[int, int]]


def build_temporal_windows(
    normalized_df: pd.DataFrame,
    window_size_sec: float = 60.0,
) -> List[TemporalWindow]:
    """
    Segment normalized flow records into contiguous, non-overlapping temporal windows.

    Parameters:
        normalized_df: DataFrame conforming to CANONICAL_COLUMNS with 'timestamp' column.
        window_size_sec: Duration of each tumbling window in seconds (default: 60.0).

    Returns:
        Ordered list of TemporalWindow instances spanning from min timestamp to max timestamp.
        Empty windows are explicitly preserved.
    """
    if normalized_df.empty:
        return []

    if "timestamp" not in normalized_df.columns:
        raise ValueError("normalized_df must contain a 'timestamp' column.")

    # Work on a view sorted chronologically
    df_sorted = normalized_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df_sorted.empty:
        return []

    min_ts = float(df_sorted["timestamp"].min())
    max_ts = float(df_sorted["timestamp"].max())

    # Align window start to floor of min timestamp
    t0 = float(np.floor(min_ts))
    total_duration = max_ts - t0
    num_windows = int(np.ceil(total_duration / window_size_sec))
    if num_windows <= 0:
        num_windows = 1

    # Extract required arrays for fast slicing
    ts_array = df_sorted["timestamp"].to_numpy(dtype=float)
    is_attack_array = (
        (df_sorted["is_attack"] == 1).to_numpy(dtype=bool)
        if "is_attack" in df_sorted.columns
        else np.zeros(len(df_sorted), dtype=bool)
    )

    # Extract raw labels / action tokens if present
    raw_label_array = (
        df_sorted["raw_label"].to_numpy(dtype=object)
        if "raw_label" in df_sorted.columns
        else np.array([""] * len(df_sorted), dtype=object)
    )

    original_indices = df_sorted.index.to_numpy()

    # Precompute window bucket assignments using searchsorted for O(N log W) efficiency
    window_edges = np.array([t0 + i * window_size_sec for i in range(num_windows + 1)])
    # searchsorted 'right' - 1 gives 0-indexed bucket for [start, end)
    bucket_indices = np.searchsorted(window_edges, ts_array, side="right") - 1
    # Clip any edge points that hit exactly window_edges[-1]
    bucket_indices = np.clip(bucket_indices, 0, num_windows - 1)

    windows: List[TemporalWindow] = []

    for w_idx in range(num_windows):
        w_start = t0 + w_idx * window_size_sec
        w_end = w_start + window_size_sec

        mask = (bucket_indices == w_idx)
        count = int(np.sum(mask))

        if count == 0:
            windows.append(
                TemporalWindow(
                    window_idx=w_idx,
                    start_time=w_start,
                    end_time=w_end,
                    total_flows=0,
                    attack_flows=0,
                    attack_ratio=0.0,
                    y=0,
                    is_empty=True,
                    active_attack_actions=(),
                    flow_indices=(),
                )
            )
        else:
            w_attack_flags = is_attack_array[mask]
            attack_count = int(np.sum(w_attack_flags))
            attack_ratio = float(attack_count / count)
            y_val = 1 if attack_count >= 1 else 0

            # Gather active attack actions/labels for ground-truth audit
            if attack_count > 0:
                w_labels = raw_label_array[mask]
                attack_labels = pd.Series(w_labels[w_attack_flags]).dropna().unique().tolist()
                actions = tuple(sorted(str(lbl) for lbl in attack_labels))
            else:
                actions = ()

            w_flow_indices = tuple(int(idx) for idx in original_indices[mask])

            windows.append(
                TemporalWindow(
                    window_idx=w_idx,
                    start_time=w_start,
                    end_time=w_end,
                    total_flows=count,
                    attack_flows=attack_count,
                    attack_ratio=attack_ratio,
                    y=y_val,
                    is_empty=False,
                    active_attack_actions=actions,
                    flow_indices=w_flow_indices,
                )
            )

    return windows


def build_forecasting_samples(
    windows: Sequence[TemporalWindow],
    history_len: int = 3,
    horizons: Sequence[int] = (1, 2, 3),
) -> List[ForecastingSample]:
    """
    Construct multi-step forecasting samples from an ordered sequence of TemporalWindows.

    Parameters:
        windows: List of TemporalWindow instances ordered chronologically.
        history_len: Number of historical windows H observed up to anchor time t (default: 3).
        horizons: Forecasting steps ahead [k1, k2, k3] relative to t (default: (1, 2, 3)).

    Returns:
        List of ForecastingSample objects.
        Windows without sufficient past history (t < H - 1) or sufficient future
        horizons (t + max(horizons) >= len(windows)) are excluded.
    """
    if not windows:
        return []

    num_windows = len(windows)
    max_h = max(horizons)
    samples: List[ForecastingSample] = []

    # Valid anchor window indices range from (history_len - 1) to (num_windows - 1 - max_h)
    start_t = history_len - 1
    end_t = num_windows - max_h

    sample_id = 0
    for t in range(start_t, end_t):
        history_indices = tuple(range(t - history_len + 1, t + 1))
        target_indices = tuple(t + k for k in horizons)

        y_dict: Dict[int, int] = {}
        empty_dict: Dict[int, bool] = {}
        target_meta: Dict[str, Any] = {}

        for k in horizons:
            target_w = windows[t + k]
            # Empty target window defaults to 0 attack
            y_dict[k] = 0 if target_w.is_empty else int(target_w.y)
            empty_dict[k] = bool(target_w.is_empty)

            target_meta[f"k{k}_window_idx"] = target_w.window_idx
            target_meta[f"k{k}_start_time"] = target_w.start_time
            target_meta[f"k{k}_total_flows"] = target_w.total_flows
            target_meta[f"k{k}_attack_flows"] = target_w.attack_flows
            target_meta[f"k{k}_attack_ratio"] = target_w.attack_ratio
            target_meta[f"k{k}_active_actions"] = target_w.active_attack_actions
            target_meta[f"k{k}_is_empty"] = target_w.is_empty

        # Anchor window ground truth (diagnostic only)
        anchor_w = windows[t]
        target_meta["anchor_window_idx"] = anchor_w.window_idx
        target_meta["anchor_y"] = anchor_w.y
        target_meta["anchor_attack_flows"] = anchor_w.attack_flows
        target_meta["anchor_is_onset_t1"] = bool(anchor_w.y == 0 and y_dict.get(1, 0) == 1)

        samples.append(
            ForecastingSample(
                sample_idx=sample_id,
                anchor_window_idx=t,
                history_window_indices=history_indices,
                target_window_indices=target_indices,
                Y=y_dict,
                target_empty_flags=empty_dict,
                target_metadata=target_meta,
            )
        )
        sample_id += 1

    return samples


def chronological_split(
    samples: Sequence[ForecastingSample],
    total_windows: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_gap_windows: int = 2,
) -> ChronologicalSplit:
    """
    Split forecasting samples chronologically into Train, Validation, and Test sets.

    Parameters:
        samples: Sequence of ForecastingSample objects.
        total_windows: Total number of temporal windows generated.
        train_ratio: Fraction of window timeline allocated to train (~0.60).
        val_ratio: Fraction allocated to validation (~0.20).
        test_ratio: Fraction allocated to testing (~0.20).
        purge_gap_windows: Practical embargo gap between partitions to mitigate straddle leakage.

    Returns:
        ChronologicalSplit container with segregated samples and window index boundaries.
    """
    if not samples:
        return ChronologicalSplit(
            train_samples=[],
            val_samples=[],
            test_samples=[],
            train_window_range=(0, 0),
            val_window_range=(0, 0),
            test_window_range=(0, 0),
            gap1_window_range=None,
            gap2_window_range=None,
        )

    # 1. Define window boundaries
    n_train = int(np.floor(total_windows * train_ratio))
    train_end = n_train - 1  # inclusive

    gap1_start = n_train
    gap1_end = gap1_start + purge_gap_windows - 1

    val_start = gap1_end + 1
    n_val = int(np.floor(total_windows * val_ratio))
    val_end = val_start + n_val - 1

    gap2_start = val_end + 1
    gap2_end = gap2_start + purge_gap_windows - 1

    test_start = gap2_end + 1
    test_end = total_windows - 1

    # 2. Strict Partition Inclusion Rule:
    # A sample is in partition P iff:
    #   min(history_indices) >= partition.start AND max(target_indices) <= partition.end
    train_samples: List[ForecastingSample] = []
    val_samples: List[ForecastingSample] = []
    test_samples: List[ForecastingSample] = []

    for s in samples:
        min_hist = min(s.history_window_indices)
        max_target = max(s.target_window_indices)

        if min_hist >= 0 and max_target <= train_end:
            train_samples.append(s)
        elif min_hist >= val_start and max_target <= val_end:
            val_samples.append(s)
        elif min_hist >= test_start and max_target <= test_end:
            test_samples.append(s)
        # Else: falls in or crosses a gap buffer, strictly excluded

    return ChronologicalSplit(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_window_range=(0, train_end),
        val_window_range=(val_start, val_end),
        test_window_range=(test_start, test_end),
        gap1_window_range=(gap1_start, gap1_end) if purge_gap_windows > 0 else None,
        gap2_window_range=(gap2_start, gap2_end) if purge_gap_windows > 0 else None,
    )
