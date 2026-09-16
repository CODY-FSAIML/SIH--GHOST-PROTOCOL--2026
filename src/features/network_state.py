"""
NetworkState V1 — Macro State Vector Module
===========================================
Extracts a 16-dimensional observable Macro State Vector for each 60-second
traffic window without accessing any attack label information.

Mathematical & Semantic Guarantees:
- Strict Observable-Only Invariant: No ground truth (is_attack, label, raw_label,
  attack_flow_count, attack_ratio) is ever accessed or exposed.
- Numerical Safety: All ratios, entropies, and stats are guaranteed finite,
  non-negative, and free of NaN/Inf.
- CTU-13 Semantic Preservation: TotPkts and TotBytes are bidirectional totals.
  dst_bytes is derived as max(0, total_bytes - src_bytes).
- Zero-State Guarantee: Empty windows produce cleanly zeroed vectors without NaNs.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Sequence, Any
import numpy as np
import pandas as pd


MACRO_FEATURE_NAMES: Tuple[str, ...] = (
    "flow_count",
    "total_packets",
    "total_bytes",
    "src_bytes_sum",
    "dst_bytes_sum",
    "byte_asymmetry_ratio",
    "mean_duration",
    "max_duration",
    "unique_src_ips",
    "unique_dst_ips",
    "src_ip_entropy",
    "dst_ip_entropy",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "closed_flow_ratio",
)


@dataclass(frozen=True)
class MacroNetworkState:
    """
    Observable Network State S_t for a single temporal window.
    
    Contains window boundary metadata plus the 16-dimensional observable vector.
    NEVER contains ground-truth attack information.
    """
    window_idx: int
    start_time: float               # POSIX seconds, inclusive
    end_time: float                 # POSIX seconds, exclusive
    vector: np.ndarray              # Shape: (16,), dtype: float64
    features: Dict[str, float]      # Feature name -> value mapping

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary with metadata and named features."""
        d = {
            "window_idx": self.window_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        d.update(self.features)
        return d


def _compute_shannon_entropy(items: Sequence[Any]) -> float:
    """
    Compute Shannon entropy H = -sum(p_i * ln(p_i)) in nats.
    
    Numerically safe:
    - Empty sequence or single unique item returns 0.0.
    - Missing/NaN values are filtered out.
    - Output is strictly non-negative and finite.
    """
    valid_items = [str(x) for x in items if x is not None and str(x).lower() != "nan" and str(x).strip() != ""]
    if len(valid_items) <= 1:
        return 0.0

    _, counts = np.unique(valid_items, return_counts=True)
    if len(counts) <= 1:
        return 0.0

    probs = counts / float(counts.sum())
    # probs > 0 is guaranteed by np.unique
    entropy = -np.sum(probs * np.log(probs))
    return float(max(0.0, entropy))


def _extract_numeric_array(df: pd.DataFrame, col_name: str, meta_key: str, default: float = 0.0) -> np.ndarray:
    """
    Extract a numeric column either directly from df[col_name] or from df['metadata'][meta_key].
    """
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default).to_numpy(dtype=float)
    if "metadata" in df.columns:
        meta_col = df["metadata"]
        if len(df) == 0:
            return np.array([], dtype=float)
        vals = [
            m.get(meta_key, default) if isinstance(m, dict) else default
            for m in meta_col
        ]
        return pd.to_numeric(pd.Series(vals), errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def _extract_state_category_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract state_category string array either from df['state_category'] or df['metadata']['state_category'].
    """
    if "state_category" in df.columns:
        return df["state_category"].astype(str).str.lower().str.strip().to_numpy()
    if "metadata" in df.columns:
        meta_col = df["metadata"]
        if len(df) == 0:
            return np.array([], dtype=object)
        vals = [
            str(m.get("state_category", "")).lower().strip() if isinstance(m, dict) else ""
            for m in meta_col
        ]
        return np.array(vals, dtype=object)
    return np.array([""] * len(df), dtype=object)


def extract_macro_state_from_flows(
    flow_df: pd.DataFrame,
    window_idx: int,
    start_time: float,
    end_time: float,
    epsilon: float = 1e-6,
    precomputed: Optional[Dict[str, np.ndarray]] = None,
) -> MacroNetworkState:
    """
    Extract the 16-dimensional Macro State Vector from a slice of flow records
    belonging to window [start_time, end_time).

    STRICT ANTI-LEAKAGE INVARIANT:
    No attack labels or targets are read. The function produces identical output
    whether is_attack/raw_label columns exist or not.
    """
    n_flows = len(flow_df) if precomputed is None else len(precomputed["durations"])

    if n_flows == 0:
        # Zero-State window guarantee
        zero_vec = np.zeros(len(MACRO_FEATURE_NAMES), dtype=np.float64)
        feat_dict = {name: 0.0 for name in MACRO_FEATURE_NAMES}
        return MacroNetworkState(
            window_idx=window_idx,
            start_time=start_time,
            end_time=end_time,
            vector=zero_vec,
            features=feat_dict,
        )

    # 1. flow_count
    flow_count = float(n_flows)

    if precomputed is not None:
        tot_pkts_arr = precomputed["tot_pkts"]
        tot_bytes_arr = precomputed["tot_bytes"]
        src_bytes_arr = precomputed["src_bytes"]
        dst_bytes_arr = precomputed["dst_bytes"]
        durations = precomputed["durations"]
        src_ips = precomputed["src_ips"]
        dst_ips = precomputed["dst_ips"]
        protos = precomputed["protos"]
        state_cats = precomputed["state_cats"]
    else:
        tot_pkts_arr = _extract_numeric_array(flow_df, "total_packets", "total_packets", default=0.0)
        tot_bytes_arr = _extract_numeric_array(flow_df, "total_bytes", "total_bytes", default=0.0)
        src_bytes_arr = _extract_numeric_array(flow_df, "src_bytes", "src_bytes", default=0.0)
        if "dst_bytes" in flow_df.columns or ("metadata" in flow_df.columns and len(flow_df) > 0 and isinstance(flow_df["metadata"].iloc[0], dict) and "dst_bytes" in flow_df["metadata"].iloc[0]):
            dst_bytes_arr = _extract_numeric_array(flow_df, "dst_bytes", "dst_bytes", default=0.0)
        else:
            dst_bytes_arr = np.maximum(0.0, tot_bytes_arr - src_bytes_arr)

        if "duration" in flow_df.columns:
            durations = pd.to_numeric(flow_df["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            durations = np.maximum(0.0, durations)
        else:
            durations = np.zeros(n_flows, dtype=float)

        src_ips = flow_df["src_ip"].dropna().to_numpy() if "src_ip" in flow_df.columns else np.array([])
        dst_ips = flow_df["dst_ip"].dropna().to_numpy() if "dst_ip" in flow_df.columns else np.array([])
        protos = flow_df["protocol"].astype(str).str.lower().str.strip().to_numpy() if "protocol" in flow_df.columns else np.array([""] * n_flows)
        state_cats = _extract_state_category_array(flow_df)

    # 2. total_packets (bidirectional total)
    total_packets = float(np.sum(np.maximum(0.0, tot_pkts_arr)))

    # 3. total_bytes (bidirectional total)
    total_bytes = float(np.sum(np.maximum(0.0, tot_bytes_arr)))

    # 4. src_bytes_sum
    src_bytes_sum = float(np.sum(np.maximum(0.0, src_bytes_arr)))

    # 5. dst_bytes_sum (CTU-13 semantic: dst_bytes = total_bytes - src_bytes)
    dst_bytes_sum = float(np.sum(np.maximum(0.0, dst_bytes_arr)))

    # 6. byte_asymmetry_ratio
    byte_sum = src_bytes_sum + dst_bytes_sum
    if byte_sum <= 0.0:
        byte_asymmetry_ratio = 0.0
    else:
        byte_asymmetry_ratio = float(abs(src_bytes_sum - dst_bytes_sum) / (byte_sum + epsilon))
    byte_asymmetry_ratio = float(np.clip(byte_asymmetry_ratio, 0.0, 1.0))

    # 7. mean_duration & 8. max_duration
    mean_duration = float(np.mean(durations)) if len(durations) > 0 else 0.0
    max_duration = float(np.max(durations)) if len(durations) > 0 else 0.0

    # 9. unique_src_ips & 10. unique_dst_ips
    unique_src_set = {str(x) for x in src_ips if str(x).lower() != "nan" and str(x).strip() != ""}
    unique_dst_set = {str(x) for x in dst_ips if str(x).lower() != "nan" and str(x).strip() != ""}

    unique_src_ips = float(len(unique_src_set))
    unique_dst_ips = float(len(unique_dst_set))

    # 11. src_ip_entropy & 12. dst_ip_entropy
    src_ip_entropy = _compute_shannon_entropy(src_ips)
    dst_ip_entropy = _compute_shannon_entropy(dst_ips)

    # 13. tcp_ratio, 14. udp_ratio, 15. icmp_ratio
    tcp_count = float(np.sum(protos == "tcp"))
    udp_count = float(np.sum(protos == "udp"))
    icmp_count = float(np.sum(protos == "icmp"))

    tcp_ratio = float(np.clip(tcp_count / flow_count, 0.0, 1.0))
    udp_ratio = float(np.clip(udp_count / flow_count, 0.0, 1.0))
    icmp_ratio = float(np.clip(icmp_count / flow_count, 0.0, 1.0))

    # 16. closed_flow_ratio (using normalized Argus state_category == 'closed')
    closed_count = float(np.sum(state_cats == "closed"))
    closed_flow_ratio = float(np.clip(closed_count / flow_count, 0.0, 1.0))

    # Build ordered vector
    vec_values = [
        flow_count,
        total_packets,
        total_bytes,
        src_bytes_sum,
        dst_bytes_sum,
        byte_asymmetry_ratio,
        mean_duration,
        max_duration,
        unique_src_ips,
        unique_dst_ips,
        src_ip_entropy,
        dst_ip_entropy,
        tcp_ratio,
        udp_ratio,
        icmp_ratio,
        closed_flow_ratio,
    ]

    vector = np.array(vec_values, dtype=np.float64)
    feat_dict = dict(zip(MACRO_FEATURE_NAMES, vec_values))

    return MacroNetworkState(
        window_idx=window_idx,
        start_time=start_time,
        end_time=end_time,
        vector=vector,
        features=feat_dict,
    )


class MacroNetworkStateBuilder:
    """
    Builder that transforms a sequence of temporal windows into a sequence
    of 16-dimensional MacroNetworkStates.
    """

    def __init__(self, window_size_sec: float = 60.0):
        self.window_size_sec = window_size_sec

    def build_states(
        self,
        normalized_df: pd.DataFrame,
        window_size_sec: Optional[float] = None,
    ) -> Tuple[List[MacroNetworkState], pd.DataFrame, np.ndarray]:
        """
        Build Macro State Vectors for all 60-second windows across normalized_df.

        Parameters:
            normalized_df: DataFrame of normalized flow records.
            window_size_sec: Window size in seconds (overrides instance setting if provided).

        Returns:
            Tuple of:
            1. states: List of MacroNetworkState objects
            2. states_df: DataFrame of shape [num_windows, 19] containing:
                          [window_idx, start_time, end_time] + 16 macro features
            3. feature_matrix: 2D numpy array of shape [num_windows, 16] (float64)
        """
        w_size = window_size_sec or self.window_size_sec

        if normalized_df.empty:
            empty_df = pd.DataFrame(columns=["window_idx", "start_time", "end_time"] + list(MACRO_FEATURE_NAMES))
            empty_mat = np.empty((0, len(MACRO_FEATURE_NAMES)), dtype=np.float64)
            return [], empty_df, empty_mat

        if "timestamp" not in normalized_df.columns:
            raise ValueError("normalized_df must contain a 'timestamp' column.")

        # Sort chronologically
        df_sorted = normalized_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df_sorted.empty:
            empty_df = pd.DataFrame(columns=["window_idx", "start_time", "end_time"] + list(MACRO_FEATURE_NAMES))
            empty_mat = np.empty((0, len(MACRO_FEATURE_NAMES)), dtype=np.float64)
            return [], empty_df, empty_mat

        min_ts = float(df_sorted["timestamp"].min())
        max_ts = float(df_sorted["timestamp"].max())

        t0 = float(np.floor(min_ts))
        total_duration = max_ts - t0
        num_windows = int(np.ceil(total_duration / w_size))
        if num_windows <= 0:
            num_windows = 1

        # Fast bucket assignments using searchsorted
        ts_array = df_sorted["timestamp"].to_numpy(dtype=float)
        window_edges = np.array([t0 + i * w_size for i in range(num_windows + 1)])
        bucket_indices = np.searchsorted(window_edges, ts_array, side="right") - 1
        bucket_indices = np.clip(bucket_indices, 0, num_windows - 1)

        # Precompute column arrays once across the sorted DataFrame
        tot_pkts_all = _extract_numeric_array(df_sorted, "total_packets", "total_packets", default=0.0)
        tot_bytes_all = _extract_numeric_array(df_sorted, "total_bytes", "total_bytes", default=0.0)
        src_bytes_all = _extract_numeric_array(df_sorted, "src_bytes", "src_bytes", default=0.0)
        if "dst_bytes" in df_sorted.columns or ("metadata" in df_sorted.columns and len(df_sorted) > 0 and isinstance(df_sorted["metadata"].iloc[0], dict) and "dst_bytes" in df_sorted["metadata"].iloc[0]):
            dst_bytes_all = _extract_numeric_array(df_sorted, "dst_bytes", "dst_bytes", default=0.0)
        else:
            dst_bytes_all = np.maximum(0.0, tot_bytes_all - src_bytes_all)

        if "duration" in df_sorted.columns:
            durations_all = pd.to_numeric(df_sorted["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            durations_all = np.maximum(0.0, durations_all)
        else:
            durations_all = np.zeros(len(df_sorted), dtype=float)

        src_ips_all = df_sorted["src_ip"].to_numpy() if "src_ip" in df_sorted.columns else np.array([""] * len(df_sorted))
        dst_ips_all = df_sorted["dst_ip"].to_numpy() if "dst_ip" in df_sorted.columns else np.array([""] * len(df_sorted))
        protos_all = df_sorted["protocol"].astype(str).str.lower().str.strip().to_numpy() if "protocol" in df_sorted.columns else np.array([""] * len(df_sorted))
        state_cats_all = _extract_state_category_array(df_sorted)

        states: List[MacroNetworkState] = []
        rows_list: List[Dict[str, Any]] = []
        matrix_rows: List[np.ndarray] = []

        # Vectorized slice processing
        split_indices = np.searchsorted(bucket_indices, np.arange(num_windows + 1))

        for w_idx in range(num_windows):
            w_start = t0 + w_idx * w_size
            w_end = w_start + w_size

            idx_start = split_indices[w_idx]
            idx_end = split_indices[w_idx + 1]

            if idx_end > idx_start:
                precomputed_slice = {
                    "tot_pkts": tot_pkts_all[idx_start:idx_end],
                    "tot_bytes": tot_bytes_all[idx_start:idx_end],
                    "src_bytes": src_bytes_all[idx_start:idx_end],
                    "dst_bytes": dst_bytes_all[idx_start:idx_end],
                    "durations": durations_all[idx_start:idx_end],
                    "src_ips": src_ips_all[idx_start:idx_end],
                    "dst_ips": dst_ips_all[idx_start:idx_end],
                    "protos": protos_all[idx_start:idx_end],
                    "state_cats": state_cats_all[idx_start:idx_end],
                }
                state = extract_macro_state_from_flows(
                    flow_df=df_sorted.iloc[0:0],
                    window_idx=w_idx,
                    start_time=w_start,
                    end_time=w_end,
                    precomputed=precomputed_slice,
                )
            else:
                state = extract_macro_state_from_flows(
                    flow_df=df_sorted.iloc[0:0],
                    window_idx=w_idx,
                    start_time=w_start,
                    end_time=w_end,
                )

            states.append(state)
            rows_list.append(state.to_dict())
            matrix_rows.append(state.vector)

        states_df = pd.DataFrame(rows_list)
        feature_matrix = np.array(matrix_rows, dtype=np.float64)

        return states, states_df, feature_matrix
