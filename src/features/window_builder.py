from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class TimeWindowBuilder:
    """
    Aggregates Normalized Network Records into temporal Network State windows
    for Transformer and Graph Neural Network (GNN) model consumption.
    """

    def __init__(self, window_size_sec: float = 10.0, step_size_sec: Optional[float] = None):
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec or window_size_sec

    def build_windows(self, normalized_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Group normalized records into time windows and extract network state summary features.
        
        Returns:
            List of window dictionaries containing aggregated metrics and node/edge features.
        """
        if normalized_df.empty:
            return []

        # Parse timestamp column to numeric float if needed
        df = normalized_df.copy()
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            timestamps = df["timestamp"].astype(float)
        elif "timestamp_str" in df.columns:
            timestamps = pd.to_datetime(df["timestamp_str"]).astype(int) / 1e9
        else:
            # Fallback to row index if no timestamp available
            timestamps = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

        df["_ts"] = timestamps
        df = df.sort_values("_ts")

        min_ts = df["_ts"].min()
        max_ts = df["_ts"].max()

        windows = []
        current_start = min_ts

        while current_start <= max_ts:
            current_end = current_start + self.window_size_sec
            window_records = df[(df["_ts"] >= current_start) & (df["_ts"] < current_end)]

            if not window_records.empty:
                # Compute network state aggregations
                num_records = len(window_records)
                total_bytes = (window_records["bytes_sent"].fillna(0) + window_records["bytes_received"].fillna(0)).sum()
                total_packets = (window_records["packets_sent"].fillna(0) + window_records["packets_received"].fillna(0)).sum()
                attack_count = (window_records["is_attack"] == 1).sum()

                # Extract unique host pairs for Graph neural net edges
                src_hosts = window_records["src_ip"].dropna().unique().tolist()
                dst_hosts = window_records["dst_ip"].dropna().unique().tolist()
                unique_hosts = list(set(src_hosts + dst_hosts))

                state = {
                    "window_start": current_start,
                    "window_end": current_end,
                    "record_count": num_records,
                    "total_bytes": total_bytes,
                    "total_packets": total_packets,
                    "attack_count": attack_count,
                    "is_attack_window": int(attack_count > 0),
                    "unique_hosts_count": len(unique_hosts),
                    "records_df": window_records
                }
                windows.append(state)

            current_start += self.step_size_sec

        return windows
