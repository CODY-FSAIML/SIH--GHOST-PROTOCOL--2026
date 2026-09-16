#!/usr/bin/env python
"""Profile the GNN preprocessing pipeline.

Outputs:
- Scenario 42 loading & preprocessing time (CSV load, normalization, macro states, windows)
- Temporal‑window construction time (re‑computed)
- Old and optimized graph construction time for the first five windows
"""

import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path (one level up from this file's directory)
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports
from src.data.adapters.ctu13_adapter import _state_category
from src.features.network_state import MacroNetworkStateBuilder
from src.data.temporal_targets import build_temporal_windows
from src.graph.communication_graph import (
    build_communication_graph,
    build_communication_graph_reference,
    build_communication_graphs,
    partition_flows_by_window,
)


def load_raw_capture(filepath: str, name: str) -> dict:
    """Load a CTU‑13 capture, normalize it, compute macro states and windows.

    Returns a dictionary with keys: name, norm_df, states_lookup, windows.
    """
    print(f"Loading {name} from {filepath} …")
    t_start = time.time()
    cols = ["StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "TotPkts", "TotBytes", "SrcBytes", "State", "Label"]
    df_raw = pd.read_csv(filepath, usecols=cols, low_memory=False)
    print(f"  Loaded {len(df_raw):,} flows in {time.time() - t_start:.2f}s")

    # Timestamp conversion
    dt = pd.to_datetime(df_raw["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
    ts = dt.astype("datetime64[ns]").astype(np.int64) / 1e9
    labels = df_raw["Label"].astype(str)
    is_attack = labels.str.contains("Botnet", case=True, regex=False).astype(int)
    state_cats = [_state_category(str(s)) for s in df_raw["State"]]

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
    # Build macro states for each 60‑second window
    states, _, _ = builder.build_states(norm_df)
    states_lookup = {s.window_idx: s for s in states}
    # Build temporal windows (list of Window objects with flow indices)
    windows = build_temporal_windows(norm_df, window_size_sec=60.0)
    return {"name": name, "norm_df": norm_df, "states_lookup": states_lookup, "windows": windows}


def windows_to_graph_data_optimized(windows, states_lookup, norm_df, scenario_name=""):
    """Optimized conversion of windows to torch_geometric Data objects.
    Returns (data_list, labels).
    """
    from torch_geometric.data import Data
    window_frames = partition_flows_by_window(norm_df, list(windows[:-1]))
    data_list = []
    labels = []
    total = len(windows) - 1
    start = time.time()
    for t in range(total):
        df_window = window_frames[t]
        target_w = windows[t + 1]
        graph = build_communication_graph(df_window)
        node_feat = graph["node_features"]
        edge_index = graph["edge_index"]
        if node_feat.shape[0] == 0:
            node_feat = np.zeros((1, 6), dtype=np.float32)
            edge_index = np.empty((2, 0), dtype=np.int64)
        data = Data(x=torch.from_numpy(node_feat).float(), edge_index=torch.from_numpy(edge_index).long())
        data.y = torch.tensor(int(not target_w.is_empty and target_w.y == 1), dtype=torch.long)
        data_list.append(data)
        labels.append(int(not target_w.is_empty and target_w.y == 1))
        if (t + 1) % 50 == 0 or (t + 1) == total:
            elapsed = time.time() - start
            avg = elapsed / (t + 1)
            remaining = (total - (t + 1)) * avg
            print(f"[{scenario_name}] (opt) Building graph {t + 1}/{total} – elapsed {elapsed:.1f}s, avg {avg:.3f}s/win, est. remaining {remaining/60:.1f}min")
    return data_list, labels


def _graph_arrays_equivalent(reference, optimized):
    return (
        reference["node_ids"] == optimized["node_ids"]
        and np.allclose(reference["node_features"], optimized["node_features"], rtol=1e-12, atol=1e-12)
        and np.array_equal(reference["edge_index"], optimized["edge_index"])
        and np.allclose(reference["edge_features"], optimized["edge_features"], rtol=1e-12, atol=1e-12)
    )


def main():
    SCENARIO_42_PATH = r"c:\\Users\\sridevi\\OneDrive\\Desktop\\sih\\data\\raw\\ctu13\\capture20110810.binetflow.txt"
    # ------------------------------------------------------------
    # A. Load & preprocess Scenario 42
    # ------------------------------------------------------------
    t0 = time.time()
    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    load_time = time.time() - t0
    print(f"\nA. Scenario 42 loading & preprocessing time: {load_time:.2f}s")

    # ------------------------------------------------------------
    # B. Temporal‑window construction time (re‑computed for clarity)
    # ------------------------------------------------------------
    t1 = time.time()
    windows = build_temporal_windows(cap42["norm_df"], window_size_sec=60.0)
    win_time = time.time() - t1
    print(f"B. Temporal-window construction time (re-computed): {win_time:.2f}s")

    # ------------------------------------------------------------
    # C. First 5 graph constructions: reference versus optimized version
    # ------------------------------------------------------------
    # We need at least 6 windows to produce 5 graphs (each graph uses window t and label from t+1)
    sample_windows = cap42["windows"][:6]
    sample_frames = partition_flows_by_window(cap42["norm_df"], list(sample_windows[:-1]))
    t2 = time.perf_counter()
    reference_graphs = [build_communication_graph_reference(frame) for frame in sample_frames]
    old_graph_time = time.perf_counter() - t2
    t3 = time.perf_counter()
    optimized_graphs = build_communication_graphs(cap42["norm_df"], list(sample_windows[:-1]))
    new_graph_time = time.perf_counter() - t3
    equivalent = all(_graph_arrays_equivalent(old, new) for old, new in zip(reference_graphs, optimized_graphs))
    speedup = old_graph_time / new_graph_time if new_graph_time > 0 else float("inf")
    print(f"C. First {len(reference_graphs)} graph constructions time (old): {old_graph_time:.2f}s")
    print(f"   First {len(optimized_graphs)} graph constructions time (optimized): {new_graph_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Equivalence (node IDs/features, edge index/features; rtol=1e-12, atol=1e-12): {equivalent}")

if __name__ == "__main__":
    main()
