import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any

# Optional import of torch_geometric – only if already installed
try:
    import torch
    from torch_geometric.data import Data
    _TORCH_GEOMETRIC_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_GEOMETRIC_AVAILABLE = False


def _compute_node_features(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Compute deterministic node features for a single window.

    Returns a sorted list of IP identifiers and a NumPy array of shape
    (num_nodes, 6) containing the following columns:
        0: in_degree
        1: out_degree
        2: in_bytes
        3: out_bytes
        4: in_packets
        5: out_packets
    """
    src_ips = df["src_ip"].dropna().astype(str).tolist()
    dst_ips = df["dst_ip"].dropna().astype(str).tolist()
    all_ips = sorted(set(src_ips + dst_ips))
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    num_nodes = len(all_ips)
    features = np.zeros((num_nodes, 6), dtype=np.float64)
    if df.empty:
        return all_ips, features

    # Directional byte columns – prefer canonical names if available
    out_bytes_series = df.get("bytes_sent", df.get("src_bytes", pd.Series()))
    in_bytes_series = df.get("bytes_received", df.get("dst_bytes", pd.Series()))
    out_bytes = pd.to_numeric(out_bytes_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    in_bytes = pd.to_numeric(in_bytes_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Bidirectional totals – prefer explicit total columns, otherwise derive
    total_bytes_series = df.get("total_bytes", None)
    if total_bytes_series is None:
        # Derive total as sum of directional bytes when both are present
        total_bytes = out_bytes + in_bytes
    else:
        total_bytes = pd.to_numeric(total_bytes_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Packets – prefer directional if present, else fall back to total_packets
    out_packets_series = df.get("packets_sent", pd.Series())
    in_packets_series = df.get("packets_received", pd.Series())
    out_packets = pd.to_numeric(out_packets_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    in_packets = pd.to_numeric(in_packets_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    total_packets_series = df.get("total_packets", None)
    if total_packets_series is None:
        total_packets = out_packets + in_packets
    else:
        total_packets = pd.to_numeric(total_packets_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    for src, dst, ob, ib, op, ip in zip(src_ips, dst_ips, out_bytes, in_bytes, out_packets, in_packets):
        src_idx = ip_to_idx[src]
        dst_idx = ip_to_idx[dst]
        # out metrics for source node
        features[src_idx, 1] += 1               # out_degree (edge count)
        features[src_idx, 3] += ob               # out_bytes (bytes sent)
        features[src_idx, 5] += op               # out_packets (sent)
        # in metrics for destination node
        features[dst_idx, 0] += 1               # in_degree
        features[dst_idx, 2] += ib               # in_bytes (bytes received)
        features[dst_idx, 4] += ip               # in_packets (received)
    return all_ips, features


def _compute_edge_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Aggregate edge features for a window.

    Returns ``edge_index`` (2, num_edges) int64 NumPy array, ``edge_feat``
    (num_edges, 6) float64 array with columns
        [flow_count, total_bytes, total_packets, mean_duration, tcp_ratio, udp_ratio]
    and a list of ``(src_ip, dst_ip)`` tuples for reference.
    """
    if df.empty:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 6), dtype=np.float64), []

    # Group by source/destination pair
    grouped = df.groupby(["src_ip", "dst_ip"], as_index=False)
    edge_meta: List[Tuple[str, str]] = []
    flow_counts: List[int] = []
    total_bytes_list: List[float] = []
    total_packets_list: List[float] = []
    mean_durations: List[float] = []
    tcp_ratios: List[float] = []
    udp_ratios: List[float] = []

    for _, row in grouped:
        src = str(row["src_ip"].iloc[0])
        dst = str(row["dst_ip"].iloc[0])
        edge_meta.append((src, dst))
        cnt = len(row)
        flow_counts.append(cnt)
        # Use canonical totals if present, otherwise fall back to directional sums
        if "total_bytes" in row.columns:
            tb = pd.to_numeric(row["total_bytes"], errors="coerce").fillna(0.0).sum()
        else:
            # Derive from bytes_sent/bytes_received if both exist
            sb = pd.to_numeric(row.get("bytes_sent", pd.Series()), errors="coerce").fillna(0.0).sum()
            db = pd.to_numeric(row.get("bytes_received", pd.Series()), errors="coerce").fillna(0.0).sum()
            tb = sb + db
        if "total_packets" in row.columns:
            tp = pd.to_numeric(row["total_packets"], errors="coerce").fillna(0.0).sum()
        else:
            sp = pd.to_numeric(row.get("packets_sent", pd.Series()), errors="coerce").fillna(0.0).sum()
            rp = pd.to_numeric(row.get("packets_received", pd.Series()), errors="coerce").fillna(0.0).sum()
            tp = sp + rp
        total_bytes_list.append(float(tb))
        total_packets_list.append(float(tp))
        if "duration" in row.columns:
            dur = pd.to_numeric(row["duration"], errors="coerce").fillna(0.0).mean()
        else:
            dur = 0.0
        mean_durations.append(float(dur))
        protos = row.get("protocol", pd.Series([""]))
        protos = protos.astype(str).str.lower().str.strip()
        tcp_cnt = (protos == "tcp").sum()
        udp_cnt = (protos == "udp").sum()
        tcp_ratios.append(float(tcp_cnt / cnt) if cnt > 0 else 0.0)
        udp_ratios.append(float(udp_cnt / cnt) if cnt > 0 else 0.0)

    # Deterministic node ordering – sorted unique IPs across all edges
    src_ips, dst_ips = zip(*edge_meta) if edge_meta else ([], [])
    all_nodes = sorted(set(src_ips + dst_ips))
    ip_to_idx = {ip: i for i, ip in enumerate(all_nodes)}
    edge_index = np.vstack([
        np.array([ip_to_idx[s] for s in src_ips], dtype=np.int64),
        np.array([ip_to_idx[d] for d in dst_ips], dtype=np.int64),
    ])
    edge_feat = np.column_stack([
        np.array(flow_counts, dtype=np.float64),
        np.array(total_bytes_list, dtype=np.float64),
        np.array(total_packets_list, dtype=np.float64),
        np.array(mean_durations, dtype=np.float64),
        np.array(tcp_ratios, dtype=np.float64),
        np.array(udp_ratios, dtype=np.float64),
    ])
    return edge_index, edge_feat, edge_meta


def _build_graph_result(
    node_ids: List[str],
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    edge_meta: List[Tuple[str, str]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "node_ids": node_ids,
        "node_features": node_feat,
        "edge_index": edge_index,
        "edge_features": edge_feat,
        "edge_meta": edge_meta,
    }
    if _TORCH_GEOMETRIC_AVAILABLE:
        data = Data(
            x=torch.from_numpy(node_feat).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_feat).float(),
        )
        result["data"] = data
    else:
        result["torch_geometric_missing"] = True
    return result


def build_communication_graph_reference(df_window: pd.DataFrame) -> Dict[str, Any]:
    """Reference implementation retained for preprocessing equivalence tests."""
    node_ids, node_feat = _compute_node_features(df_window)
    edge_index, edge_feat, edge_meta = _compute_edge_features(df_window)
    return _build_graph_result(node_ids, node_feat, edge_index, edge_feat, edge_meta)


def _build_communication_graph_optimized(df_window: pd.DataFrame) -> Dict[str, Any]:
    """Build one graph using vectorized node and edge aggregation."""
    if df_window.empty:
        return _build_graph_result(
            [],
            np.empty((0, 6), dtype=np.float64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 6), dtype=np.float64),
            [],
        )

    src_series = df_window["src_ip"].dropna().astype(str)
    dst_series = df_window["dst_ip"].dropna().astype(str)
    node_ids = sorted(set(src_series.tolist()) | set(dst_series.tolist()))
    ip_to_idx = {ip: i for i, ip in enumerate(node_ids)}

    # Preserve the existing node calculation's independent dropna behavior.
    node_features = np.zeros((len(node_ids), 6), dtype=np.float64)
    src_values = src_series.to_numpy()
    dst_values = dst_series.to_numpy()
    out_bytes = pd.to_numeric(df_window.get("bytes_sent", pd.Series()), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    in_bytes = pd.to_numeric(df_window.get("bytes_received", df_window.get("dst_bytes", pd.Series())), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out_packets = pd.to_numeric(df_window.get("packets_sent", pd.Series()), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    in_packets = pd.to_numeric(df_window.get("packets_received", pd.Series()), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    edge_out_bytes = pd.to_numeric(df_window.get("bytes_sent", pd.Series(index=df_window.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    edge_in_bytes = pd.to_numeric(df_window.get("bytes_received", df_window.get("dst_bytes", pd.Series(index=df_window.index))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    edge_out_packets = pd.to_numeric(df_window.get("packets_sent", pd.Series(index=df_window.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    edge_in_packets = pd.to_numeric(df_window.get("packets_received", pd.Series(index=df_window.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for src, dst, out_byte, in_byte, out_packet, in_packet in zip(
        src_values, dst_values, out_bytes, in_bytes, out_packets, in_packets
    ):
        node_features[ip_to_idx[src], 1] += 1
        node_features[ip_to_idx[src], 3] += out_byte
        node_features[ip_to_idx[src], 5] += out_packet
        node_features[ip_to_idx[dst], 0] += 1
        node_features[ip_to_idx[dst], 2] += in_byte
        node_features[ip_to_idx[dst], 4] += in_packet

    # Normalize columns once, then aggregate all edge metrics in one groupby.
    work = pd.DataFrame(index=df_window.index)
    work["src_ip"] = df_window["src_ip"]
    work["dst_ip"] = df_window["dst_ip"]
    if "total_bytes" in df_window.columns:
        work["total_bytes"] = pd.to_numeric(df_window["total_bytes"], errors="coerce").fillna(0.0)
    else:
        work["total_bytes"] = edge_out_bytes + edge_in_bytes
    if "total_packets" in df_window.columns:
        work["total_packets"] = pd.to_numeric(df_window["total_packets"], errors="coerce").fillna(0.0)
    else:
        work["total_packets"] = edge_out_packets + edge_in_packets
    work["duration"] = pd.to_numeric(df_window.get("duration", pd.Series(index=df_window.index)), errors="coerce").fillna(0.0)
    protocols = df_window.get("protocol", pd.Series("", index=df_window.index)).astype(str).str.lower().str.strip()
    work["tcp"] = (protocols == "tcp").astype(np.int64)
    work["udp"] = (protocols == "udp").astype(np.int64)
    grouped = work.groupby(["src_ip", "dst_ip"], sort=True, dropna=True).agg(
        flow_count=("src_ip", "size"),
        total_bytes=("total_bytes", "sum"),
        total_packets=("total_packets", "sum"),
        mean_duration=("duration", "mean"),
        tcp_count=("tcp", "sum"),
        udp_count=("udp", "sum"),
    ).reset_index()

    edge_meta = [(str(src), str(dst)) for src, dst in zip(grouped["src_ip"], grouped["dst_ip"])]
    edge_index = np.array(
        [[ip_to_idx[src] for src, _ in edge_meta], [ip_to_idx[dst] for _, dst in edge_meta]],
        dtype=np.int64,
    ) if edge_meta else np.empty((2, 0), dtype=np.int64)
    counts = grouped["flow_count"].to_numpy(dtype=np.float64)
    edge_features = np.column_stack([
        counts,
        grouped["total_bytes"].to_numpy(dtype=np.float64),
        grouped["total_packets"].to_numpy(dtype=np.float64),
        grouped["mean_duration"].to_numpy(dtype=np.float64),
        grouped["tcp_count"].to_numpy(dtype=np.float64) / counts,
        grouped["udp_count"].to_numpy(dtype=np.float64) / counts,
    ]) if edge_meta else np.empty((0, 6), dtype=np.float64)
    return _build_graph_result(node_ids, node_features, edge_index, edge_features, edge_meta)


def partition_flows_by_window(
    normalized_df: pd.DataFrame, windows: List[Any]
) -> List[pd.DataFrame]:
    """Partition normalized flows once using the indices stored on each window."""
    partitions: List[List[int]] = [[] for _ in windows]
    window_by_flow_index = {
        flow_index: window_position
        for window_position, window in enumerate(windows)
        for flow_index in window.flow_indices
    }
    for position, flow_index in enumerate(normalized_df.index):
        window_position = window_by_flow_index.get(flow_index)
        if window_position is not None:
            partitions[window_position].append(position)
    return [
        normalized_df.iloc[positions] if positions else normalized_df.iloc[0:0]
        for positions in partitions
    ]


def build_communication_graph(df_window: pd.DataFrame) -> Dict[str, Any]:
    """Construct a directed communication graph for a single 60‑second window.

    Parameters
    ----------
    df_window: pd.DataFrame
        Normalized flow records belonging to the window. Expected columns are
        ``src_ip``, ``dst_ip``, ``total_bytes``, ``total_packets``, ``src_bytes``,
        optionally ``dst_bytes``, ``duration`` and ``protocol``. No label or
        attack‑related columns are accessed, preserving strict anti‑leakage.

    Returns
    -------
    dict
        ``{"node_ids": List[str], "node_features": np.ndarray,
          "edge_index": np.ndarray, "edge_features": np.ndarray,
          "edge_meta": List[Tuple[str, str]]}``
        If ``torch_geometric`` is available, a ``"data"`` key containing a
        ``torch_geometric.data.Data`` object is also provided.
    """
    return _build_communication_graph_optimized(df_window)


def build_communication_graphs(
    normalized_df: pd.DataFrame, windows: List[Any]
) -> List[Dict[str, Any]]:
    """Partition flows once and build optimized graphs in temporal order."""
    return [
        _build_communication_graph_optimized(df_window)
        for df_window in partition_flows_by_window(normalized_df, windows)
    ]
