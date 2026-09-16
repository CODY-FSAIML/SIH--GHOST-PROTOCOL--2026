"""Packet and session-level feature extraction for offline PCAP ingestion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _layer_payload_size(packet: Any, layer_name: str) -> Optional[int]:
    try:
        layer = packet[layer_name]
        return int(len(bytes(layer.payload)))
    except (IndexError, AttributeError, TypeError, ValueError):
        return None


def packet_to_record(packet: Any, packet_index: int) -> Dict[str, Any]:
    """Extract one deterministic packet record without labels or inferred values."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    if ip is None:
        return {"packet_index": packet_index, "timestamp": _finite(getattr(packet, "time", None)), "src_ip": None, "dst_ip": None, "protocol": None, "packet_length": int(len(packet)), "metadata": {"non_ip": True}}

    if packet.haslayer(TCP):
        protocol = "tcp"
        transport = packet[TCP]
        src_port = int(transport.sport) if transport.sport is not None else None
        dst_port = int(transport.dport) if transport.dport is not None else None
        tcp_flags = str(transport.flags)
        tcp_window = int(transport.window) if transport.window is not None else None
        payload_size = _layer_payload_size(packet, TCP)
        tcp_seq = int(transport.seq) if transport.seq is not None else None
        tcp_ack = int(transport.ack) if transport.ack is not None else None
        tcp_payload_length = payload_size or 0
    elif packet.haslayer(UDP):
        protocol = "udp"
        transport = packet[UDP]
        src_port = int(transport.sport) if transport.sport is not None else None
        dst_port = int(transport.dport) if transport.dport is not None else None
        tcp_flags = None
        tcp_window = None
        payload_size = _layer_payload_size(packet, UDP)
        tcp_seq = None
        tcp_ack = None
        tcp_payload_length = 0
    else:
        protocol = str(getattr(ip, "proto", "ip")).lower()
        src_port = None
        dst_port = None
        tcp_flags = None
        tcp_window = None
        payload_size = None
        tcp_seq = None
        tcp_ack = None
        tcp_payload_length = 0

    ip_flags = None
    try:
        ip_flags = int(ip.flags.value)
    except AttributeError:
        pass
    return {
        "packet_index": packet_index,
        "timestamp": _finite(getattr(packet, "time", None)),
        "src_ip": str(ip.src),
        "dst_ip": str(ip.dst),
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_length": int(len(packet)),
        "ttl": int(ip.ttl) if hasattr(ip, "ttl") and ip.ttl is not None else None,
        "tcp_flags": tcp_flags,
        "tcp_window": tcp_window,
        "ip_flags": ip_flags,
        "fragmented": bool(ip_flags is not None and (ip_flags & 0x1 or (ip_flags & 0x1FFF))),
        "payload_size": payload_size,
        "tcp_seq": tcp_seq,
        "tcp_ack": tcp_ack,
        "tcp_payload_length": tcp_payload_length,
    }


def packets_to_dataframe(packets: Iterable[Any]) -> pd.DataFrame:
    records = [packet_to_record(packet, index) for index, packet in enumerate(packets)]
    columns = [
        "packet_index", "timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port",
        "packet_length", "ttl", "tcp_flags", "tcp_window", "ip_flags", "fragmented", "payload_size",
        "tcp_seq", "tcp_ack", "tcp_payload_length",
    ]
    return pd.DataFrame(records, columns=columns)


def _flag_count(group: pd.DataFrame, flag: str) -> int:
    return int(group["tcp_flags"].fillna("").astype(str).str.contains(flag, regex=False).sum())


def _safe_stats(values: pd.Series) -> Dict[str, Optional[float]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        return {"mean": None, "variance": None, "max": None}
    return {"mean": float(np.mean(numeric)), "variance": float(np.var(numeric)), "max": float(np.max(numeric))}


def _retransmission_count(group: pd.DataFrame) -> Tuple[int, str]:
    tcp = group[(group["protocol"] == "tcp") & group["tcp_seq"].notna()]
    if tcp.empty:
        return 0, "none"
    keys = list(zip(tcp["src_ip"], tcp["dst_ip"], tcp["src_port"], tcp["dst_port"], tcp["tcp_seq"], tcp["tcp_payload_length"]))
    repeated = len(keys) - len(set(keys))
    return int(max(0, repeated)), "heuristic repeated TCP sequence/payload tuple"


def aggregate_packet_flows(packet_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directional endpoint/protocol sessions into normalized-compatible rows.

    Flows are grouped by ``src_ip, dst_ip, protocol``. Multiple destination ports
    remain in metadata as a diversity/list feature rather than being fabricated
    into one canonical port field.
    """
    if packet_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "timestamp_str", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "duration", "bytes_sent", "bytes_received", "packets_sent", "packets_received", "flow_id",
            "dataset_name", "label", "raw_label", "is_attack", "metadata",
        ])
    work = packet_df.dropna(subset=["timestamp", "src_ip", "dst_ip", "protocol"]).sort_values(["timestamp", "packet_index"], kind="mergesort")
    rows: List[Dict[str, Any]] = []
    for (src_ip, dst_ip, protocol), group in work.groupby(["src_ip", "dst_ip", "protocol"], sort=True, dropna=False):
        group = group.sort_values(["timestamp", "packet_index"], kind="mergesort")
        timestamps = group["timestamp"].to_numpy(dtype=float)
        iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)
        ttl_stats = _safe_stats(group["ttl"])
        window_stats = _safe_stats(group["tcp_window"])
        payload_stats = _safe_stats(group["payload_size"])
        packet_sizes = pd.to_numeric(group["packet_length"], errors="coerce").dropna().to_numpy(dtype=float)
        retransmissions, retransmission_method = _retransmission_count(group)
        dst_ports = sorted({int(value) for value in group["dst_port"].dropna().tolist()})
        port_diffs = np.diff(dst_ports) if len(dst_ports) > 1 else np.array([], dtype=int)
        sequential = bool(len(port_diffs) > 0 and np.all(port_diffs == 1))
        randomized = bool(len(dst_ports) > 2 and not sequential and len(set(dst_ports)) == len(dst_ports))
        total_bytes = int(np.sum(packet_sizes)) if len(packet_sizes) else 0
        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        metadata = {
            "packet_count": int(len(group)),
            "total_bytes": total_bytes,
            "mean_packet_size": float(np.mean(packet_sizes)) if len(packet_sizes) else None,
            "ttl_mean": ttl_stats["mean"], "ttl_variance": ttl_stats["variance"],
            "tcp_window_mean": window_stats["mean"], "tcp_window_variance": window_stats["variance"], "tcp_window_max": window_stats["max"],
            "payload_size_mean": payload_stats["mean"], "payload_size_variance": payload_stats["variance"], "payload_size_max": payload_stats["max"],
            "iat_mean": float(np.mean(iats)) if len(iats) else None,
            "iat_variance": float(np.var(iats)) if len(iats) else None,
            "iat_max": float(np.max(iats)) if len(iats) else None,
            "syn_count": _flag_count(group, "S"), "ack_count": _flag_count(group, "A"),
            "fin_count": _flag_count(group, "F"), "rst_count": _flag_count(group, "R"),
            "retransmission_count": retransmissions,
            "retransmission_method": retransmission_method,
            "destination_port_diversity": len(dst_ports),
            "destination_ports": dst_ports,
            "sequential_destination_ports": sequential,
            "randomized_destination_ports": randomized,
            "fragmented_packet_count": int(group["fragmented"].fillna(False).sum()),
            "packet_feature_schema": "pcap_packet_v1",
        }
        first = group.iloc[0]
        rows.append({
            "timestamp": float(timestamps[0]),
            "timestamp_str": pd.to_datetime(timestamps[0], unit="s", utc=True).isoformat(),
            "src_ip": str(src_ip), "src_port": int(first["src_port"]) if pd.notna(first["src_port"]) else None,
            "dst_ip": str(dst_ip), "dst_port": int(first["dst_port"]) if pd.notna(first["dst_port"]) else None,
            "protocol": str(protocol), "duration": duration,
            "bytes_sent": total_bytes, "bytes_received": 0,
            "packets_sent": int(len(group)), "packets_received": 0,
            "flow_id": f"pcap:{src_ip}:{dst_ip}:{protocol}", "dataset_name": "PCAP",
            "label": None, "raw_label": None, "is_attack": None,
            "total_packets": int(len(group)), "total_bytes": total_bytes, "src_bytes": total_bytes,
            "dst_bytes": 0, "state_category": "other", "metadata": metadata,
        })
    return pd.DataFrame(rows)
