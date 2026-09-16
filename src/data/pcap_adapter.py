"""Offline Scapy PCAP adapter into the existing normalized flow pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple

import pandas as pd

from ..features.packet_features import aggregate_packet_flows, packets_to_dataframe


class PCAPAdapter:
    """Read packets with Scapy and expose packet, flow, and normalized views."""

    dataset_name = "PCAP"

    def read_packets(self, path: str | Path) -> pd.DataFrame:
        from scapy.utils import PcapReader
        with PcapReader(str(path)) as reader:
            return packets_to_dataframe(reader)

    def packets_to_flows(self, packets: pd.DataFrame) -> pd.DataFrame:
        return aggregate_packet_flows(packets)

    def load(self, path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        packet_df = self.read_packets(path)
        return packet_df, self.packets_to_flows(packet_df)

    @staticmethod
    def to_network_state_input(flow_df: pd.DataFrame) -> pd.DataFrame:
        """Return the columns and semantics consumed by MacroNetworkStateBuilder."""
        required = {
            "timestamp": 0.0, "duration": 0.0, "protocol": "", "src_ip": "", "dst_ip": "",
            "total_packets": 0.0, "total_bytes": 0.0, "src_bytes": 0.0,
            "state_category": "other", "raw_label": None, "is_attack": None,
        }
        normalized = flow_df.copy()
        for column, default in required.items():
            if column not in normalized.columns:
                normalized[column] = default
        normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="coerce")
        normalized["duration"] = pd.to_numeric(normalized["duration"], errors="coerce").fillna(0.0)
        normalized["protocol"] = normalized["protocol"].astype(str).str.lower().str.strip()
        normalized["total_packets"] = pd.to_numeric(normalized["total_packets"], errors="coerce").fillna(0.0)
        normalized["total_bytes"] = pd.to_numeric(normalized["total_bytes"], errors="coerce").fillna(0.0)
        normalized["src_bytes"] = pd.to_numeric(normalized["src_bytes"], errors="coerce").fillna(0.0)
        normalized["is_attack"] = pd.Series([None] * len(normalized), index=normalized.index, dtype=object)
        return normalized

    def load_for_network_state(self, path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        packet_df, flow_df = self.load(path)
        normalized = self.to_network_state_input(flow_df)
        return packet_df, flow_df, normalized
