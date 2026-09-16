#!/usr/bin/env python
"""Generate a tiny offline PCAP and run it through the existing state pipeline."""

import tempfile
import sys
from pathlib import Path

import numpy as np
from scapy.all import IP, TCP, UDP, Raw, wrpcap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pcap_adapter import PCAPAdapter
from src.features.network_state import MACRO_FEATURE_NAMES, MacroNetworkStateBuilder


def build_demo_pcap(path: Path) -> None:
    packets = [
        IP(src="192.0.2.10", dst="198.51.100.20", ttl=64) / TCP(sport=40000, dport=443, flags="S", window=8192),
        IP(src="192.0.2.10", dst="198.51.100.20", ttl=63) / TCP(sport=40000, dport=443, flags="PA", window=8192) / Raw(b"demo-payload"),
        IP(src="192.0.2.11", dst="198.51.100.53", ttl=64) / UDP(sport=53000, dport=53) / Raw(b"dns"),
    ]
    for index, packet in enumerate(packets):
        packet.time = 1700000000.0 + index * 0.5
    wrpcap(str(path), packets)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "netguard_demo.pcap"
        build_demo_pcap(path)
        packet_df, flow_df, normalized = PCAPAdapter().load_for_network_state(path)
        states, _, matrix = MacroNetworkStateBuilder(window_size_sec=60.0).build_states(normalized)

    print("PCAP packet pipeline")
    print(f"Packets parsed: {len(packet_df)}")
    print(f"Flows generated: {len(flow_df)}")
    print("Packet-level feature summary:")
    print(packet_df[["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "packet_length", "ttl", "tcp_flags", "tcp_window", "payload_size", "fragmented"]].to_string(index=False))
    print("Normalized records:")
    print(normalized[["timestamp", "src_ip", "dst_ip", "protocol", "duration", "total_packets", "total_bytes", "src_bytes", "is_attack"]].to_string(index=False))
    print(f"Network-state windows: {len(states)}")
    print(f"Network-state feature names: {', '.join(MACRO_FEATURE_NAMES)}")
    print(f"Finite state matrix: {bool(np.isfinite(matrix).all())}")


if __name__ == "__main__":
    main()
