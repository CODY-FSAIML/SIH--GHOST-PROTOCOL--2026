import tempfile
from pathlib import Path

import numpy as np
from scapy.all import IP, TCP, UDP, Raw, wrpcap

from src.data.pcap_adapter import PCAPAdapter
from src.features.network_state import MacroNetworkStateBuilder


def _pcap(path: Path):
    packets = [
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=64) / TCP(sport=1234, dport=80, flags="S", window=4096),
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=63) / TCP(sport=1234, dport=80, flags="PA", window=4096) / Raw(b"hello"),
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=62) / TCP(sport=1234, dport=80, flags="FA", window=2048),
        IP(src="10.0.0.3", dst="10.0.0.4", ttl=128) / UDP(sport=5353, dport=53) / Raw(b"dns"),
        IP(src="10.0.0.5", dst="10.0.0.6", flags="MF", frag=1) / UDP(sport=1, dport=2),
    ]
    for index, packet in enumerate(packets):
        packet.time = 1000.0 + index * 0.25
    wrpcap(str(path), packets)


def test_pcap_parsing_and_packet_fields():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.pcap"
        _pcap(path)
        packet_df, flow_df = PCAPAdapter().load(path)
    assert len(packet_df) == 5
    assert {"10.0.0.1", "10.0.0.3"}.issubset(set(packet_df["src_ip"].dropna()))
    assert int(packet_df.iloc[0]["src_port"]) == 1234
    assert int(packet_df.iloc[0]["dst_port"]) == 80
    assert int(packet_df.iloc[0]["ttl"]) == 64
    assert packet_df.iloc[0]["tcp_flags"] == "S"
    assert int(packet_df.iloc[0]["tcp_window"]) == 4096
    assert packet_df.iloc[1]["payload_size"] == 5
    assert bool(packet_df.iloc[4]["fragmented"])
    assert len(flow_df) == 3


def test_flow_features_and_missing_values():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.pcap"
        _pcap(path)
        packet_df, flow_df = PCAPAdapter().load(path)
    tcp = flow_df[flow_df["protocol"] == "tcp"].iloc[0]
    assert tcp["metadata"]["packet_count"] == 3
    assert tcp["metadata"]["syn_count"] == 1
    assert tcp["metadata"]["fin_count"] == 1
    assert tcp["metadata"]["payload_size_mean"] is not None
    assert tcp["metadata"]["iat_mean"] > 0
    udp = flow_df[flow_df["protocol"] == "udp"].iloc[0]
    assert udp["metadata"]["tcp_window_mean"] is None
    assert udp["metadata"]["retransmission_method"] == "none"
    assert tcp["metadata"]["retransmission_method"].startswith("heuristic")


def test_deterministic_output_and_network_state_compatibility():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.pcap"
        _pcap(path)
        adapter = PCAPAdapter()
        first_packets, first_flows, first_norm = adapter.load_for_network_state(path)
        second_packets, second_flows, second_norm = adapter.load_for_network_state(path)
    assert first_packets.equals(second_packets)
    assert first_flows.equals(second_flows)
    state_builder = MacroNetworkStateBuilder(window_size_sec=60.0)
    states, _, matrix = state_builder.build_states(first_norm)
    assert len(states) == 1
    assert matrix.shape == (1, 16)
    assert np.isfinite(matrix).all()
    assert not first_norm["is_attack"].notna().any()
