import numpy as np

from src.explainability.feature_explanations import rank_driving_indicators
from src.explainability.mitre_mapping import map_observable_behavior


def _state(**updates):
    values = {
        "flow_count": 10,
        "total_packets": 100,
        "total_bytes": 1000,
        "src_bytes_sum": 500,
        "dst_bytes_sum": 500,
        "byte_asymmetry_ratio": 0.0,
        "mean_duration": 1.0,
        "max_duration": 2.0,
        "unique_src_ips": 2,
        "unique_dst_ips": 2,
        "src_ip_entropy": 0.5,
        "dst_ip_entropy": 0.5,
        "tcp_ratio": 0.5,
        "udp_ratio": 0.4,
        "icmp_ratio": 0.1,
        "closed_flow_ratio": 0.5,
    }
    values.update(updates)
    return values


def test_reconnaissance_mapping_uses_valid_terms_and_ids():
    result = map_observable_behavior(
        _state(flow_count=30, unique_dst_ips=10, dst_ip_entropy=2.0, tcp_ratio=0.8),
        _state(flow_count=10, unique_dst_ips=2, dst_ip_entropy=0.5, tcp_ratio=0.4),
    )
    assert result["tactic"] == {"name": "Reconnaissance", "id": "TA0043"}
    assert result["technique"] == {"name": "Active Scanning", "id": "T1595"}
    assert 0.0 <= result["evidence_score"] <= 1.0
    assert "ports" not in " ".join(result["supporting_indicators"]).lower()


def test_insufficient_evidence_does_not_force_mapping():
    result = map_observable_behavior(_state(), _state())
    assert result["behavior"] == "Insufficient evidence"
    assert result["tactic"] is None
    assert result["technique"] is None


def test_feature_ranking_and_determinism():
    current = _state(flow_count=20, unique_dst_ips=6)
    history = [_state(flow_count=10, unique_dst_ips=2), _state(flow_count=10, unique_dst_ips=2)]
    first = rank_driving_indicators(current, history, top_k=16)
    second = rank_driving_indicators(current, history, top_k=16)
    assert first == second
    assert first[0]["feature"] in {"flow_count", "unique_dst_ips"}
    assert all(np.isfinite(item["importance"]) for item in first)


def test_missing_optional_features_are_not_fabricated():
    current = {"flow_count": 20, "tcp_ratio": 0.8}
    result = rank_driving_indicators(current, [{"flow_count": 10, "tcp_ratio": 0.4}])
    names = {item["feature"] for item in result}
    assert "tcp_syn_count" not in names
    assert "destination_port_count" not in names
    assert names == {"flow_count", "tcp_ratio"}


def test_no_packet_level_or_causal_claims():
    result = map_observable_behavior(_state(flow_count=20), _state(flow_count=10))
    text = result["explanation"].lower()
    assert "causal" not in text
    assert "syn" not in text
    assert "payload" not in text or "unavailable" in text
