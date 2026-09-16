"""
Unit Tests for NetworkState V1 — Macro State Vector
===================================================
Verifies:
1. 60-second aggregation
2. Flow count
3. Total packet aggregation
4. Total byte aggregation
5. Source / destination byte calculation
6. Byte asymmetry formula
7. Mean duration
8. Max duration
9. Unique source / destination IP counts
10. Source IP entropy
11. Destination IP entropy
12. TCP / UDP / ICMP ratios
13. Closed-flow ratio
14. Empty window behavior
15. Zero-denominator numerical safety
16. No NaN / infinity
17. Label leakage prevention (identical output with or without attack labels)
18. Deterministic repeated execution
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.network_state import (
    MACRO_FEATURE_NAMES,
    MacroNetworkState,
    MacroNetworkStateBuilder,
    extract_macro_state_from_flows,
)


def _create_synthetic_flow_slice() -> pd.DataFrame:
    """Helper to build a controlled single-window DataFrame of 4 flows."""
    return pd.DataFrame({
        "timestamp": [100.0, 110.0, 120.0, 130.0],
        "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3"],
        "dst_ip": ["192.168.1.1", "192.168.1.2", "192.168.1.1", "192.168.1.1"],
        "duration": [1.0, 2.0, 3.0, 4.0],
        "protocol": ["tcp", "tcp", "udp", "icmp"],
        "total_packets": [10, 20, 30, 40],
        "total_bytes": [1000, 2000, 3000, 4000],
        "src_bytes": [600, 1500, 1000, 2000],
        "dst_bytes": [400, 500, 2000, 2000],
        "state_category": ["closed", "established", "closed", "attempt"],
        "is_attack": [0, 1, 0, 1],
        "label": ["background", "botnet", "background", "botnet"],
        "raw_label": ["flow=bg", "flow=botnet", "flow=bg", "flow=botnet"],
    })


class TestNetworkState:

    def test_feature_names_count(self):
        """Verify exactly 16 feature names in expected order."""
        assert len(MACRO_FEATURE_NAMES) == 16
        assert MACRO_FEATURE_NAMES[0] == "flow_count"
        assert MACRO_FEATURE_NAMES[-1] == "closed_flow_ratio"

    def test_single_window_aggregations(self):
        """Verify flow count, packets, bytes, durations, and IPs."""
        df = _create_synthetic_flow_slice()
        state = extract_macro_state_from_flows(df, window_idx=0, start_time=100.0, end_time=160.0)

        # 1. flow_count = 4
        assert state.features["flow_count"] == 4.0

        # 2. total_packets = 10 + 20 + 30 + 40 = 100
        assert state.features["total_packets"] == 100.0

        # 3. total_bytes = 1000 + 2000 + 3000 + 4000 = 10000
        assert state.features["total_bytes"] == 10000.0

        # 4. src_bytes_sum = 600 + 1500 + 1000 + 2000 = 5100
        assert state.features["src_bytes_sum"] == 5100.0

        # 5. dst_bytes_sum = 400 + 500 + 2000 + 2000 = 4900
        assert state.features["dst_bytes_sum"] == 4900.0

        # 6. byte_asymmetry_ratio: abs(5100 - 4900) / (5100 + 4900 + 1e-6) = 200 / 10000.000001 = 0.02
        expected_asym = 200.0 / (10000.0 + 1e-6)
        assert np.isclose(state.features["byte_asymmetry_ratio"], expected_asym)

        # 7. mean_duration = (1 + 2 + 3 + 4) / 4 = 2.5
        assert state.features["mean_duration"] == 2.5

        # 8. max_duration = 4.0
        assert state.features["max_duration"] == 4.0

        # 9. unique_src_ips = 3 (10.0.0.1, 10.0.0.2, 10.0.0.3)
        assert state.features["unique_src_ips"] == 3.0

        # 10. unique_dst_ips = 2 (192.168.1.1, 192.168.1.2)
        assert state.features["unique_dst_ips"] == 2.0

    def test_entropy_calculations(self):
        """Verify Shannon entropy calculation."""
        df = _create_synthetic_flow_slice()
        state = extract_macro_state_from_flows(df, window_idx=0, start_time=100.0, end_time=160.0)

        # src_ip counts: 10.0.0.1 (2), 10.0.0.2 (1), 10.0.0.3 (1) -> probs: [0.5, 0.25, 0.25]
        # H = -(0.5*ln(0.5) + 0.25*ln(0.25) + 0.25*ln(0.25)) = 1.03972
        expected_src_ent = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 0.25 * np.log(0.25))
        assert np.isclose(state.features["src_ip_entropy"], expected_src_ent)

        # dst_ip counts: 192.168.1.1 (3), 192.168.1.2 (1) -> probs: [0.75, 0.25]
        expected_dst_ent = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
        assert np.isclose(state.features["dst_ip_entropy"], expected_dst_ent)

    def test_protocol_and_closed_ratios(self):
        """Verify protocol ratios and closed connection ratio."""
        df = _create_synthetic_flow_slice()
        state = extract_macro_state_from_flows(df, window_idx=0, start_time=100.0, end_time=160.0)

        # Protocols: 2 TCP, 1 UDP, 1 ICMP out of 4
        assert state.features["tcp_ratio"] == 0.5
        assert state.features["udp_ratio"] == 0.25
        assert state.features["icmp_ratio"] == 0.25

        # Closed flows: 2 "closed" out of 4 -> 0.5
        assert state.features["closed_flow_ratio"] == 0.5

    def test_empty_window_behavior(self):
        """Verify that empty windows cleanly return all zeros with no NaN or Inf."""
        empty_df = pd.DataFrame(columns=list(_create_synthetic_flow_slice().columns))
        state = extract_macro_state_from_flows(empty_df, window_idx=5, start_time=300.0, end_time=360.0)

        assert state.window_idx == 5
        assert state.start_time == 300.0
        assert state.end_time == 360.0
        assert len(state.vector) == 16
        assert (state.vector == 0.0).all()
        assert not np.isnan(state.vector).any()
        assert not np.isinf(state.vector).any()

    def test_zero_denominator_numerical_safety(self):
        """Verify zero packet/byte counts do not produce division-by-zero errors."""
        zero_df = pd.DataFrame({
            "timestamp": [10.0],
            "src_ip": ["10.0.0.1"],
            "dst_ip": ["10.0.0.2"],
            "duration": [0.0],
            "protocol": ["other"],
            "total_packets": [0],
            "total_bytes": [0],
            "src_bytes": [0],
            "dst_bytes": [0],
            "state_category": ["unknown"],
        })
        state = extract_macro_state_from_flows(zero_df, window_idx=0, start_time=0.0, end_time=60.0)

        assert state.features["byte_asymmetry_ratio"] == 0.0
        assert state.features["tcp_ratio"] == 0.0
        assert state.features["src_ip_entropy"] == 0.0
        assert not np.isnan(state.vector).any()
        assert not np.isinf(state.vector).any()

    def test_metadata_extraction_fallback(self):
        """Verify extraction from metadata dictionary when columns are nested (as in CTU-13 adapter)."""
        df_meta = pd.DataFrame({
            "timestamp": [10.0, 20.0],
            "src_ip": ["1.1.1.1", "2.2.2.2"],
            "dst_ip": ["3.3.3.3", "4.4.4.4"],
            "duration": [1.0, 2.0],
            "protocol": ["tcp", "udp"],
            "metadata": [
                {
                    "total_packets": 5,
                    "total_bytes": 500,
                    "src_bytes": 200,
                    "dst_bytes": 300,
                    "state_category": "closed",
                },
                {
                    "total_packets": 15,
                    "total_bytes": 1500,
                    "src_bytes": 800,
                    "dst_bytes": 700,
                    "state_category": "established",
                },
            ],
        })
        state = extract_macro_state_from_flows(df_meta, window_idx=0, start_time=0.0, end_time=60.0)

        assert state.features["total_packets"] == 20.0
        assert state.features["total_bytes"] == 2000.0
        assert state.features["src_bytes_sum"] == 1000.0
        assert state.features["dst_bytes_sum"] == 1000.0
        assert state.features["byte_asymmetry_ratio"] == 0.0
        assert state.features["closed_flow_ratio"] == 0.5

    def test_macro_state_builder_multi_window(self):
        """Verify 60s tumbling window segmentation across multiple continuous windows."""
        # 180 seconds -> 3 windows of 60s
        timestamps = [10.0, 20.0, 70.0, 130.0, 140.0]  # w0 has 2, w1 has 1, w2 has 2
        df = pd.DataFrame({
            "timestamp": timestamps,
            "src_ip": ["1.1.1.1"] * 5,
            "dst_ip": ["2.2.2.2"] * 5,
            "duration": [1.0] * 5,
            "protocol": ["tcp"] * 5,
            "total_packets": [10] * 5,
            "total_bytes": [100] * 5,
            "src_bytes": [50] * 5,
            "dst_bytes": [50] * 5,
            "state_category": ["closed"] * 5,
        })

        builder = MacroNetworkStateBuilder(window_size_sec=60.0)
        states, states_df, matrix = builder.build_states(df)

        assert len(states) == 3
        assert matrix.shape == (3, 16)
        assert states_df.shape == (3, 19)

        assert states[0].features["flow_count"] == 2.0
        assert states[1].features["flow_count"] == 1.0
        assert states[2].features["flow_count"] == 2.0

        assert (states_df["flow_count"].to_numpy() == np.array([2.0, 1.0, 2.0])).all()

    def test_strict_label_leakage_invariance(self):
        """
        CRITICAL TEST: Macro State must be 100% identical whether ground truth attack
        labels are present or completely stripped.
        """
        df_with_labels = _create_synthetic_flow_slice()

        # Create identical copy completely stripped of labels
        df_no_labels = df_with_labels.drop(columns=["is_attack", "label", "raw_label"])

        state_with = extract_macro_state_from_flows(df_with_labels, window_idx=0, start_time=100.0, end_time=160.0)
        state_without = extract_macro_state_from_flows(df_no_labels, window_idx=0, start_time=100.0, end_time=160.0)

        # Vectors must be numerically identical down to bit-exact float equality
        assert np.array_equal(state_with.vector, state_without.vector)
        assert state_with.features == state_without.features

    def test_deterministic_repeated_execution(self):
        """Verify that calling builder twice on identical data produces exact identical matrices."""
        df = _create_synthetic_flow_slice()
        builder = MacroNetworkStateBuilder(window_size_sec=60.0)

        _, _, mat1 = builder.build_states(df)
        _, _, mat2 = builder.build_states(df)

        assert np.array_equal(mat1, mat2)


if __name__ == "__main__":
    test_instance = TestNetworkState()
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

    print("=" * 60)
    print(f"RUNNING {len(test_methods)} UNIT TESTS FOR NETWORKSTATE V1")
    print("=" * 60)

    passed = 0
    failed = 0
    for method_name in test_methods:
        try:
            getattr(test_instance, method_name)()
            print(f"  [PASS] {method_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {method_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    if failed > 0:
        exit(1)
