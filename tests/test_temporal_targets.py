"""
Unit Tests for Temporal Forecasting Targets and Chronological Splitting
======================================================================
Verifies:
1. 60-second window construction
2. Chronological ordering
3. Attack label construction
4. Empty-window handling
5. t+1 / t+2 / t+3 target generation
6. Final-window exclusion
7. Chronological train/validation/test split
8. No overlap across partitions
9. Attack metadata does not appear in input features
10. Deterministic repeated execution
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.temporal_targets import (
    TemporalWindow,
    ForecastingSample,
    ChronologicalSplit,
    build_temporal_windows,
    build_forecasting_samples,
    chronological_split,
)


def _create_synthetic_flow_df(
    start_time: float = 1000.0,
    duration_seconds: float = 600.0,
    flow_step: float = 10.0,
    attack_intervals: list = None,
) -> pd.DataFrame:
    """
    Helper to generate a predictable synthetic DataFrame conforming to canonical schema.
    """
    attack_intervals = attack_intervals or []
    timestamps = np.arange(start_time, start_time + duration_seconds, flow_step)
    n = len(timestamps)

    is_attack = []
    raw_labels = []
    for ts in timestamps:
        in_attack = any(s <= ts < e for s, e in attack_intervals)
        if in_attack:
            is_attack.append(1)
            raw_labels.append("flow=From-Botnet-V42-UDP-DNS")
        else:
            is_attack.append(0)
            raw_labels.append("flow=Background")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "src_ip": ["192.168.1.1"] * n,
        "dst_ip": ["10.0.0.1"] * n,
        "src_port": [12345] * n,
        "dst_port": [80] * n,
        "protocol": ["tcp"] * n,
        "duration": [0.5] * n,
        "raw_label": raw_labels,
        "label": ["botnet" if a == 1 else "background" for a in is_attack],
        "is_attack": is_attack,
    })
    return df


class TestTemporalTargets:

    def test_60s_window_construction(self):
        """Verify 60-second non-overlapping tumbling window generation."""
        # 300 seconds of traffic -> exactly 5 windows of 60s
        df = _create_synthetic_flow_df(start_time=1000.0, duration_seconds=300.0, flow_step=10.0)
        windows = build_temporal_windows(df, window_size_sec=60.0)

        assert len(windows) == 5
        for idx, w in enumerate(windows):
            assert w.window_idx == idx
            assert w.start_time == 1000.0 + idx * 60.0
            assert w.end_time == w.start_time + 60.0
            assert w.total_flows == 6  # 6 flows per 60s at 10s step

    def test_chronological_ordering(self):
        """Verify strictly monotonic chronological ordering of windows."""
        # Shuffle inputs to ensure sorting works internally
        df = _create_synthetic_flow_df(start_time=5000.0, duration_seconds=360.0, flow_step=5.0)
        df_shuffled = df.sample(frac=1.0, random_state=42)

        windows = build_temporal_windows(df_shuffled, window_size_sec=60.0)
        for i in range(len(windows) - 1):
            assert windows[i].start_time < windows[i + 1].start_time
            assert windows[i].end_time == windows[i + 1].start_time
            assert windows[i].window_idx + 1 == windows[i + 1].window_idx

    def test_attack_label_construction(self):
        """Verify binary y=1 iff at least one attack flow is present in the window."""
        # Window 0: 0 to 60s (Benign)
        # Window 1: 60 to 120s (Contains attack at 80-90s)
        # Window 2: 120 to 180s (Benign)
        df = _create_synthetic_flow_df(
            start_time=0.0,
            duration_seconds=180.0,
            flow_step=10.0,
            attack_intervals=[(80.0, 95.0)],
        )
        windows = build_temporal_windows(df, window_size_sec=60.0)

        assert len(windows) == 3
        assert windows[0].y == 0
        assert windows[0].attack_flows == 0

        assert windows[1].y == 1
        assert windows[1].attack_flows > 0
        assert windows[1].attack_ratio > 0.0
        assert "flow=From-Botnet-V42-UDP-DNS" in windows[1].active_attack_actions

        assert windows[2].y == 0
        assert windows[2].attack_flows == 0

    def test_empty_window_handling(self):
        """Verify explicit preservation of empty windows with y=0."""
        # Flow at 0s, next flow at 130s -> Window 1 (60s - 120s) is completely empty
        df = pd.DataFrame({
            "timestamp": [0.0, 130.0],
            "is_attack": [0, 1],
            "raw_label": ["bg", "atk"],
        })
        windows = build_temporal_windows(df, window_size_sec=60.0)

        # Spans [0, 60), [60, 120), [120, 180) -> 3 windows
        assert len(windows) == 3
        assert windows[0].total_flows == 1
        assert windows[0].is_empty is False

        assert windows[1].total_flows == 0
        assert windows[1].is_empty is True
        assert windows[1].y == 0
        assert windows[1].attack_flows == 0

        assert windows[2].total_flows == 1
        assert windows[2].is_empty is False
        assert windows[2].y == 1

    def test_target_generation_k_steps(self):
        """Verify t+1, t+2, t+3 multi-horizon target construction."""
        # 10 windows. Let window 5 contain attack.
        df = _create_synthetic_flow_df(
            start_time=0.0,
            duration_seconds=600.0,
            flow_step=10.0,
            attack_intervals=[(310.0, 350.0)],  # Inside window 5 (300 - 360s)
        )
        windows = build_temporal_windows(df, window_size_sec=60.0)
        assert windows[5].y == 1

        samples = build_forecasting_samples(windows, history_len=3, horizons=(1, 2, 3))

        # Check sample at t=4:
        # t=4 (window 4) -> targets are t+1 (w5), t+2 (w6), t+3 (w7)
        sample_t4 = next(s for s in samples if s.anchor_window_idx == 4)
        assert sample_t4.Y[1] == 1  # window 5 has attack
        assert sample_t4.Y[2] == 0  # window 6 benign
        assert sample_t4.Y[3] == 0  # window 7 benign

        # Check sample at t=3:
        # t=3 -> targets are w4 (0), w5 (1), w6 (0)
        sample_t3 = next(s for s in samples if s.anchor_window_idx == 3)
        assert sample_t3.Y[1] == 0
        assert sample_t3.Y[2] == 1
        assert sample_t3.Y[3] == 0

    def test_final_window_exclusion(self):
        """Verify final windows without full future horizons are strictly excluded."""
        # 6 windows (idx 0 to 5)
        # With H=3 and horizons=(1, 2, 3):
        # Valid anchor t must have t-2 >= 0 (t >= 2)
        # And t+3 <= 5 (t <= 2)
        # Only t=2 is valid!
        df = _create_synthetic_flow_df(start_time=0.0, duration_seconds=360.0, flow_step=10.0)
        windows = build_temporal_windows(df, window_size_sec=60.0)
        assert len(windows) == 6

        samples = build_forecasting_samples(windows, history_len=3, horizons=(1, 2, 3))
        assert len(samples) == 1
        assert samples[0].anchor_window_idx == 2
        assert samples[0].history_window_indices == (0, 1, 2)
        assert samples[0].target_window_indices == (3, 4, 5)

    def test_chronological_split_no_overlap(self):
        """Verify train/validation/test chronological segregation and zero overlap."""
        # 100 windows
        df = _create_synthetic_flow_df(start_time=0.0, duration_seconds=6000.0, flow_step=10.0)
        windows = build_temporal_windows(df, window_size_sec=60.0)
        assert len(windows) == 100

        samples = build_forecasting_samples(windows, history_len=3, horizons=(1, 2, 3))
        split = chronological_split(
            samples=samples,
            total_windows=len(windows),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            purge_gap_windows=2,
        )

        assert len(split.train_samples) > 0
        assert len(split.val_samples) > 0
        assert len(split.test_samples) > 0

        # Verify boundary ranges
        train_max_target = max(max(s.target_window_indices) for s in split.train_samples)
        val_min_hist = min(min(s.history_window_indices) for s in split.val_samples)
        val_max_target = max(max(s.target_window_indices) for s in split.val_samples)
        test_min_hist = min(min(s.history_window_indices) for s in split.test_samples)

        # Train target must precede Val history by at least purge gap
        assert train_max_target < split.val_window_range[0]
        assert train_max_target <= split.train_window_range[1]
        assert val_min_hist >= split.val_window_range[0]

        # Val target must precede Test history by at least purge gap
        assert val_max_target < split.test_window_range[0]
        assert val_max_target <= split.val_window_range[1]
        assert test_min_hist >= split.test_window_range[0]

        # Ensure no sample index appears in multiple partitions
        train_ids = {s.sample_idx for s in split.train_samples}
        val_ids = {s.sample_idx for s in split.val_samples}
        test_ids = {s.sample_idx for s in split.test_samples}
        assert train_ids.isdisjoint(val_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids.isdisjoint(test_ids)

    def test_attack_metadata_not_in_input_history(self):
        """Verify strict anti-leakage: history windows only expose window indexing."""
        df = _create_synthetic_flow_df(start_time=0.0, duration_seconds=600.0, flow_step=10.0)
        windows = build_temporal_windows(df, window_size_sec=60.0)
        samples = build_forecasting_samples(windows, history_len=3, horizons=(1, 2, 3))

        for s in samples:
            # history_window_indices must only contain integer indices
            assert all(isinstance(idx, int) for idx in s.history_window_indices)
            assert not hasattr(s, "is_attack")
            assert not hasattr(s, "attack_ratio")
            assert not hasattr(s, "raw_label")

            # Target metadata must be strictly in target_metadata dictionary
            assert "k1_attack_flows" in s.target_metadata
            assert "k2_attack_flows" in s.target_metadata
            assert "k3_attack_flows" in s.target_metadata

    def test_deterministic_repeated_execution(self):
        """Verify exact deterministic reproducibility across repeated runs."""
        df = _create_synthetic_flow_df(
            start_time=0.0,
            duration_seconds=1200.0,
            flow_step=10.0,
            attack_intervals=[(200.0, 300.0), (800.0, 950.0)],
        )

        windows1 = build_temporal_windows(df, window_size_sec=60.0)
        samples1 = build_forecasting_samples(windows1, history_len=3, horizons=(1, 2, 3))
        split1 = chronological_split(samples1, len(windows1), 0.6, 0.2, 0.2, 2)

        windows2 = build_temporal_windows(df, window_size_sec=60.0)
        samples2 = build_forecasting_samples(windows2, history_len=3, horizons=(1, 2, 3))
        split2 = chronological_split(samples2, len(windows2), 0.6, 0.2, 0.2, 2)

        assert [w.y for w in windows1] == [w.y for w in windows2]
        assert [s.Y for s in samples1] == [s.Y for s in samples2]
        assert len(split1.train_samples) == len(split2.train_samples)
        assert len(split1.val_samples) == len(split2.val_samples)
        assert len(split1.test_samples) == len(split2.test_samples)


if __name__ == "__main__":
    test_instance = TestTemporalTargets()
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

    print("=" * 60)
    print(f"RUNNING {len(test_methods)} UNIT TESTS FOR TEMPORAL TARGETS")
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
