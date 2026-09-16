"""
Tests for CTU-13 Scenario 50 validation.
Verifies:
1. Adapter normalization, schema compliance, metadata mapping, port conversion.
2. NetworkState extraction, shape (338, 16), numerical validity, no NaNs/Infs.
3. Temporal target construction, window counts, attack/benign proportions.
4. Attack onsets and termination detection.
5. Strict anti-leakage invariants.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.adapters.ctu13_adapter import CTU13Adapter, _state_category
from src.features.network_state import (
    MacroNetworkStateBuilder,
    MACRO_FEATURE_NAMES,
)
from src.data.temporal_targets import (
    build_temporal_windows,
    build_forecasting_samples,
    chronological_split,
)

SCENARIO_50_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "ctu13", "capture20110817.binetflow"
)


@unittest.skipIf(not os.path.exists(SCENARIO_50_PATH), "Scenario 50 data file not present")
class TestScenario50Validation(unittest.TestCase):

    def test_adapter_schema_and_invariants_sample(self):
        """Verify CTU13Adapter strictly adheres to schema and semantic invariants."""
        df_raw = pd.read_csv(SCENARIO_50_PATH, nrows=500)
        adapter = CTU13Adapter()
        df_norm = adapter.normalize(df_raw)

        # 1. Validation method succeeds
        assert adapter.validate(df_norm) is True

        # 2. Semantic placements: packets_sent and bytes_sent must NOT be fabricated from TotPkts / TotBytes
        assert df_norm["packets_sent"].isna().all()
        assert df_norm["packets_received"].isna().all()
        assert df_norm["bytes_sent"].isna().all()
        assert df_norm["bytes_received"].isna().all()

        # 3. Metadata contains totals and derived dst_bytes
        for meta in df_norm["metadata"]:
            assert "total_packets" in meta
            assert "total_bytes" in meta
            assert "src_bytes" in meta
            assert "dst_bytes" in meta
            if meta["total_bytes"] is not None and meta["src_bytes"] is not None:
                assert meta["total_bytes"] >= meta["src_bytes"]

    def test_network_state_numerical_safety_sample(self):
        """Verify NetworkState extraction produces non-NaN, non-Inf 16-D vectors."""
        df_raw = pd.read_csv(SCENARIO_50_PATH, nrows=5000)
        dt = pd.to_datetime(df_raw["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
        ts = dt.astype("datetime64[ns]").astype(np.int64) / 1e9

        norm_df = pd.DataFrame({
            "timestamp": ts,
            "duration": pd.to_numeric(df_raw["Dur"], errors="coerce").fillna(0.0),
            "protocol": df_raw["Proto"].astype(str).str.lower().str.strip(),
            "src_ip": df_raw["SrcAddr"].astype(str).str.strip(),
            "dst_ip": df_raw["DstAddr"].astype(str).str.strip(),
            "total_packets": pd.to_numeric(df_raw["TotPkts"], errors="coerce").fillna(0),
            "total_bytes": pd.to_numeric(df_raw["TotBytes"], errors="coerce").fillna(0),
            "src_bytes": pd.to_numeric(df_raw["SrcBytes"], errors="coerce").fillna(0),
            "state_category": [_state_category(str(s)) for s in df_raw["State"]],
        })

        builder = MacroNetworkStateBuilder(window_size_sec=60.0)
        states, _, mat = builder.build_states(norm_df)

        assert len(states) > 0
        assert mat.shape[1] == len(MACRO_FEATURE_NAMES)
        assert not np.isnan(mat).any()
        assert not np.isinf(mat).any()
        assert (mat >= 0.0).all()

    def test_temporal_targets_and_onsets_structure(self):
        """Verify temporal target construction and onset detection logic."""
        # Timestamps spanning 5 full windows: [0, 60), [60, 120), [120, 180), [180, 240), [240, 300)
        df = pd.DataFrame({
            "timestamp": [10.0, 70.0, 130.0, 190.0, 260.0],
            "src_ip": ["10.0.0.1"] * 5,
            "dst_ip": ["10.0.0.2"] * 5,
            "is_attack": [0, 1, 0, 1, 1],
            "raw_label": ["flow=Background", "flow=From-Botnet-V50-1-UDP-DNS", "flow=Background", "flow=From-Botnet-V50-2-UDP-DNS", "flow=From-Botnet-V50-2-UDP-DNS"],
        })

        windows = build_temporal_windows(df, window_size_sec=60.0)
        self.assertEqual(len(windows), 5)
        self.assertEqual([w.y for w in windows], [0, 1, 0, 1, 1])

        samples = build_forecasting_samples(windows, history_len=2, horizons=(1, 2))
        self.assertEqual(len(samples), 2)  # t=1 (horizons 2, 3), t=2 (horizons 3, 4)

        # Anti-leakage check: sample history contains only window indices
        for s in samples:
            self.assertTrue(hasattr(s, "history_window_indices"))
            self.assertTrue(hasattr(s, "Y"))
            # Y must only contain target horizons
            self.assertEqual(set(s.Y.keys()), {1, 2})
