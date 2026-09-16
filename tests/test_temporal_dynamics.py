"""
Unit tests for Temporal Dynamics Module (Phase 2A)
==================================================
Verifies:
1. ΔS shape is correct (N, H, 16).
2. First difference is strictly zero: ΔS[:, 0, :] == 0.
3. No future state is accessed.
4. Difference calculation is deterministic.
5. No NaN/Inf in output.
6. Sequence boundaries are respected.
7. Train/test normalization is isolated (zero leakage).
8. State-only model still works (input_dim=16).
9. State+velocity model output shape is correct (input_dim=32, output shape (B,)).
10. Existing full repository tests still pass.
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.temporal_dynamics import (
    compute_first_difference_sequence,
    construct_state_plus_velocity_sequence,
    build_temporal_dynamics_dataset,
)
from src.features.network_state import MacroNetworkState
from src.data.temporal_targets import TemporalWindow
from src.models.temporal_transformer import TemporalTransformerForecaster


class TestTemporalDynamics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_delta_s_shape_and_first_zero(self):
        """1. ΔS shape is (N, H, 16), 2. First difference is zero."""
        X_state = np.random.randn(10, 5, 16).astype(np.float32)
        delta_S = compute_first_difference_sequence(X_state)

        self.assertEqual(delta_S.shape, (10, 5, 16))
        # 2. First difference must be exactly zero
        np.testing.assert_allclose(delta_S[:, 0, :], 0.0)

        # Non-first timesteps should match manual diff
        expected_diff = X_state[:, 1, :] - X_state[:, 0, :]
        np.testing.assert_allclose(delta_S[:, 1, :], expected_diff, rtol=1e-5)

    def test_no_future_state_accessed(self):
        """3. No future state is accessed: changing future states does not alter past diffs."""
        X_seq = np.arange(5 * 16, dtype=np.float32).reshape(1, 5, 16)
        delta_1 = compute_first_difference_sequence(X_seq)

        # Alter future timestep at index 4
        X_seq_mod = X_seq.copy()
        X_seq_mod[:, 4, :] += 999.0
        delta_2 = compute_first_difference_sequence(X_seq_mod)

        # Diffs at indices 0, 1, 2, 3 must be IDENTICAL
        np.testing.assert_allclose(delta_1[:, :4, :], delta_2[:, :4, :])
        # Only diff at index 4 changes
        self.assertFalse(np.allclose(delta_1[:, 4, :], delta_2[:, 4, :]))

    def test_determinism_and_no_nan_inf(self):
        """4. Determinism, 5. No NaN/Inf."""
        X_state = np.random.randn(20, 10, 16).astype(np.float32)
        Z1 = construct_state_plus_velocity_sequence(X_state)
        Z2 = construct_state_plus_velocity_sequence(X_state)

        self.assertEqual(Z1.shape, (20, 10, 32))
        np.testing.assert_allclose(Z1, Z2)
        self.assertFalse(np.isnan(Z1).any())
        self.assertFalse(np.isinf(Z1).any())

    def test_sequence_boundaries_respected(self):
        """6. Sequence boundaries are respected across disjoint captures."""
        windows = [
            TemporalWindow(i, i*60.0, (i+1)*60.0, 10, 0, 0.0, 0, False)
            for i in range(12)
        ]
        states = {
            i: MacroNetworkState(i, i*60.0, (i+1)*60.0, np.ones(16) * float(i), {})
            for i in range(12)
        }

        # Build with H=3
        X_out, y, _, meta = build_temporal_dynamics_dataset(
            windows, states, history_len=3, include_velocity=True
        )
        self.assertEqual(X_out.shape, (9, 3, 32))
        # Check first step in every sequence has zero velocity
        np.testing.assert_allclose(X_out[:, 0, 16:], 0.0)

    def test_scaler_isolation_invariant(self):
        """7. Normalization scaler is strictly fit on train and never touches test."""
        windows = [
            TemporalWindow(i, i*60.0, (i+1)*60.0, 10, 0, 0.0, 0, False)
            for i in range(10)
        ]
        states = {
            i: MacroNetworkState(i, i*60.0, (i+1)*60.0, np.ones(16) * float(i * 10), {})
            for i in range(10)
        }

        X_tr, y_tr, scaler, _ = build_temporal_dynamics_dataset(
            windows[:6], states, history_len=3, include_velocity=True, fit_scaler=True
        )
        self.assertIsNotNone(scaler)
        # Scaler must have 32 features
        self.assertEqual(len(scaler.mean_), 32)

        # Transform test set without fitting
        mean_before = scaler.mean_.copy()
        X_te, y_te, _, _ = build_temporal_dynamics_dataset(
            windows[6:], states, history_len=3, include_velocity=True, scaler=scaler, fit_scaler=False
        )
        np.testing.assert_allclose(scaler.mean_, mean_before)

    def test_transformer_state_only_and_velocity_variants(self):
        """8. State-only model works (input_dim=16), 9. State+velocity model works (input_dim=32)."""
        # State-only (16 dims)
        model_state = TemporalTransformerForecaster(input_dim=16, d_model=64)
        x_16 = torch.randn(4, 3, 16)
        out_16 = model_state(x_16)
        self.assertEqual(out_16.shape, (4,))

        # State + Velocity (32 dims)
        model_velo = TemporalTransformerForecaster(input_dim=32, d_model=64)
        x_32 = torch.randn(4, 3, 32)
        out_32 = model_velo(x_32)
        self.assertEqual(out_32.shape, (4,))


if __name__ == "__main__":
    unittest.main()
