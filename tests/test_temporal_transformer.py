"""
Unit tests for Temporal Transformer Forecaster (Phase 1)
=======================================================
Verifies:
1. Input shape = (B, H, 16)
2. Output shape = (B,)
3. Model accepts H=3
4. Model accepts H=10
5. No NaN/Inf in output
6. Sigmoid is NOT applied inside model output (logits range unconstrained)
7. Sequence never crosses capture boundaries
8. No future window appears in a history sequence
9. pos_weight is calculated only from training labels
10. Deterministic seed produces reproducible results
11. Gradients propagate
12. Optimizer step changes trainable parameters
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTemporalTransformer(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(42)
        np.random.seed(42)

    def test_input_output_shapes_h10(self):
        """1. Input shape = (B, 10, 16), 2. Output shape = (B,)."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=64, nhead=4, num_layers=2)
        x = self.torch.randn(8, 10, 16)
        out = model(x)
        self.assertEqual(out.shape, (8,))

    def test_model_accepts_h3(self):
        """3. Model accepts H=3."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=64, nhead=4, num_layers=2)
        x = self.torch.randn(4, 3, 16)
        out = model(x)
        self.assertEqual(out.shape, (4,))

    def test_model_accepts_h10(self):
        """4. Model accepts H=10."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=64, nhead=4, num_layers=2)
        x = self.torch.randn(4, 10, 16)
        out = model(x)
        self.assertEqual(out.shape, (4,))

    def test_no_nan_inf_in_output(self):
        """5. No NaN/Inf in output."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=64, nhead=4, num_layers=2)
        x = self.torch.randn(16, 10, 16)
        out = model(x)
        self.assertFalse(self.torch.isnan(out).any())
        self.assertFalse(self.torch.isinf(out).any())

    def test_sigmoid_not_applied_inside_model(self):
        """6. Sigmoid is not applied inside model output (logits can be <0 or >1)."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=64, nhead=4, num_layers=2)
        x = self.torch.randn(32, 10, 16) * 10.0
        out = model(x)
        # Verify logits span beyond [0, 1]
        has_outside_unit_interval = (out < 0.0).any() or (out > 1.0).any()
        self.assertTrue(has_outside_unit_interval)

    def test_sequence_never_crosses_capture_boundaries(self):
        """7. Sequences are constructed independently per capture."""
        from src.data.temporal_targets import TemporalWindow
        from src.features.network_state import MacroNetworkState
        from src.models.temporal_dataset import build_temporal_sequence_dataset

        # Create two small disjoint captures
        windows_A = [
            TemporalWindow(i, i*60.0, (i+1)*60.0, 10, 0, 0.0, 0, False)
            for i in range(12)
        ]
        states_A = {
            i: MacroNetworkState(i, i*60.0, (i+1)*60.0, np.ones(16) * 1.0, {})
            for i in range(12)
        }

        windows_B = [
            TemporalWindow(i, i*60.0, (i+1)*60.0, 10, 1, 0.1, 1, False)
            for i in range(12)
        ]
        states_B = {
            i: MacroNetworkState(i, i*60.0, (i+1)*60.0, np.ones(16) * 2.0, {})
            for i in range(12)
        }

        XA, yA, _, metaA = build_temporal_sequence_dataset(windows_A, states_A, history_len=10)
        XB, yB, _, metaB = build_temporal_sequence_dataset(windows_B, states_B, history_len=10)

        self.assertEqual(len(XA), 2)  # anchors 9, 10
        self.assertEqual(len(XB), 2)
        # All windows in A have value 1.0, in B have value 2.0
        np.testing.assert_allclose(XA, 1.0)
        np.testing.assert_allclose(XB, 2.0)

    def test_no_future_window_appears_in_history(self):
        """8. No future window (>= t+1) appears in history sequence."""
        from src.data.temporal_targets import TemporalWindow
        from src.features.network_state import MacroNetworkState
        from src.models.temporal_dataset import build_temporal_sequence_dataset

        windows = [
            TemporalWindow(i, i*60.0, (i+1)*60.0, 10, 0, 0.0, 0, False)
            for i in range(15)
        ]
        states = {
            i: MacroNetworkState(i, i*60.0, (i+1)*60.0, np.zeros(16), {})
            for i in range(15)
        }

        _, _, _, meta = build_temporal_sequence_dataset(windows, states, history_len=10)
        for m in meta:
            t = m["anchor_window_idx"]
            target_idx = m["target_window_idx"]
            hist = m["history_window_indices"]
            self.assertEqual(target_idx, t + 1)
            self.assertTrue(all(h <= t for h in hist))
            self.assertNotIn(target_idx, hist)

    def test_pos_weight_calculated_only_from_training_labels(self):
        """9. pos_weight = num_negative / num_positive strictly on training data."""
        from src.models.temporal_dataset import compute_pos_weight
        y_train = np.array([0, 0, 0, 1])
        pw = compute_pos_weight(y_train)
        self.assertAlmostEqual(pw, 3.0 / 1.0)

    def test_deterministic_seed_produces_reproducible_results(self):
        """10. Deterministic seed produces identical forward pass."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        self.torch.manual_seed(123)
        m1 = TemporalTransformerForecaster(input_dim=16, d_model=32, nhead=2, num_layers=1)
        m1.eval()

        self.torch.manual_seed(123)
        m2 = TemporalTransformerForecaster(input_dim=16, d_model=32, nhead=2, num_layers=1)
        m2.eval()

        self.torch.manual_seed(42)
        x = self.torch.randn(2, 5, 16)
        with self.torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)

        self.assertTrue(self.torch.allclose(out1, out2))

    def test_gradients_propagate_and_optimizer_step(self):
        """11. Gradients propagate and 12. Optimizer step changes trainable parameters."""
        from src.models.temporal_transformer import TemporalTransformerForecaster
        model = TemporalTransformerForecaster(input_dim=16, d_model=32, nhead=2, num_layers=1)
        optimizer = self.torch.optim.Adam(model.parameters(), lr=1e-2)

        x = self.torch.randn(4, 5, 16)
        y = self.torch.tensor([1.0, 0.0, 1.0, 0.0])

        initial_params = [p.clone().detach() for p in model.parameters()]

        optimizer.zero_grad()
        logits = model(x)
        loss = self.torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        # 11. Verify gradients exist and are non-zero
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(self.torch.isnan(p.grad).any())

        optimizer.step()

        # 12. Verify parameters changed
        changed = any(
            not self.torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
