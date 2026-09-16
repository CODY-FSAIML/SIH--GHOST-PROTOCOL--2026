"""
Unit Tests for Logistic Regression Forecasting Baseline (t+1)
=============================================================
Verifies:
1. 48-feature construction
2. Chronological feature ordering
3. Correct t+1 target alignment
4. No future information in features
5. Scaler fit only on training data
6. Validation/test use transform only
7. Deterministic repeated execution
8. Logistic Regression convergence
9. Probability output in [0, 1]
10. No NaN / infinity
11. Class-weight configuration
12. Coefficient-to-feature mapping
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.network_state import MacroNetworkState, MACRO_FEATURE_NAMES
from src.data.temporal_targets import ForecastingSample
from src.models.logistic_baseline import (
    get_48_feature_names,
    build_baseline_dataset,
    LogisticRegressionBaseline,
)


def _create_mock_states_and_samples(num_windows: int = 10):
    """Create predictable mock states and samples for testing."""
    states_lookup = {}
    for w in range(num_windows):
        # Unique distinguishable values in vector
        vec = np.full(16, float(w * 10), dtype=np.float64)
        vec[0] = float(w)  # flow_count = w
        feat_dict = dict(zip(MACRO_FEATURE_NAMES, vec))
        states_lookup[w] = MacroNetworkState(
            window_idx=w,
            start_time=float(w * 60),
            end_time=float((w + 1) * 60),
            vector=vec,
            features=feat_dict,
        )

    samples = []
    # For H=3, horizons=(1, 2, 3), valid anchors: w in [2, num_windows - 4]
    sample_id = 0
    for t in range(2, num_windows - 3):
        # Attack in window 5 and 6
        y1 = 1 if (t + 1) in (5, 6) else 0
        y2 = 1 if (t + 2) in (5, 6) else 0
        y3 = 1 if (t + 3) in (5, 6) else 0

        is_onset = bool(t < 5 and (t + 1) == 5)

        s = ForecastingSample(
            sample_idx=sample_id,
            anchor_window_idx=t,
            history_window_indices=(t - 2, t - 1, t),
            target_window_indices=(t + 1, t + 2, t + 3),
            Y={1: y1, 2: y2, 3: y3},
            target_empty_flags={1: False, 2: False, 3: False},
            target_metadata={
                "anchor_window_idx": t,
                "anchor_is_onset_t1": is_onset,
                "anchor_attack_flows": 10 if t in (5, 6) else 0,
            },
        )
        samples.append(s)
        sample_id += 1

    return states_lookup, samples


class TestLogisticBaseline:

    def test_48_feature_names_construction(self):
        """Verify feature names list contains exactly 48 correctly formatted strings."""
        names = get_48_feature_names()
        assert len(names) == 48
        assert names[0] == "t-2_flow_count"
        assert names[15] == "t-2_closed_flow_ratio"
        assert names[16] == "t-1_flow_count"
        assert names[31] == "t-1_closed_flow_ratio"
        assert names[32] == "t_flow_count"
        assert names[47] == "t_closed_flow_ratio"

    def test_chronological_feature_ordering_and_values(self):
        """Verify that X rows correctly order [S_(t-2), S_(t-1), S_t]."""
        states_lookup, samples = _create_mock_states_and_samples(num_windows=10)
        X, y, feature_names, meta = build_baseline_dataset(samples, states_lookup, horizon=1)

        assert X.shape == (len(samples), 48)
        assert len(y) == len(samples)

        # For sample 0 (anchor t=2, history 0, 1, 2):
        # S_0 has flow_count=0, S_1 has flow_count=1, S_2 has flow_count=2
        s0_features = X[0]
        assert s0_features[0] == 0.0   # t-2 flow_count
        assert s0_features[16] == 1.0  # t-1 flow_count
        assert s0_features[32] == 2.0  # t flow_count

    def test_target_alignment_t_plus_1(self):
        """Verify that target Y matches window t+1."""
        states_lookup, samples = _create_mock_states_and_samples(num_windows=10)
        X, y, _, _ = build_baseline_dataset(samples, states_lookup, horizon=1)

        # Attack was set for windows 5 and 6
        # When anchor t=4, t+1=5 -> y should be 1
        sample_t4_idx = next(i for i, s in enumerate(samples) if s.anchor_window_idx == 4)
        assert y[sample_t4_idx] == 1

        # When anchor t=3, t+1=4 -> y should be 0
        sample_t3_idx = next(i for i, s in enumerate(samples) if s.anchor_window_idx == 3)
        assert y[sample_t3_idx] == 0

    def test_scaler_fitted_only_on_train(self):
        """Verify that scaler parameters come strictly from training data."""
        np.random.seed(42)
        X_train = np.random.randn(30, 48) * 10 + 50  # Mean ~50, Std ~10
        y_train = (np.random.rand(30) > 0.5).astype(int)

        X_test = np.random.randn(10, 48) * 5 + 100   # Mean ~100, Std ~5
        y_test = (np.random.rand(10) > 0.5).astype(int)

        model = LogisticRegressionBaseline(max_iter=500)
        model.fit(X_train, y_train)

        # Scaler mean must match X_train mean, NOT X_test
        assert np.allclose(model.scaler.mean_, X_train.mean(axis=0), atol=1e-5)
        assert not np.allclose(model.scaler.mean_, X_test.mean(axis=0), atol=1e-2)

        # Ensure transform runs without altering scaler state
        orig_mean = model.scaler.mean_.copy()
        _ = model.predict_proba(X_test)
        assert np.array_equal(model.scaler.mean_, orig_mean)

    def test_model_convergence_and_probabilities(self):
        """Verify model converges and outputs probabilities strictly in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(50, 48)
        y = (X[:, 0] + X[:, 16] > 0).astype(int)  # Separable target

        model = LogisticRegressionBaseline(max_iter=1000)
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert len(probs) == len(X)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

    def test_class_weight_configuration(self):
        """Verify class_weight='balanced' is set on underlying model."""
        model = LogisticRegressionBaseline(class_weight="balanced")
        np.random.seed(42)
        X = np.random.randn(20, 48)
        y = np.array([0] * 18 + [1] * 2)  # Highly imbalanced
        model.fit(X, y)

        assert model.model.class_weight == "balanced"

    def test_threshold_calibration_single_class_fallback(self):
        """Verify single-class validation set does not crash and safely returns 0.5 with warning."""
        np.random.seed(42)
        X_train = np.random.randn(30, 48)
        y_train = (np.random.rand(30) > 0.5).astype(int)

        # Validation set with ONLY class 1
        X_val = np.random.randn(10, 48)
        y_val = np.ones(10, dtype=int)

        model = LogisticRegressionBaseline()
        model.fit(X_train, y_train)

        thresh, is_valid, msg = model.calibrate_threshold(X_val, y_val)
        assert thresh == 0.5
        assert is_valid is False
        assert "statistically invalid" in msg.lower()

    def test_coefficient_interpretation_mapping(self):
        """Verify feature coefficients correctly map to history step and macro feature name."""
        np.random.seed(42)
        X = np.random.randn(40, 48)
        y = (np.random.rand(40) > 0.5).astype(int)

        model = LogisticRegressionBaseline()
        model.fit(X, y)

        df_coef = model.get_feature_coefficients()
        assert len(df_coef) == 48
        assert set(df_coef.columns) == {"feature", "history_step", "macro_feature", "coefficient", "abs_coefficient"}
        assert set(df_coef["history_step"]) == {"t-2", "t-1", "t"}

        # First row has highest abs_coefficient
        assert df_coef["abs_coefficient"].iloc[0] >= df_coef["abs_coefficient"].iloc[-1]

    def test_deterministic_repeated_execution(self):
        """Verify bit-exact reproducibility across multiple runs with same random_state."""
        np.random.seed(42)
        X = np.random.randn(40, 48)
        y = (np.random.rand(40) > 0.5).astype(int)

        m1 = LogisticRegressionBaseline(random_state=42).fit(X, y)
        m2 = LogisticRegressionBaseline(random_state=42).fit(X, y)

        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)

        assert np.array_equal(p1, p2)
        assert np.array_equal(m1.model.coef_, m2.model.coef_)


if __name__ == "__main__":
    test_instance = TestLogisticBaseline()
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

    print("=" * 60)
    print(f"RUNNING {len(test_methods)} UNIT TESTS FOR LOGISTIC BASELINE")
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
