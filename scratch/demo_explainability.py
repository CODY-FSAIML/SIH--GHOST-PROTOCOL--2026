#!/usr/bin/env python
"""Smoke demo for observable feature explanations and ATT&CK context."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.run_temporal_dynamics_ablation import SCENARIO_50_PATH, load_raw_capture  # noqa: E402
from src.explainability.feature_explanations import rank_driving_indicators  # noqa: E402
from src.explainability.mitre_mapping import explain_observation  # noqa: E402
from src.features.network_state import MACRO_FEATURE_NAMES  # noqa: E402


def main():
    capture = load_raw_capture(SCENARIO_50_PATH, "Scenario 50 explainability demo")
    states = [capture["states_lookup"][index].features for index in sorted(capture["states_lookup"])]
    current = states[-1]
    history = states[-6:-1]
    if not history:
        raise RuntimeError("Scenario 50 did not produce enough state history")
    velocity = {name: float(current.get(name, 0.0) - history[-1].get(name, 0.0)) for name in MACRO_FEATURE_NAMES}
    interpretation = explain_observation(current, history, velocity=velocity, top_k=5)
    print("Predicted behavior:")
    print(f"  {interpretation['behavior']}")
    print("ATT&CK-aligned tactic:")
    print(f"  {interpretation['tactic'] or 'Insufficient evidence'}")
    print("ATT&CK-aligned technique:")
    print(f"  {interpretation['technique'] or 'Insufficient evidence'}")
    print("Evidence score:")
    print(f"  {interpretation['evidence_score']:.3f} (heuristic evidence, not calibrated confidence)")
    print("Supporting indicators:")
    for indicator in interpretation["supporting_indicators"]:
        print(f"  - {indicator}")
    print("Driving indicators:")
    for item in interpretation["driving_indicators"]:
        print(f"  - {item['feature']}: {item['direction']} ({item['change']:.4f}), importance={item['importance']:.4f}")
    print("Explanation:")
    print(f"  {interpretation['explanation']}")
    assert np.isfinite(interpretation["evidence_score"])
    assert interpretation["driving_indicators"]
    print("EXPLAINABILITY_SMOKE=PASS")


if __name__ == "__main__":
    main()
