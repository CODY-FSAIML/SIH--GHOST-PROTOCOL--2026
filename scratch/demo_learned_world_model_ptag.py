#!/usr/bin/env python
"""Smoke demo for learned-world-model-driven PTAG trajectories."""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.run_temporal_dynamics_ablation import SCENARIO_50_PATH, load_raw_capture  # noqa: E402
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.forecasting.world_model_ptag_adapter import LearnedWorldModelPTAGAdapter  # noqa: E402


def main():
    capture = load_raw_capture(SCENARIO_50_PATH, "Scenario 50 learned-world-model demo")
    scaler = joblib.load(ROOT / "models" / "temporal_h10_state_velocity_scaler.pkl")
    X, _, _, _ = build_temporal_dynamics_dataset(
        capture["windows"], capture["states_lookup"], history_len=10,
        include_velocity=True, scaler=scaler, fit_scaler=False,
    )
    adapter = LearnedWorldModelPTAGAdapter.from_checkpoint(
        str(ROOT / "models" / "latent_world_model_h10_best.pt"),
        candidate_count=1,
        seed=42,
    )
    result = adapter.build(X[0], horizon=3, top_k=3, beam_width=4)
    print("Learned World Model -> PTAG")
    print(f"Mode: {result.metadata['mode']}")
    print(f"Future states: {result.future_states.shape}")
    print(f"Predicted attack-risk scores: {result.attack_risk_scores.tolist()}")
    print(f"PTAG nodes: {len(result.graph.nodes)}")
    print(f"PTAG edges: {sum(len(edges) for edges in result.graph.edges.values())}")
    print(f"Ranked trajectories: {len(result.trajectories)}")
    for trajectory in result.trajectories:
        print(
            f"  rank={trajectory.rank} path={' -> '.join(trajectory.path)} "
            f"model_score={trajectory.cumulative_probability:.8f}"
        )
    assert np.isfinite(result.future_states).all()
    assert np.isfinite(result.attack_risk_scores).all()
    assert result.trajectories
    print("LEARNED_WM_PTAG_SMOKE=PASS")


if __name__ == "__main__":
    main()
