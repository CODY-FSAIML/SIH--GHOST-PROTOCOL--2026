#!/usr/bin/env python
"""Demonstrate model-signal-conditioned PTAG trajectory search."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_temporal_dynamics_ablation import SCENARIO_50_PATH, load_raw_capture  # noqa: E402
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.forecasting.ptag import build_dynamic_ptag  # noqa: E402
from src.forecasting.trajectory_search import search_top_k_trajectories  # noqa: E402
from src.forecasting.world_model import DemoTransitionModel, ForecastSignal  # noqa: E402
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    metadata = json.loads((models_dir / "temporal_h10_state_velocity_metadata.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        models_dir / "temporal_h10_state_velocity_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    scaler = joblib.load(models_dir / "temporal_h10_state_velocity_scaler.pkl")
    model = TemporalTransformerForecaster(**metadata["architecture_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    capture = load_raw_capture(SCENARIO_50_PATH, "Scenario 50 trajectory demo")
    X, _, _, sample_meta = build_temporal_dynamics_dataset(
        capture["windows"],
        capture["states_lookup"],
        history_len=10,
        include_velocity=True,
        scaler=scaler,
        fit_scaler=False,
    )
    if len(X) == 0:
        raise RuntimeError("Scenario 50 produced no forecastable H10 samples")

    with torch.no_grad():
        logit = model(torch.as_tensor(X[:1], dtype=torch.float32))
        forecast_score = float(torch.sigmoid(logit)[0].item())

    current_meta = sample_meta[0]
    current_state_vector = X[0, -1, :16]
    signal = ForecastSignal(
        score=forecast_score,
        source="verified temporal H10 State + Velocity checkpoint",
        features={
            "anchor_window_idx": current_meta["anchor_window_idx"],
            "target_window_idx": current_meta["target_window_idx"],
        },
    )
    transition_model = DemoTransitionModel()
    graph = build_dynamic_ptag(
        current_state="observed_current_state",
        signal=signal,
        transition_model=transition_model,
        horizon=3,
    )
    trajectories = search_top_k_trajectories(
        graph,
        horizon=3,
        beam_width=8,
        top_k=3,
        min_probability=0.001,
    )

    print("Current State")
    print(f"  Scenario 50 window: {current_meta['anchor_window_idx']}")
    print(f"  Forecast target: window {current_meta['target_window_idx']}")
    print(f"  Current state feature vector (16 values): {np.array2string(current_state_vector, precision=3)}")
    print(f"  Model-derived sigmoid score: {forecast_score:.6f}")
    print("  Score semantics: not claimed to be calibrated")
    print("  Transition mode: demo/simulation (not learned)")

    print("\nCandidate Future States")
    candidates = transition_model.predict_next("observed_current_state", signal)
    for candidate in candidates:
        print(f"  {candidate.state}: transition_probability={candidate.probability:.6f}")

    print("\nTop-K probable trajectories")
    for trajectory in trajectories:
        print(
            f"  rank={trajectory.rank} path={' -> '.join(trajectory.path)} "
            f"states={' -> '.join(trajectory.states)} "
            f"probability={trajectory.cumulative_probability:.8f} "
            f"log_probability={trajectory.cumulative_log_probability:.8f}"
        )
    print(f"\nPTAG nodes: {len(graph.nodes)}; PTAG edges: {sum(len(edges) for edges in graph.edges.values())}")
    print(f"Ranked trajectories returned: {len(trajectories)}")


if __name__ == "__main__":
    main()
