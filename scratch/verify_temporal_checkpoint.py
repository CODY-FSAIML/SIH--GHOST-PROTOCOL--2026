#!/usr/bin/env python
"""Verify the reusable Transformer H10 State + Velocity checkpoint."""

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
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    checkpoint_path = models_dir / "temporal_h10_state_velocity_best.pt"
    scaler_path = models_dir / "temporal_h10_state_velocity_scaler.pkl"
    metadata_path = models_dir / "temporal_h10_state_velocity_metadata.json"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["architecture_parameters"]

    model = TemporalTransformerForecaster(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = metadata["parameter_count"]
    if parameter_count != expected_parameter_count:
        raise RuntimeError(f"Parameter count mismatch: {parameter_count} != {expected_parameter_count}")

    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50 verification")
    X50, _, _, sample_meta = build_temporal_dynamics_dataset(
        cap50["windows"],
        cap50["states_lookup"],
        history_len=10,
        include_velocity=True,
        scaler=scaler,
        fit_scaler=False,
    )
    if len(X50) < 5:
        raise RuntimeError(f"Expected at least 5 Scenario 50 samples, got {len(X50)}")
    if not np.isfinite(X50).all():
        raise RuntimeError("Scenario 50 inference features contain NaN or Inf values")

    sample_indices = np.linspace(0, len(X50) - 1, 5, dtype=int)
    with torch.no_grad():
        logits = model(torch.as_tensor(X50[sample_indices], dtype=torch.float32))
        scores = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("Checkpoint scores contain NaN or Inf values")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Parameter count: {parameter_count}")
    print(f"Scenario 50 samples: {len(X50)}")
    print("Representative Scenario 50 scores (sigmoid scores, not calibrated probabilities):")
    for index, score in zip(sample_indices, scores):
        meta = sample_meta[index]
        print(
            f"  sample={index} anchor_window={meta['anchor_window_idx']} "
            f"target_window={meta['target_window_idx']} score={score:.8f}"
        )
    print("Finite-score check: PASS")
    print("Checkpoint verification: PASS")


if __name__ == "__main__":
    main()
