#!/usr/bin/env python
"""Create the verified Transformer H10 State + Velocity checkpoint."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_temporal_dynamics_ablation import (  # noqa: E402
    SCENARIO_42_PATH,
    SCENARIO_50_PATH,
    load_raw_capture,
    split_capture_windows,
    train_model,
)
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402


MODEL_CONFIG = {
    "input_dim": 32,
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "max_len": 100,
}
TRAINING_CONFIG = {
    "seed": 42,
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 7,
    "optimizer": "AdamW",
    "loss": "BCEWithLogitsLoss",
    "pos_weight_source": "Scenario 42 training labels only",
}


def main() -> None:
    started = time.perf_counter()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "temporal_h10_state_velocity_best.pt"
    scaler_path = models_dir / "temporal_h10_state_velocity_scaler.pkl"
    metadata_path = models_dir / "temporal_h10_state_velocity_metadata.json"

    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")
    w42_train, w42_val, _ = split_capture_windows(
        cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2
    )

    X_train, y_train, scaler, _ = build_temporal_dynamics_dataset(
        w42_train,
        cap42["states_lookup"],
        history_len=10,
        include_velocity=True,
        fit_scaler=True,
    )
    X_val, y_val, _, _ = build_temporal_dynamics_dataset(
        w42_val,
        cap42["states_lookup"],
        history_len=10,
        include_velocity=True,
        scaler=scaler,
        fit_scaler=False,
    )
    X50, y50, _, _ = build_temporal_dynamics_dataset(
        cap50["windows"],
        cap50["states_lookup"],
        history_len=10,
        include_velocity=True,
        scaler=scaler,
        fit_scaler=False,
    )

    if X_train.shape[2] != MODEL_CONFIG["input_dim"]:
        raise ValueError(f"Unexpected training feature dimension: {X_train.shape}")
    if len(scaler.mean_) != MODEL_CONFIG["input_dim"]:
        raise ValueError("Scaler does not contain the expected 32 features.")

    model, train_meta = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=MODEL_CONFIG["input_dim"],
        epochs=TRAINING_CONFIG["epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        patience=TRAINING_CONFIG["early_stopping_patience"],
        seed=TRAINING_CONFIG["seed"],
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "model_config": MODEL_CONFIG,
        "parameter_count": parameter_count,
        "history_len": 10,
        "feature_definition": "[16 NetworkState features, 16 first-difference velocity features]",
        "training_metadata": train_meta,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": "Transformer H10 + State + Velocity",
        "history_length": 10,
        "input_feature_dimension": 16,
        "velocity_feature_dimension": 16,
        "total_input_dimension": 32,
        "architecture_parameters": MODEL_CONFIG,
        "parameter_count": parameter_count,
        "training_dataset": {
            "capture": "CTU-13 Scenario 42",
            "split": "chronological training windows with purge_gap=2",
            "sample_count": int(len(y_train)),
            "positive_count": int(np.sum(y_train == 1)),
            "negative_count": int(np.sum(y_train == 0)),
        },
        "validation_dataset": {
            "capture": "CTU-13 Scenario 42",
            "split": "chronological validation windows with purge_gap=2",
            "sample_count": int(len(y_val)),
            "positive_count": int(np.sum(y_val == 1)),
            "negative_count": int(np.sum(y_val == 0)),
        },
        "evaluation_dataset": {
            "capture": "CTU-13 Scenario 50",
            "split": "completely held-out full timeline",
            "sample_count": int(len(y50)),
        },
        "seed": 42,
        "training_hyperparameters": {
            **TRAINING_CONFIG,
            "pos_weight": float(train_meta["pos_weight"]),
            "epochs_trained": int(train_meta["epochs"]),
            "best_validation_loss": float(train_meta["best_val_loss"]),
        },
        "scaler_description": {
            "class": "sklearn.preprocessing.StandardScaler",
            "fit_data": "Scenario 42 training State + Velocity sequences only",
            "feature_count": 32,
            "test_data_used_for_fit": False,
        },
        "checkpoint_creation_details": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)),
            "source_configuration": "scratch/run_temporal_dynamics_ablation.py H10 State + Velocity branch",
            "torch_version": torch.__version__,
            "creation_runtime_seconds": round(time.perf_counter() - started, 3),
            "metrics_included": ["best_validation_loss"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Parameter count: {parameter_count}")
    print(f"Train samples: {len(y_train)}; validation samples: {len(y_val)}; Scenario 50 samples: {len(y50)}")
    print(f"Epochs trained: {train_meta['epochs']}")
    print(f"Best validation loss: {train_meta['best_val_loss']:.8f}")


if __name__ == "__main__":
    main()
