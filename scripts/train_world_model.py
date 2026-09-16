#!/usr/bin/env python
"""Train and save the learned latent world model.

Primary setup: Scenario 42 chronological train/validation, Scenario 50 held out.
No test samples are used for fitting the scaler or optimizing the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_temporal_dynamics_ablation import (  # noqa: E402
    SCENARIO_42_PATH,
    SCENARIO_50_PATH,
    load_raw_capture,
    split_capture_windows,
)
from src.forecasting.learned_world_model import (  # noqa: E402
    LearnedLatentWorldModel,
    LearnedWorldModelConfig,
    build_transition_dataset,
)
from src.models.temporal_dataset import compute_pos_weight  # noqa: E402


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    cap42 = load_raw_capture(SCENARIO_42_PATH, "Scenario 42")
    cap50 = load_raw_capture(SCENARIO_50_PATH, "Scenario 50")
    train_windows, val_windows, _ = split_capture_windows(cap42["windows"], train_ratio=0.6, val_ratio=0.2, purge_gap=2)

    X_train, Y_train, y_train, scaler, train_meta = build_transition_dataset(
        train_windows, cap42["states_lookup"], history_len=args.history_len, fit_scaler=True
    )
    X_val, Y_val, y_val, _, val_meta = build_transition_dataset(
        val_windows, cap42["states_lookup"], history_len=args.history_len, scaler=scaler
    )
    X_test, Y_test, y_test, _, test_meta = build_transition_dataset(
        cap50["windows"], cap50["states_lookup"], history_len=args.history_len, scaler=scaler
    )

    config = LearnedWorldModelConfig(
        history_len=args.history_len,
        latent_dim=args.latent_dim,
        transition_hidden_dim=args.transition_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
    )
    model = LearnedLatentWorldModel.from_temporal_checkpoint(args.temporal_checkpoint, config=config)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(y_train.astype(np.float32))), batch_size=args.batch_size, shuffle=False)
    val_tensors = (torch.from_numpy(X_val), torch.from_numpy(Y_val), torch.from_numpy(y_val.astype(np.float32)))
    pos_weight = compute_pos_weight(y_train)
    risk_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for X_batch, Y_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            state_loss = nn.functional.smooth_l1_loss(output["predicted_state"], Y_batch)
            attack_loss = risk_loss(output["attack_risk_logit"], y_batch)
            loss = state_loss + args.lambda_risk * attack_loss
            loss.backward()
            optimizer.step()
            train_losses.append((float(loss.item()), float(state_loss.item()), float(attack_loss.item())))
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensors[0])
            val_state_loss = nn.functional.smooth_l1_loss(val_output["predicted_state"], val_tensors[1])
            val_attack_loss = risk_loss(val_output["attack_risk_logit"], val_tensors[2])
            val_total = val_state_loss + args.lambda_risk * val_attack_loss
        mean_train = float(np.mean([item[0] for item in train_losses]))
        history.append({"epoch": epoch, "train_loss": mean_train, "val_loss": float(val_total.item())})
        print(f"Epoch {epoch:02d} | train_loss={mean_train:.6f} | val_loss={val_total.item():.6f}")
        if val_total.item() < best_loss:
            best_loss = float(val_total.item())
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val loss {best_loss:.6f})")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint state was selected.")
    model.load_state_dict(best_state)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "history_len": args.history_len,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "held_out_scenario50_samples": len(y_test),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "pos_weight": float(pos_weight),
        "lambda_risk": args.lambda_risk,
        "epochs_requested": args.epochs,
        "epochs_trained": epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "losses": {"state": "SmoothL1Loss", "risk": "BCEWithLogitsLoss"},
        "scaler_fit": "Scenario 42 training sequences only",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "training_runtime_seconds": time.perf_counter() - started,
        "risk_score_semantics": model.risk_score_semantics,
        "evaluation_metrics": "not computed by this training script",
    }
    payload = model.checkpoint_payload(scaler, metadata)
    payload["temporal_checkpoint"] = str(Path(args.temporal_checkpoint))
    torch.save(payload, output_path)
    print(f"Saved checkpoint: {output_path}")
    print(json.dumps({"parameter_count": metadata["model_parameter_count"], "train_samples": len(y_train), "validation_samples": len(y_val), "scenario50_samples": len(y_test), "epochs_trained": epoch, "runtime_seconds": metadata["training_runtime_seconds"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "latent_world_model_h10_best.pt"))
    parser.add_argument("--temporal-checkpoint", default=str(PROJECT_ROOT / "models" / "temporal_h10_state_velocity_best.pt"))
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--transition-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
