import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.forecasting.learned_world_model import (
    LearnedLatentWorldModel,
    LearnedWorldModelConfig,
)


def _model():
    torch.manual_seed(42)
    return LearnedLatentWorldModel(
        LearnedWorldModelConfig(
            history_len=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            latent_dim=12,
            transition_hidden_dim=20,
            decoder_hidden_dim=20,
            max_len=10,
        )
    )


def test_forward_shapes_and_finite_outputs():
    model = _model().eval()
    history = torch.randn(3, 4, 32)
    output = model(history)
    assert output["latent"].shape == (3, 12)
    assert output["next_latent"].shape == (3, 12)
    assert output["predicted_state"].shape == (3, 32)
    assert output["attack_risk_logit"].shape == (3,)
    assert all(torch.isfinite(value).all() for value in output.values() if torch.is_tensor(value))


def test_one_step_and_k_step_rollout():
    model = _model().eval()
    history = torch.randn(1, 4, 32)
    one_step = model(history)
    rollout = model.rollout(history, steps=3)
    assert one_step["predicted_state"].shape == (1, 32)
    assert rollout["predicted_states"].shape == (1, 3, 32)
    assert rollout["latent_states"].shape == (1, 3, 12)
    assert rollout["attack_risk_scores"].shape == (1, 3)
    assert torch.isfinite(rollout["predicted_states"]).all()
    assert torch.isfinite(rollout["attack_risk_scores"]).all()


def test_checkpoint_save_load_and_deterministic_inference():
    model = _model().eval()
    scaler = StandardScaler().fit(np.random.RandomState(42).randn(20, 32))
    payload = model.checkpoint_payload(scaler, {"seed": 42})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "world_model.pt"
        torch.save(payload, path)
        loaded, loaded_scaler, metadata = LearnedLatentWorldModel.from_checkpoint(path)
        loaded.eval()
        history = torch.randn(1, 4, 32)
        with torch.no_grad():
            first = model(history)["predicted_state"]
            second = loaded(history)["predicted_state"]
        assert torch.allclose(first, second)
        assert np.allclose(scaler.mean_, loaded_scaler.mean_)
        assert metadata["training_metadata"]["seed"] == 42
