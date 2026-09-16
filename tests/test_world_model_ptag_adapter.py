import numpy as np
import torch

from src.forecasting.learned_world_model import LearnedLatentWorldModel, LearnedWorldModelConfig
from src.forecasting.world_model_ptag_adapter import LearnedWorldModelPTAGAdapter


def _adapter(candidate_count=1, noise=0.0):
    torch.manual_seed(42)
    model = LearnedLatentWorldModel(LearnedWorldModelConfig(
        history_len=4, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, latent_dim=12, transition_hidden_dim=20,
        decoder_hidden_dim=20, max_len=10,
    ))
    return LearnedWorldModelPTAGAdapter(model, candidate_count, noise, seed=42)


def test_adapter_initialization_and_one_step_conversion():
    adapter = _adapter()
    result = adapter.build(torch.zeros(4, 32), horizon=1, top_k=1, beam_width=2)
    assert result.metadata["mode"] == "learned_model_driven"
    assert result.future_states.shape == (1, 32)
    assert result.attack_risk_scores.shape == (1,)
    assert len(result.graph.nodes) == 2
    assert len(result.graph.outgoing("t0:observed")) == 1
    assert result.trajectories[0].metadata if hasattr(result.trajectories[0], "metadata") else True


def test_deterministic_finite_k_step_inference():
    history = np.zeros((4, 32), dtype=np.float32)
    first = _adapter().build(history, horizon=3, top_k=3, beam_width=4)
    second = _adapter().build(history, horizon=3, top_k=3, beam_width=4)
    np.testing.assert_allclose(first.future_states, second.future_states)
    np.testing.assert_allclose(first.attack_risk_scores, second.attack_risk_scores)
    assert np.isfinite(first.future_states).all()
    assert np.isfinite(first.attack_risk_scores).all()
    assert len(first.trajectories) == 1


def test_controlled_candidates_and_score_ordering():
    result = _adapter(candidate_count=3, noise=0.05).build(
        torch.zeros(4, 32), horizon=2, top_k=2, beam_width=2, min_score=0.0
    )
    assert result.metadata["candidate_count"] == 3
    assert len(result.graph.outgoing("t0:observed")) == 3
    assert 1 <= len(result.trajectories) <= 2
    assert all(result.trajectories[index].cumulative_log_probability >= result.trajectories[index + 1].cumulative_log_probability for index in range(len(result.trajectories) - 1))
    assert all(np.isfinite(item.cumulative_probability) for item in result.trajectories)


def test_multiple_candidates_require_explicit_noise():
    try:
        _adapter(candidate_count=2, noise=0.0)
    except ValueError:
        return
    raise AssertionError("Multiple candidates must require explicit latent noise")
