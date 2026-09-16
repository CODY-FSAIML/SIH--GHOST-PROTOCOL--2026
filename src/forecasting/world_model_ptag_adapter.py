"""Adapter from the learned latent world model to PTAG and trajectory search.

The trained world model is deterministic. By default this adapter creates one
future candidate per step. Optional latent perturbations create controlled demo
candidates around the model prediction; they are candidate-generation samples,
not learned uncertainty estimates or ground-truth attack paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .learned_world_model import LearnedLatentWorldModel
from .ptag import PTAGNode, PTAGEdge, ProbabilisticTemporalAttackGraph
from .trajectory_search import Trajectory, search_top_k_trajectories
from .world_model import TransitionCandidate


@dataclass(frozen=True)
class WorldModelPTAGResult:
    current_state: np.ndarray
    future_states: np.ndarray
    attack_risk_scores: np.ndarray
    latent_states: np.ndarray
    graph: ProbabilisticTemporalAttackGraph
    trajectories: List[Trajectory]
    metadata: Dict[str, Any]


class LearnedWorldModelPTAGAdapter:
    """Convert learned continuous rollouts into model-driven PTAG candidates."""

    def __init__(
        self,
        model: LearnedLatentWorldModel,
        candidate_count: int = 1,
        latent_noise_std: float = 0.0,
        seed: int = 42,
    ):
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if candidate_count > 1 and latent_noise_std <= 0.0:
            raise ValueError("Multiple candidates require an explicit positive latent_noise_std")
        self.model = model.eval()
        self.candidate_count = candidate_count
        self.latent_noise_std = float(latent_noise_std)
        self.seed = seed
        self.mode = "learned_model_driven"

    def _candidate_outputs(self, history: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        with torch.no_grad():
            latent = self.model.encode_history(history)
            next_latent = self.model.predict_next_latent(latent)
            outputs = []
            generator = torch.Generator(device=next_latent.device).manual_seed(self.seed)
            for candidate_index in range(self.candidate_count):
                candidate_latent = next_latent
                if candidate_index > 0:
                    noise = torch.randn(
                        candidate_latent.shape,
                        generator=generator,
                        device=candidate_latent.device,
                        dtype=candidate_latent.dtype,
                    ) * self.latent_noise_std
                    candidate_latent = candidate_latent + noise
                predicted_state, risk_logit = self.model.decode_next(candidate_latent)
                outputs.append({
                    "latent": candidate_latent,
                    "predicted_state": predicted_state,
                    "risk_score": torch.sigmoid(risk_logit),
                })
            return outputs

    def build(
        self,
        history: torch.Tensor | np.ndarray,
        horizon: int = 3,
        top_k: int = 3,
        beam_width: int = 8,
        min_score: float = 0.0,
    ) -> WorldModelPTAGResult:
        """Generate future candidates, PTAG edges, and ranked trajectories."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        history_tensor = torch.as_tensor(history, dtype=torch.float32)
        if history_tensor.dim() == 2:
            history_tensor = history_tensor.unsqueeze(0)
        if history_tensor.shape[0] != 1:
            raise ValueError("Adapter inference accepts one observed history at a time")
        if history_tensor.shape[1] != self.model.config.history_len:
            raise ValueError("Observed history length does not match model configuration")

        graph = ProbabilisticTemporalAttackGraph()
        root = PTAGNode("t0:observed", "observed_current_state", 0, {"mode": self.mode})
        graph.add_node(root)
        current_history = history_tensor.clone()
        all_states: List[np.ndarray] = []
        all_risks: List[float] = []
        all_latents: List[np.ndarray] = []
        frontier = [root]
        for step in range(1, horizon + 1):
            next_frontier = []
            outputs = self._candidate_outputs(current_history)
            for source in frontier:
                scores = [float(output["risk_score"].item()) for output in outputs]
                weights = np.asarray(scores, dtype=np.float64)
                weights = np.maximum(weights, 1e-12)
                weights /= weights.sum()
                candidates = []
                for candidate_index, (output, score, weight) in enumerate(zip(outputs, scores, weights)):
                    state = output["predicted_state"][0].detach().cpu().numpy().astype(np.float32)
                    latent = output["latent"][0].detach().cpu().numpy().astype(np.float32)
                    state_name = f"step{step}_candidate{candidate_index}_{source.node_id}"
                    candidates.append(TransitionCandidate(
                        state=state_name,
                        probability=float(weight),
                        score=float(score),
                        metadata={
                            "mode": self.mode,
                            "candidate_generation": "deterministic_model_output" if candidate_index == 0 else "controlled_latent_perturbation",
                            "timestep": step,
                            "source_state": source.state,
                            "destination_state": state.tolist(),
                            "latent_state": latent.tolist(),
                            "attack_risk_score": score,
                            "score_semantics": "predicted attack-risk score; calibration not established",
                            "normalized_weight_semantics": "normalized candidate weight; not a calibrated probability",
                        },
                    ))
                edges = graph.add_transitions(source, candidates, step)
                for edge in edges:
                    next_frontier.append(graph.nodes[edge.target])
            # A rollout history is the primary deterministic branch. For optional
            # candidates, each branch gets its own predicted state as next context.
            primary = outputs[0]["predicted_state"].unsqueeze(1)
            current_history = torch.cat([current_history[:, 1:, :], primary], dim=1)
            frontier = next_frontier
            all_states.append(outputs[0]["predicted_state"][0].detach().cpu().numpy())
            all_risks.append(float(outputs[0]["risk_score"].item()))
            all_latents.append(outputs[0]["latent"][0].detach().cpu().numpy())

        trajectories = search_top_k_trajectories(
            graph,
            start_node=root.node_id,
            horizon=horizon,
            beam_width=beam_width,
            top_k=top_k,
            min_probability=min_score,
            score_mode="model_score",
        )
        return WorldModelPTAGResult(
            current_state=history_tensor[0, -1].detach().cpu().numpy(),
            future_states=np.asarray(all_states, dtype=np.float32),
            attack_risk_scores=np.asarray(all_risks, dtype=np.float32),
            latent_states=np.asarray(all_latents, dtype=np.float32),
            graph=graph,
            trajectories=trajectories,
            metadata={
                "mode": self.mode,
                "candidate_count": self.candidate_count,
                "latent_noise_std": self.latent_noise_std,
                "risk_score_semantics": "predicted attack-risk score; calibration not established",
                "path_score_formula": "sum(log(max(edge.score, 1e-12)))",
                "trajectory_score_semantics": "model-derived trajectory score; not a probability",
                "horizon": horizon,
            },
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs):
        model, _, _ = LearnedLatentWorldModel.from_checkpoint(checkpoint_path)
        return cls(model, **kwargs)
