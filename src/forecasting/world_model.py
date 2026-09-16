"""State-transition interfaces for future network-behavior forecasting.

The trainable latent implementation lives in ``learned_world_model.py``. The
string-state interface below remains for the PTAG demo transition layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol


@dataclass(frozen=True)
class ForecastSignal:
    """Model-derived signal used to condition a transition layer."""

    score: float
    source: str
    score_semantics: str = "model sigmoid score; calibration not established"
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionCandidate:
    """One possible next behavior state and its transition weight."""

    state: str
    probability: float
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StateTransitionModel(Protocol):
    mode: str

    def predict_next(
        self, current_state: str, signal: ForecastSignal
    ) -> List[TransitionCandidate]:
        """Return normalized candidate transitions for one future step."""


class LearnedTransitionModel:
    """String-state interface placeholder; not a trained transition model.

    Use :class:`LearnedLatentWorldModel` for learned continuous state
    transitions and predicted attack-risk scores.
    """

    mode = "learned_interface_unavailable"

    def predict_next(self, current_state: str, signal: ForecastSignal) -> List[TransitionCandidate]:
        raise NotImplementedError(
            "No learned transition model is available in this project. "
            "Use DemoTransitionModel explicitly for demonstration only."
        )


class DemoTransitionModel:
    """Deterministic simulation transition layer, not a learned model.

    The forecast score controls relative weights among generic network-behavior
    states. The resulting probabilities are demo/simulation values and are not
    calibrated probabilities or trained transition estimates.
    """

    mode = "demo"

    def __init__(self, state_names: List[str] | None = None):
        self.state_names = state_names or [
            "stable_activity",
            "elevated_activity",
            "anomalous_activity",
            "attack_likely_activity",
        ]
        if len(self.state_names) < 2:
            raise ValueError("DemoTransitionModel needs at least two candidate states.")

    def predict_next(self, current_state: str, signal: ForecastSignal) -> List[TransitionCandidate]:
        score = min(1.0, max(0.0, float(signal.score)))
        # These are transparent simulation weights, intentionally not trained.
        weights = {
            self.state_names[0]: max(0.05, 1.0 - score),
            self.state_names[1]: 0.25 + 0.5 * (1.0 - abs(score - 0.5) * 2.0),
            self.state_names[2]: 0.15 + score,
            self.state_names[3]: 0.05 + 1.5 * score,
        }
        weights = {name: weights.get(name, 1.0) for name in self.state_names}
        total = sum(weights.values())
        candidates = []
        for state, weight in weights.items():
            probability = float(weight / total)
            candidates.append(
                TransitionCandidate(
                    state=state,
                    probability=probability,
                    score=probability,
                    metadata={
                        "transition_mode": self.mode,
                        "from_state": current_state,
                        "signal_source": signal.source,
                    },
                )
            )
        return candidates


def validate_transition_distribution(candidates: List[TransitionCandidate], tolerance: float = 1e-9) -> None:
    if not candidates:
        raise ValueError("Transition model returned no candidates.")
    probabilities = [candidate.probability for candidate in candidates]
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("Transition probabilities must be non-negative.")
    if abs(sum(probabilities) - 1.0) > tolerance:
        raise ValueError("Transition probabilities must sum to one.")
