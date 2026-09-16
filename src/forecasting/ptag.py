"""Dynamic Probabilistic Temporal Attack Graph (PTAG).

Nodes represent generic future network-behavior states. No MITRE or attack-stage
vocabulary is embedded here; domain labels are supplied by the transition model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from .world_model import ForecastSignal, StateTransitionModel, validate_transition_distribution


@dataclass(frozen=True)
class PTAGNode:
    node_id: str
    state: str
    step: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PTAGEdge:
    source: str
    target: str
    probability: float
    score: float
    log_probability: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProbabilisticTemporalAttackGraph:
    """A layered directed graph with normalized probabilistic transitions."""

    def __init__(self):
        self.nodes: Dict[str, PTAGNode] = {}
        self.edges: Dict[str, List[PTAGEdge]] = {}

    def add_node(self, node: PTAGNode) -> None:
        self.nodes[node.node_id] = node
        self.edges.setdefault(node.node_id, [])

    def add_transitions(
        self,
        source: PTAGNode,
        candidates: Iterable[Any],
        step: int,
    ) -> List[PTAGEdge]:
        candidates = list(candidates)
        validate_transition_distribution(candidates)
        created = []
        for candidate in candidates:
            target_id = f"t{step}:{candidate.state}"
            target = PTAGNode(
                node_id=target_id,
                state=candidate.state,
                step=step,
                metadata=dict(candidate.metadata),
            )
            self.add_node(target)
            probability = float(candidate.probability)
            import math
            edge = PTAGEdge(
                source=source.node_id,
                target=target_id,
                probability=probability,
                score=float(candidate.score),
                log_probability=math.log(probability) if probability > 0.0 else float("-inf"),
                metadata=dict(candidate.metadata),
            )
            self.edges[source.node_id].append(edge)
            created.append(edge)
        return created

    def outgoing(self, node_id: str) -> List[PTAGEdge]:
        return list(self.edges.get(node_id, []))

    def transitions_are_normalized(self, tolerance: float = 1e-9) -> bool:
        return all(
            abs(sum(edge.probability for edge in edges) - 1.0) <= tolerance
            for edges in self.edges.values()
            if edges
        )


def build_dynamic_ptag(
    current_state: str,
    signal: ForecastSignal,
    transition_model: StateTransitionModel,
    horizon: int,
) -> ProbabilisticTemporalAttackGraph:
    """Expand a dynamic PTAG for ``horizon`` future steps."""
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    graph = ProbabilisticTemporalAttackGraph()
    root = PTAGNode(node_id="t0:current", state=current_state, step=0)
    graph.add_node(root)
    frontier = [root]
    for step in range(1, horizon + 1):
        next_frontier = {}
        for source in frontier:
            candidates = transition_model.predict_next(source.state, signal)
            graph.add_transitions(source, candidates, step)
            for edge in graph.outgoing(source.node_id):
                next_frontier[edge.target] = graph.nodes[edge.target]
        frontier = list(next_frontier.values())
    return graph
