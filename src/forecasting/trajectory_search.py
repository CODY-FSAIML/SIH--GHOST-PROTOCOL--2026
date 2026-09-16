"""Numerically stable top-K trajectory search over a PTAG."""

from dataclasses import dataclass
import heapq
import math
from typing import List, Optional, Tuple

from .ptag import PTAGEdge, PTAGNode, ProbabilisticTemporalAttackGraph


@dataclass(frozen=True)
class Trajectory:
    states: Tuple[str, ...]
    transition_probabilities: Tuple[float, ...]
    transition_scores: Tuple[float, ...]
    cumulative_log_probability: float
    cumulative_probability: float
    path: Tuple[str, ...]
    rank: int = 0
    cumulative_score: float = 0.0


@dataclass
class _SearchRecord:
    node_id: str
    log_probability: float
    parent: Optional["_SearchRecord"]
    edge: Optional[PTAGEdge]


def _reconstruct(record: _SearchRecord, graph: ProbabilisticTemporalAttackGraph) -> Trajectory:
    records = []
    current = record
    while current.parent is not None:
        records.append(current)
        current = current.parent
    records.reverse()
    nodes = [graph.nodes[item.node_id] for item in records]
    probabilities = tuple(item.edge.probability for item in records if item.edge is not None)
    scores = tuple(item.edge.score for item in records if item.edge is not None)
    states = tuple(node.state for node in nodes)
    path = tuple(node.node_id for node in nodes)
    return Trajectory(
        states=states,
        transition_probabilities=probabilities,
        transition_scores=scores,
        cumulative_log_probability=record.log_probability,
        cumulative_probability=math.exp(record.log_probability),
        path=path,
    )


def search_top_k_trajectories(
    graph: ProbabilisticTemporalAttackGraph,
    start_node: str = "t0:current",
    horizon: int = 3,
    beam_width: int = 10,
    top_k: int = 5,
    min_probability: float = 0.0,
    score_mode: str = "probability",
) -> List[Trajectory]:
    """Return ranked, cycle-free trajectories using beam search in log space.

    ``score_mode='probability'`` preserves the original PTAG behavior. With
    ``score_mode='model_score'``, edge ``score`` values are the ranking signal;
    the path score is ``sum(log(max(edge.score, 1e-12)))``. These are model-
    derived trajectory scores, not calibrated probabilities.
    """
    if horizon < 1 or beam_width < 1 or top_k < 1:
        raise ValueError("horizon, beam_width, and top_k must be positive")
    if min_probability < 0.0 or min_probability > 1.0:
        raise ValueError("min_probability must be in [0, 1]")
    if score_mode not in {"probability", "model_score"}:
        raise ValueError("score_mode must be 'probability' or 'model_score'")
    if start_node not in graph.nodes:
        raise KeyError(f"Unknown start node: {start_node}")

    frontier = [_SearchRecord(start_node, 0.0, None, None)]
    for _ in range(horizon):
        heap = []
        counter = 0
        for record in frontier:
            used_states = {graph.nodes[record.node_id].state}
            ancestor = record.parent
            while ancestor is not None:
                used_states.add(graph.nodes[ancestor.node_id].state)
                ancestor = ancestor.parent
            for edge in graph.outgoing(record.node_id):
                if edge.probability <= 0.0:
                    continue
                target_state = graph.nodes[edge.target].state
                if target_state in used_states:
                    continue
                edge_log_score = edge.log_probability
                if score_mode == "model_score":
                    edge_log_score = math.log(max(float(edge.score), 1e-12))
                next_log_probability = record.log_probability + edge_log_score
                if min_probability > 0.0 and math.exp(next_log_probability) < min_probability:
                    continue
                candidate = _SearchRecord(edge.target, next_log_probability, record, edge)
                heapq.heappush(heap, (-next_log_probability, counter, candidate))
                counter += 1
        frontier = [heapq.heappop(heap)[2] for _ in range(min(beam_width, len(heap)))]
        if not frontier:
            break

    trajectories = [_reconstruct(record, graph) for record in frontier]
    trajectories.sort(key=lambda item: item.cumulative_log_probability, reverse=True)
    return [
        Trajectory(
            states=item.states,
            transition_probabilities=item.transition_probabilities,
            transition_scores=item.transition_scores,
            cumulative_log_probability=item.cumulative_log_probability,
            cumulative_probability=item.cumulative_probability,
            path=item.path,
            rank=rank,
            cumulative_score=item.cumulative_probability if score_mode == "model_score" else 0.0,
        )
        for rank, item in enumerate(trajectories[:top_k], start=1)
    ]
