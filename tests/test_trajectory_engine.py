import math

from src.forecasting.ptag import PTAGNode, ProbabilisticTemporalAttackGraph
from src.forecasting.trajectory_search import search_top_k_trajectories
from src.forecasting.world_model import DemoTransitionModel, ForecastSignal, TransitionCandidate


def _graph():
    graph = ProbabilisticTemporalAttackGraph()
    root = PTAGNode("t0:current", "current", 0)
    graph.add_node(root)
    graph.add_transitions(
        root,
        [
            TransitionCandidate("quiet", 0.6, 0.6),
            TransitionCandidate("rising", 0.3, 0.3),
            TransitionCandidate("anomalous", 0.1, 0.1),
        ],
        1,
    )
    for source in ["t1:quiet", "t1:rising", "t1:anomalous"]:
        graph.add_transitions(
            graph.nodes[source],
            [TransitionCandidate("stable_next", 0.75, 0.75), TransitionCandidate("risk_next", 0.25, 0.25)],
            2,
        )
    return graph


def test_graph_creation_and_probability_normalization():
    graph = _graph()
    assert len(graph.nodes) == 6
    assert graph.transitions_are_normalized()


def test_beam_width_and_top_k_ordering():
    results = search_top_k_trajectories(_graph(), horizon=2, beam_width=2, top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].cumulative_log_probability >= results[1].cumulative_log_probability
    assert all(len(result.states) == 2 for result in results)


def test_probability_pruning():
    results = search_top_k_trajectories(_graph(), horizon=2, beam_width=20, top_k=20, min_probability=0.2)
    assert results
    assert all(result.cumulative_probability >= 0.2 for result in results)


def test_cycle_prevention_and_backtracking():
    graph = ProbabilisticTemporalAttackGraph()
    root = PTAGNode("t0:current", "current", 0)
    graph.add_node(root)
    graph.add_transitions(root, [TransitionCandidate("repeat", 0.7, 0.7), TransitionCandidate("new", 0.3, 0.3)], 1)
    graph.add_transitions(graph.nodes["t1:repeat"], [TransitionCandidate("repeat", 1.0, 1.0)], 2)
    graph.add_transitions(graph.nodes["t1:new"], [TransitionCandidate("finish", 1.0, 1.0)], 2)
    results = search_top_k_trajectories(graph, horizon=2, beam_width=10, top_k=10)
    assert len(results) == 1
    assert results[0].states == ("new", "finish")
    assert results[0].path == ("t1:new", "t2:finish")
    assert math.isclose(results[0].cumulative_probability, 0.3)


def test_demo_transition_is_explicit_and_deterministic():
    model = DemoTransitionModel()
    signal = ForecastSignal(0.7, source="test")
    first = model.predict_next("current", signal)
    second = model.predict_next("current", signal)
    assert model.mode == "demo"
    assert [(item.state, item.probability) for item in first] == [(item.state, item.probability) for item in second]
    assert math.isclose(sum(item.probability for item in first), 1.0)
