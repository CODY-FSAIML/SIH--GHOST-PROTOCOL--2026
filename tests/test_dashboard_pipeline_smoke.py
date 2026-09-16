import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, Raw, wrpcap
import dashboard.app as dashboard_app

from dashboard.app import (
    load_capture_file,
    discover_builtin_datasets,
    graph_dot,
    load_models,
    load_uploaded_capture,
    load_uploaded_pcap,
    load_selected_scenario,
    packet_summary,
    _parse_ctu13_timestamps,
    prepare_temporal_data,
    temporal_data_for_capture,
    world_model_result,
)
from src.explainability.mitre_mapping import explain_observation
from src.features.network_state import MACRO_FEATURE_NAMES


def _csv_bytes():
    return b"StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n" \
        b"2011/08/10 10:00:00.000000,0.1,tcp,10.0.0.1,1234,->,10.0.0.2,80,CON,0,0,2,100,60,flow=Background\n"


def test_dashboard_scenario_modes_and_pipeline_helpers():
    scenario42 = load_capture_file(str(Path("data/raw/ctu13/capture20110810.binetflow.txt")))
    scenario50 = load_capture_file(str(Path("data/raw/ctu13/capture20110817.binetflow")))
    assert len(scenario42["windows"]) > 0
    assert len(scenario50["windows"]) == 338

    uploaded = load_uploaded_capture(_csv_bytes(), "upload.csv")
    assert len(uploaded["windows"]) == 1
    assert uploaded["norm_df"]["is_attack"].iloc[0] == 0

    X, raw_X, meta = prepare_temporal_data(
        scenario50["norm_df"], scenario50["windows"], scenario50["states_lookup"], load_models()["scaler"]
    )
    models = load_models()
    result = world_model_result(models, X[0], horizon=2, top_k=2, beam_width=4)
    assert result is not None
    assert result.future_states.shape == (2, 32)
    assert len(result.trajectories) == 1
    interpretation = explain_observation(
        scenario50["states_lookup"][9].features,
        [scenario50["states_lookup"][index].features for index in range(4, 9)],
        velocity={name: float(raw_X[0, -1, 16 + index]) for index, name in enumerate(MACRO_FEATURE_NAMES)},
    )
    assert np.isfinite(interpretation["evidence_score"])


def test_dashboard_pcap_path():
    packets = [
        IP(src="192.0.2.1", dst="198.51.100.1") / TCP(sport=1234, dport=443, flags="S") / Raw(b"x"),
    ]
    packets[0].time = 1700000000.0
    path = Path("tests") / "_dashboard_smoke_tmp.pcap"
    try:
        wrpcap(str(path), packets)
        capture = load_uploaded_pcap(path.read_bytes(), "smoke.pcap")
        assert len(capture["packet_df"]) == 1
        assert len(capture["flow_df"]) == 1
        assert packet_summary(capture["packet_df"], capture["flow_df"])["packets"] == 1
        assert len(capture["states_lookup"]) == 1
    finally:
        path.unlink(missing_ok=True)


def test_dashboard_invalid_csv_fails_without_process_crash():
    try:
        load_uploaded_capture(b"not,a,valid,ctu13,file\n1,2,3,4,5", "invalid.csv")
    except Exception as error:
        assert isinstance(error, Exception)
        return
    raise AssertionError("Invalid CSV should be rejected by the adapter")


def test_selected_scenario_dispatches_only_the_selected_capture(monkeypatch):
    calls = []

    def fake_load(path):
        calls.append(Path(path).name)
        return {"name": Path(path).name}

    monkeypatch.setattr(dashboard_app, "load_capture_file", fake_load)
    dataset = {"dataset_name": "CTU-13 Scenario 50", "path": Path("data/raw/ctu13/capture20110817.binetflow"), "available": True}
    assert load_selected_scenario(dataset)["name"] == "capture20110817.binetflow"
    assert calls == ["capture20110817.binetflow"]


def test_builtin_dataset_discovery_is_lightweight_and_extensible(monkeypatch, tmp_path):
    benchmark_dir = tmp_path / "ctu13"
    benchmark_dir.mkdir()
    (benchmark_dir / "capture20110810.binetflow.txt").touch()
    (benchmark_dir / "capture20110817.binetflow").touch()
    (benchmark_dir / "future_capture.binetflow").touch()
    (benchmark_dir / "notes.txt").touch()
    monkeypatch.setattr(dashboard_app, "BUILTIN_DATASET_DIRECTORIES", ((benchmark_dir, "CTU-13"),))
    entries = discover_builtin_datasets()
    assert [item["dataset_name"] for item in entries] == [
        "CTU-13 Scenario 42", "CTU-13 Scenario 50", "CTU-13 future_capture.binetflow",
    ]
    assert all(item["available"] and item["format"] == "binetflow" for item in entries)


def test_builtin_capture_cache_reuses_a_processed_scenario(monkeypatch, tmp_path):
    capture_file = tmp_path / "tiny.binetflow"
    capture_file.write_bytes(_csv_bytes())
    reads = 0
    original_read_csv = dashboard_app.pd.read_csv

    def counted_read_csv(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_csv(*args, **kwargs)

    dashboard_app.load_capture_file.clear()
    monkeypatch.setattr(dashboard_app.pd, "read_csv", counted_read_csv)
    first = dashboard_app.load_capture_file(str(capture_file))
    second = dashboard_app.load_capture_file(str(capture_file))
    assert len(first["windows"]) == len(second["windows"]) == 1
    assert reads == 1


def test_temporal_history_is_attached_once_to_cached_capture(monkeypatch):
    calls = 0
    capture = {"norm_df": object(), "windows": object(), "states_lookup": object()}
    expected = (np.empty((1, 10, 32)), np.empty((1, 10, 32)), [{"anchor_window_idx": 9}])

    def fake_prepare(*_args):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(dashboard_app, "prepare_temporal_data", fake_prepare)
    assert temporal_data_for_capture(capture, object()) is expected
    assert temporal_data_for_capture(capture, object()) is expected
    assert calls == 1


def test_ctu13_timestamp_parser_preserves_2011_epoch_seconds():
    timestamps = _parse_ctu13_timestamps(pd.Series([
        "2011/08/10 09:46:53.047277", "2011/08/10 15:54:07.000000",
    ]))
    rendered = pd.to_datetime(timestamps, unit="s")
    assert rendered.iloc[0].year == 2011
    assert rendered.iloc[0].hour == 9
    assert rendered.iloc[1].hour == 15


def test_ctu13_loader_builds_h10_windows_from_2011_timestamps(tmp_path):
    rows = [
        "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"
    ]
    for minute in range(12):
        rows.append(
            f"2011/08/10 09:{minute:02d}:00.000000,0.1,tcp,10.0.0.1,1234,->,10.0.0.2,80,CON,0,0,2,100,60,flow=Background"
        )
    capture_file = tmp_path / "capture20110810.binetflow.txt"
    capture_file.write_text("\n".join(rows), encoding="utf-8")
    load_capture_file.clear()
    capture = load_capture_file(str(capture_file))
    assert len(capture["windows"]) == 11
    X, _, meta = prepare_temporal_data(capture["norm_df"], capture["windows"], capture["states_lookup"], None)
    assert X.shape == (1, 10, 32)
    assert meta[0]["anchor_window_idx"] == 9


def test_deterministic_world_model_path_is_not_hidden_by_pruning(monkeypatch):
    calls = []
    visible_result = SimpleNamespace(trajectories=[object()])

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self, *_args, min_score, **_kwargs):
            calls.append(min_score)
            return SimpleNamespace(trajectories=[]) if min_score else visible_result

    monkeypatch.setattr(dashboard_app, "LearnedWorldModelPTAGAdapter", FakeAdapter)
    result = world_model_result({"world_model": object()}, np.zeros((10, 32)), 5, 3, 8, min_score=0.001)
    assert result is visible_result
    assert calls == [0.001, 0.0]


def test_topology_dot_uses_bounded_activity_sizes_and_selected_edges():
    graph = {
        "node_ids": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        "node_features": np.array([[1, 10, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=float),
        "edge_meta": [("10.0.0.1", "10.0.0.2"), ("10.0.0.2", "10.0.0.3")],
        "edge_features": np.array([[12, 2400, 24, 0, 1, 0], [1, 50, 1, 0, 1, 0]], dtype=float),
    }
    dot, shown_nodes, edge_count = graph_dot(graph, max_nodes=2)
    assert shown_nodes == 2
    assert edge_count == 2
    assert "fixedsize=true" in dot
    assert "width=0." in dot
    assert "flows: 12" in dot
