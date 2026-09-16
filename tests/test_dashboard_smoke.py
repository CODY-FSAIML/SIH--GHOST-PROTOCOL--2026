import numpy as np

from dashboard.app import build_trajectory, load_capture_file, load_models, prepare_temporal_data, temporal_scores


def test_scenario50_dashboard_inference_smoke():
    capture = load_capture_file(r"data/raw/ctu13/capture20110817.binetflow")
    models = load_models()
    X, raw_X, meta = prepare_temporal_data(
        capture["norm_df"], capture["windows"], capture["states_lookup"], models["scaler"]
    )
    scores = temporal_scores(models["temporal"], X)
    graph, trajectories = build_trajectory(float(scores[0]), horizon=2, top_k=2, beam_width=4)
    assert len(capture["windows"]) == 338
    assert len(X) == len(meta) == 328
    assert np.isfinite(scores).all()
    assert raw_X.shape == (328, 10, 32)
    assert len(graph.nodes) > 1
    assert len(trajectories) == 2
