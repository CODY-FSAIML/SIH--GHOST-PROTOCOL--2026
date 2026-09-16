"""NetGuardAI Streamlit demonstration dashboard.

This app performs inference only. It does not train models or fit any scaler.
"""

from __future__ import annotations

import io
import gc
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:  # Allows helper imports and smoke checks before Streamlit is installed.
    st = None

from src.data.adapters.ctu13_adapter import CTU13Adapter  # noqa: E402
from src.data.adapters.ctu13_adapter import _state_category  # noqa: E402
from src.data.adapters.cicids2018_adapter import CICIDS2018Adapter  # noqa: E402
from src.data.pcap_adapter import PCAPAdapter  # noqa: E402
from src.features.network_state import MACRO_FEATURE_NAMES, MacroNetworkStateBuilder  # noqa: E402
from src.features.temporal_dynamics import build_temporal_dynamics_dataset  # noqa: E402
from src.forecasting.ptag import build_dynamic_ptag  # noqa: E402
from src.forecasting.trajectory_search import search_top_k_trajectories  # noqa: E402
from src.forecasting.world_model import DemoTransitionModel, ForecastSignal  # noqa: E402
from src.forecasting.learned_world_model import LearnedLatentWorldModel  # noqa: E402
from src.forecasting.world_model_ptag_adapter import LearnedWorldModelPTAGAdapter  # noqa: E402
from src.graph.communication_graph import build_communication_graphs  # noqa: E402
from src.models.fusion_model import TemporalGraphFusionModel  # noqa: E402
from src.models.graph_gnn import GraphSAGEModel  # noqa: E402
from src.models.temporal_transformer import TemporalTransformerForecaster  # noqa: E402
from src.explainability.feature_explanations import rank_driving_indicators  # noqa: E402
from src.explainability.mitre_mapping import explain_observation  # noqa: E402

BUILTIN_DATASET_DIRECTORIES = ((PROJECT_ROOT / "data/raw/ctu13", "CTU-13"),)
TEMPORAL_CHECKPOINT = PROJECT_ROOT / "models/temporal_h10_state_velocity_best.pt"
TEMPORAL_SCALER = PROJECT_ROOT / "models/temporal_h10_state_velocity_scaler.pkl"
GNN_CHECKPOINT = PROJECT_ROOT / "scratch/gnn_only_best.pt"
FUSION_CHECKPOINT = PROJECT_ROOT / "models/temporal_graph_fusion_best.pt"
WORLD_MODEL_CHECKPOINT = PROJECT_ROOT / "models/latent_world_model_h10_best.pt"


def _cache_data(func):
    return st.cache_data(show_spinner=False)(func) if st is not None else func


def _cache_resource(func):
    return st.cache_resource(show_spinner=False)(func) if st is not None else func


def _metadata_value(metadata: Any, key: str, default: Any = 0.0) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def discover_builtin_datasets() -> List[Dict[str, Any]]:
    """Return compatible local benchmark captures without reading their contents."""
    entries: List[Dict[str, Any]] = []
    scenario_dates = {"20110810": "42", "20110817": "50"}
    for directory, family in BUILTIN_DATASET_DIRECTORIES:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or ".binetflow" not in path.name.lower():
                continue
            scenario = next((number for date, number in scenario_dates.items() if date in path.name), None)
            capture_name = f"{family} Scenario {scenario}" if scenario else f"{family} {path.name}"
            entries.append({
                "dataset_name": capture_name,
                "dataset_family": family,
                "path": path,
                "format": "binetflow",
                "available": path.is_file(),
                "source_type": "Built-in flow telemetry",
            })
    return entries


def _select_flow_adapter(columns: Iterable[str]):
    available = set(columns)
    ctu_required = {"StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport", "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"}
    if ctu_required.issubset(available):
        return CTU13Adapter()
    cic_required = {"Timestamp", "Src IP", "Dst IP", "Protocol", "Flow Duration", "Label"}
    if cic_required.issubset(available):
        return CICIDS2018Adapter()
    raise ValueError(
        "Unsupported flow telemetry format. Provide CTU-13-compatible fields "
        "or CIC-IDS2018-compatible fields (Timestamp, Src IP, Dst IP, Protocol, Flow Duration, Label)."
    )


def _parse_ctu13_timestamps(values: pd.Series) -> pd.Series:
    """Return CTU-13 timestamps as POSIX seconds using an explicit ns epoch."""
    parsed = pd.to_datetime(values, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    return parsed.astype("datetime64[ns]").astype(np.int64) / 1e9


def _canonical_to_norm(canonical: pd.DataFrame) -> pd.DataFrame:
    metadata = canonical.get("metadata")
    if metadata is None:
        timestamp = pd.to_datetime(canonical["timestamp_str"], errors="coerce").astype("datetime64[ns]").astype(np.int64) / 1e9
        sent_bytes = pd.to_numeric(canonical["bytes_sent"], errors="coerce").fillna(0.0)
        received_bytes = pd.to_numeric(canonical["bytes_received"], errors="coerce").fillna(0.0)
        sent_packets = pd.to_numeric(canonical["packets_sent"], errors="coerce").fillna(0.0)
        received_packets = pd.to_numeric(canonical["packets_received"], errors="coerce").fillna(0.0)
        return pd.DataFrame({
            "timestamp": timestamp,
            "duration": pd.to_numeric(canonical["duration"], errors="coerce").fillna(0.0),
            "protocol": canonical["protocol"].astype(str).str.lower().str.strip(),
            "src_ip": canonical["src_ip"].astype(str).str.strip(),
            "dst_ip": canonical["dst_ip"].astype(str).str.strip(),
            "total_packets": sent_packets + received_packets,
            "total_bytes": sent_bytes + received_bytes,
            "src_bytes": sent_bytes,
            "state_category": "other",
            "raw_label": canonical["raw_label"].astype(str),
            "is_attack": pd.to_numeric(canonical["is_attack"], errors="coerce").fillna(0).astype(int),
        }, index=canonical.index)
    return pd.DataFrame({
        "timestamp": pd.to_numeric(canonical["timestamp"], errors="coerce"),
        "duration": pd.to_numeric(canonical["duration"], errors="coerce").fillna(0.0),
        "protocol": canonical["protocol"].astype(str).str.lower().str.strip(),
        "src_ip": canonical["src_ip"].astype(str).str.strip(),
        "dst_ip": canonical["dst_ip"].astype(str).str.strip(),
        "total_packets": [float(_metadata_value(item, "total_packets")) for item in canonical["metadata"]],
        "total_bytes": [float(_metadata_value(item, "total_bytes")) for item in canonical["metadata"]],
        "src_bytes": [float(_metadata_value(item, "src_bytes")) for item in canonical["metadata"]],
        "state_category": [str(_metadata_value(item, "state_category", "other")) for item in canonical["metadata"]],
        "raw_label": canonical["raw_label"].astype(str),
        "is_attack": pd.to_numeric(canonical["is_attack"], errors="coerce").fillna(0).astype(int),
    }, index=canonical.index)


def _capture_from_norm(norm_df: pd.DataFrame, name: str) -> Dict[str, Any]:
    builder = MacroNetworkStateBuilder(window_size_sec=60.0)
    states, _, _ = builder.build_states(norm_df)
    from src.data.temporal_targets import build_temporal_windows
    windows = build_temporal_windows(norm_df, window_size_sec=60.0)
    return {"name": name, "norm_df": norm_df, "states_lookup": {s.window_idx: s for s in states}, "windows": windows}


@_cache_resource
def load_capture_file(path_string: str) -> Dict[str, Any]:
    """Load exactly one built-in CTU capture; the path is the cache key.

    Keeping this loader in the dashboard avoids importing an experiment runner and
    ensures changing the sidebar scenario cannot trigger work for the other one.
    ``cache_resource`` deliberately shares this very large immutable-in-practice
    capture instead of serializing a second full dataframe per rerun.
    """
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Built-in scenario capture is unavailable: {path.name}")
    columns = ["StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "TotPkts", "TotBytes", "SrcBytes", "State", "Label"]
    raw = pd.read_csv(path, usecols=columns, low_memory=False)
    # This is the established CTU dashboard preprocessing path.  It avoids a
    # second canonical dataframe (and its per-flow metadata dictionaries) for
    # multi-million-row built-in demonstrations.
    # Explicit ns conversion is essential with pandas versions that preserve
    # source resolution (for example microseconds) in an integer datetime cast.
    timestamp = _parse_ctu13_timestamps(raw["StartTime"])
    labels = raw["Label"].astype(str)
    norm_df = pd.DataFrame({
        "timestamp": timestamp,
        "duration": pd.to_numeric(raw["Dur"], errors="coerce").fillna(0.0),
        "protocol": raw["Proto"].astype(str).str.lower().str.strip(),
        "src_ip": raw["SrcAddr"].astype(str).str.strip(),
        "dst_ip": raw["DstAddr"].astype(str).str.strip(),
        "total_packets": pd.to_numeric(raw["TotPkts"], errors="coerce").fillna(0),
        "total_bytes": pd.to_numeric(raw["TotBytes"], errors="coerce").fillna(0),
        "src_bytes": pd.to_numeric(raw["SrcBytes"], errors="coerce").fillna(0),
        "state_category": [_state_category(str(value)) for value in raw["State"]],
        "raw_label": labels,
        "is_attack": labels.str.contains("Botnet", case=True, regex=False).astype(int),
    })
    # State/window construction is the high-water mark.  Release the raw
    # parsed table before entering it so a selected capture is not retained
    # twice in memory.
    del raw, labels
    gc.collect()
    return _capture_from_norm(norm_df, path.stem)


def load_selected_scenario(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Load only the selected discovery entry, never every built-in capture."""
    if not dataset.get("available") or not isinstance(dataset.get("path"), Path):
        raise ValueError("Selected built-in benchmark capture is unavailable.")
    return load_capture_file(str(dataset["path"]))


@_cache_data
def load_uploaded_capture(file_bytes: bytes, name: str) -> Dict[str, Any]:
    raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    adapter = _select_flow_adapter(raw.columns)
    canonical = adapter.normalize(raw)
    adapter.validate(canonical)
    return _capture_from_norm(_canonical_to_norm(canonical), name)


@_cache_data
def load_uploaded_pcap(file_bytes: bytes, name: str) -> Dict[str, Any]:
    adapter = PCAPAdapter()
    handle = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
    try:
        handle.write(file_bytes)
        handle.flush()
        packet_df, flow_df, normalized = adapter.load_for_network_state(handle.name)
    finally:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
    capture = _capture_from_norm(normalized, name)
    capture["packet_df"] = packet_df
    capture["flow_df"] = flow_df
    capture["input_mode"] = "PCAP / Packet telemetry"
    return capture


@_cache_resource
def load_temporal_assets() -> Dict[str, Any]:
    import joblib
    if not TEMPORAL_CHECKPOINT.is_file() or not TEMPORAL_SCALER.is_file():
        raise FileNotFoundError("Temporal Model checkpoint or scaler unavailable.")
    temporal_payload = torch.load(TEMPORAL_CHECKPOINT, map_location="cpu", weights_only=False)
    temporal_model = TemporalTransformerForecaster(**temporal_payload["model_config"])
    temporal_model.load_state_dict(temporal_payload["state_dict"])
    temporal_model.eval()
    scaler = joblib.load(TEMPORAL_SCALER)
    return {"temporal": temporal_model, "scaler": scaler}


@_cache_resource
def load_graph_assets() -> Dict[str, Any]:
    if not GNN_CHECKPOINT.is_file():
        raise FileNotFoundError("GNN checkpoint unavailable.")
    graph_model = GraphSAGEModel(in_dim=6, hidden_dim=32, num_layers=2)
    graph_model.load_state_dict(torch.load(GNN_CHECKPOINT, map_location="cpu", weights_only=False))
    graph_model.eval()

    fusion_model = None
    if FUSION_CHECKPOINT.exists():
        temporal_model = load_temporal_assets()["temporal"]
        fusion_payload = torch.load(FUSION_CHECKPOINT, map_location="cpu", weights_only=False)
        fusion_model = TemporalGraphFusionModel(temporal_model, graph_model, fusion_hidden_dim=32, dropout=0.1)
        fusion_model.load_state_dict(fusion_payload["state_dict"])
        fusion_model.eval()
    return {"graph": graph_model, "fusion": fusion_model}


@_cache_resource
def load_world_model_assets() -> Dict[str, Any]:
    world_model = None
    if WORLD_MODEL_CHECKPOINT.exists():
        world_model, world_scaler, checkpoint = LearnedLatentWorldModel.from_checkpoint(WORLD_MODEL_CHECKPOINT)
        world_model.eval()
        parameter_count = int(checkpoint.get("parameter_count", sum(parameter.numel() for parameter in world_model.parameters())))
    else:
        world_scaler = None
        parameter_count = None
    return {"world_model": world_model, "world_scaler": world_scaler, "parameter_count": parameter_count}


@_cache_resource
def load_models() -> Dict[str, Any]:
    """Compatibility helper for tests and scripts that need the full model stack."""
    return {**load_temporal_assets(), **load_graph_assets(), **load_world_model_assets()}


@_cache_data
def prepare_temporal_data(norm_df: pd.DataFrame, windows: List[Any], states_lookup: Dict[int, Any], _scaler: Any):
    X, _, _, meta = build_temporal_dynamics_dataset(
        windows, states_lookup, history_len=10, include_velocity=True, scaler=_scaler, fit_scaler=False
    )
    raw_X, _, _, _ = build_temporal_dynamics_dataset(
        windows, states_lookup, history_len=10, include_velocity=True, scaler=None, fit_scaler=False
    )
    return X, raw_X, meta


def temporal_data_for_capture(capture: Dict[str, Any], scaler: Any):
    """Build H10 inputs once for a cached capture without hashing its full dataframe.

    Built-in captures are ``st.cache_resource`` objects. Storing this small
    derived result on that resource prevents widget reruns from repeatedly
    hashing millions of flows before dashboard rendering can continue.
    """
    cached = capture.get("_temporal_h10")
    if cached is None:
        cached = prepare_temporal_data(capture["norm_df"], capture["windows"], capture["states_lookup"], scaler)
        capture["_temporal_h10"] = cached
    return cached


def temporal_scores(model: TemporalTransformerForecaster, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0, dtype=float)
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def fusion_score(models: Dict[str, Any], norm_df: pd.DataFrame, window: Any, temporal_x: np.ndarray) -> float | None:
    if models["fusion"] is None:
        return None
    graph = build_communication_graphs(norm_df, [window])[0]
    node_features = graph["node_features"]
    edge_index = graph["edge_index"]
    if node_features.shape[0] == 0:
        node_features = np.zeros((1, 6), dtype=np.float32)
        edge_index = np.empty((2, 0), dtype=np.int64)
    with torch.no_grad():
        score = torch.sigmoid(models["fusion"](
            torch.as_tensor(temporal_x[None, ...], dtype=torch.float32),
            torch.as_tensor(node_features, dtype=torch.float32),
            torch.as_tensor(edge_index, dtype=torch.long),
            torch.zeros(len(node_features), dtype=torch.long),
        ))
    return float(score.item())


def graph_dot(graph: Dict[str, Any], max_nodes: int = 40) -> Tuple[str, int, int]:
    node_ids = graph["node_ids"]
    features = graph["node_features"]
    activity = np.sum(np.abs(features), axis=1) if len(features) else np.empty(0)
    ranked = sorted(range(len(node_ids)), key=lambda index: (-float(activity[index]), node_ids[index]))[:max_nodes]
    selected = {node_ids[index] for index in ranked}
    max_activity = float(activity[ranked].max()) if ranked and float(activity[ranked].max()) > 0 else 1.0
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="transparent", nodesep=0.28, ranksep=0.42];',
        '  node [shape=box, fixedsize=true, style="rounded,filled", fillcolor="#17324d", color="#5ce1e6", penwidth=1.0];',
        '  edge [color="#54788d", arrowsize=0.55];',
    ]
    for index in ranked:
        label = node_ids[index].replace('"', "'")
        normalized_activity = min(1.0, float(activity[index]) / max_activity)
        # Point nodes remain compact even for a high-volume host. Activity is
        # still encoded, but bounded to a subtle visual range.
        size = 0.08 + 0.10 * normalized_activity
        tooltip = f"Entity: {label} | activity: {float(activity[index]):.0f}"
        lines.append(f'  "{label}" [label="", width={size:.2f}, height={size:.2f}, tooltip="{tooltip}"];')
    for edge_index, (source, target) in enumerate(graph["edge_meta"]):
        if source in selected and target in selected:
            edge_feature = graph.get("edge_features", np.empty((0, 6)))[edge_index]
            flows, total_bytes, packets = (float(edge_feature[0]), float(edge_feature[1]), float(edge_feature[2]))
            width = min(3.0, 0.6 + np.log1p(max(0.0, flows)) * 0.35)
            tooltip = f"{source} → {target} | flows: {flows:.0f} | bytes: {total_bytes:.0f} | packets: {packets:.0f}"
            lines.append(f'  "{source}" -> "{target}" [penwidth={width:.2f}, tooltip="{tooltip}"];')
    lines.append("}")
    return "\n".join(lines), len(selected), len(graph["edge_meta"])


def build_trajectory(score: float, horizon: int, top_k: int, beam_width: int):
    signal = ForecastSignal(score=score, source="verified temporal/fusion model score")
    graph = build_dynamic_ptag("observed_current_state", signal, DemoTransitionModel(), horizon)
    trajectories = search_top_k_trajectories(graph, horizon=horizon, top_k=top_k, beam_width=beam_width, min_probability=0.001)
    return graph, trajectories


def world_model_result(models: Dict[str, Any], history: np.ndarray, horizon: int, top_k: int, beam_width: int, min_score: float = 0.001):
    if models.get("world_model") is None:
        return None
    adapter = LearnedWorldModelPTAGAdapter(models["world_model"], candidate_count=1, seed=42)
    result = adapter.build(history, horizon=horizon, top_k=top_k, beam_width=beam_width, min_score=min_score)
    # A deterministic one-candidate rollout has one real path.  Its product of
    # model scores can fall below the UI pruning threshold across several steps,
    # even though every PTAG edge is valid. Re-run only this presentation search
    # without that cutoff so the actual ranked path is visible.
    if not result.trajectories and min_score > 0.0:
        result = adapter.build(history, horizon=horizon, top_k=top_k, beam_width=beam_width, min_score=0.0)
    return result


def packet_summary(packet_df: pd.DataFrame, flow_df: pd.DataFrame) -> Dict[str, Any]:
    if packet_df.empty:
        return {"packets": 0, "flows": len(flow_df), "ttl": None, "flags": {}, "tcp_window": None, "payload": None, "iat": None, "retransmissions": 0, "fragmented": 0, "port_diversity": 0}
    metadata = flow_df.get("metadata", pd.Series(dtype=object)).tolist()
    def values(key):
        return [item.get(key) for item in metadata if isinstance(item, dict) and item.get(key) is not None]
    ttl = pd.to_numeric(packet_df.get("ttl"), errors="coerce").dropna()
    windows = values("tcp_window_mean")
    payloads = values("payload_size_mean")
    iats = values("iat_mean")
    return {
        "packets": len(packet_df), "flows": len(flow_df),
        "ttl": {"mean": float(ttl.mean()), "variance": float(ttl.var(ddof=0))} if len(ttl) else None,
        "flags": {flag: int(sum(int(item.get(f"{flag.lower()}_count", 0) or 0) for item in metadata if isinstance(item, dict))) for flag in ["SYN", "ACK", "FIN", "RST"]},
        "tcp_window": float(np.mean(windows)) if windows else None,
        "payload": float(np.mean(payloads)) if payloads else None,
        "iat": float(np.mean(iats)) if iats else None,
        "retransmissions": int(sum(int(item.get("retransmission_count", 0) or 0) for item in metadata if isinstance(item, dict))),
        "fragmented": int(sum(int(item.get("fragmented_packet_count", 0) or 0) for item in metadata if isinstance(item, dict))),
        "port_diversity": int(sum(int(item.get("destination_port_diversity", 0) or 0) for item in metadata if isinstance(item, dict))),
    }


def _render_pipeline():
    st.markdown(
        '<div class="pipeline-strip">'
        '<span>NETWORK TRAFFIC</span><b>→</b><span>NETWORK STATE</span><b>→</b>'
        '<span>TEMPORAL + GRAPH</span><b>→</b><span>UNIFIED FORECAST</span><b>→</b>'
        '<span>PTAG</span><b>→</b><span>TOP-K FUTURES</span><b>→</b><span>SOC DECISION</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_card(label: str, value: str, detail: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install the dashboard dependencies before running this app.")
    st.set_page_config(page_title="NETGUARD AI | SOC Forecast", page_icon="N", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#f0f4f8; --muted:#a8b8c8; --line:#243746; --cyan:#5ce1e6; --lime:#b6e35b; --amber:#f4b860; }
    .stApp { background: radial-gradient(circle at 75% 50%, rgba(92,225,230,0.1) 0%, rgba(16,36,51,0.8) 30%, #05080c 80%); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] {
        background-color: rgba(12, 18, 28, 0.75) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(92, 225, 230, 0.15) !important;
        border-radius: 24px !important;
        margin: 16px !important;
        height: calc(100vh - 32px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 24px rgba(92, 225, 230, 0.08);
        overflow: hidden;
    }
    .block-container { max-width:1500px; padding:2rem 3rem 1rem; }
    h1, h2, h3, p, label, li { font-family:'Space Grotesk',sans-serif; }
    h1 { letter-spacing:.03em; font-weight:700; }
    h2 { margin-top:1.35rem; font-size:1.05rem; letter-spacing:.08em; text-transform:uppercase; color:#dce8f0; }
    .brand { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(92,225,230,0.15); padding-bottom:1.3rem; margin-bottom:1rem; }
    .brand-mark { color:#f0f4f8; font:600 1.55rem 'Space Grotesk',sans-serif; letter-spacing:.16em; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }
    .glow-cyan { color:var(--cyan); text-shadow: 0 0 16px rgba(92,225,230,0.7); font-weight:700; }
    .brand-sub { color:var(--muted); font:500 .68rem 'IBM Plex Mono',monospace; letter-spacing:.14em; margin-top:.25rem; }
    .status { border:1px solid #416b62; background:rgba(16,37,31,0.6); color:var(--lime); border-radius:999px; padding:.45rem .8rem; font:500 .68rem 'IBM Plex Mono',monospace; letter-spacing:.08em; backdrop-filter:blur(8px); }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(182, 227, 91, 0.7); } 70% { box-shadow: 0 0 0 8px rgba(182, 227, 91, 0); } 100% { box-shadow: 0 0 0 0 rgba(182, 227, 91, 0); } }
    .status-dot { display:inline-block; width:7px; height:7px; background:var(--lime); border-radius:50%; box-shadow:0 0 10px var(--lime); margin-right:.45rem; animation: pulse 2s infinite; }
    .metric-card { min-height:105px; background:linear-gradient(145deg,rgba(24,38,51,.7),rgba(11,20,29,.8)); backdrop-filter:blur(12px); border:1px solid var(--line); border-radius:12px; padding:1rem 1.05rem; box-shadow:0 8px 24px rgba(0,0,0,.15); transition:transform 0.2s ease, box-shadow 0.2s ease; }
    .metric-card:hover { transform:translateY(-2px); box-shadow:0 12px 32px rgba(0,0,0,.3); border-color:#3a5469; }
    .metric-label { color:var(--muted); font:500 .68rem 'IBM Plex Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font-size:1.55rem; font-weight:600; margin:.45rem 0 .2rem; }
    .metric-detail { color:#678092; font:400 .7rem 'IBM Plex Mono',monospace; }
    .score-card { border-color:rgba(92,225,230,0.3); box-shadow:0 0 28px rgba(92,225,230,.08); }
    .score-card .metric-value { color:var(--cyan); text-shadow: 0 0 16px rgba(92,225,230,0.3); }
    .panel { background:rgba(16,25,36,.65); backdrop-filter:blur(16px); border:1px solid var(--line); border-radius:12px; padding:1.25rem 1.35rem 1rem; min-height:100%; transition:transform 0.2s ease; box-shadow:0 8px 24px rgba(0,0,0,.1); }
    .panel-title { color:#dce8f0; font-size:.92rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
    .panel-kicker { color:var(--muted); font:400 .68rem 'IBM Plex Mono',monospace; margin:.3rem 0 .8rem; }
    .hero-score { color:var(--cyan); font-size:3.6rem; line-height:1; font-weight:700; letter-spacing:-.03em; text-shadow: 0 0 24px rgba(92,225,230,0.2); }
    .hero-score-label { color:var(--muted); font:500 .68rem 'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
    .pipeline-strip { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; border:1px solid var(--line); border-radius:7px; background:rgba(16,25,36,.62); padding:.65rem .8rem; color:var(--muted); font:500 .61rem 'IBM Plex Mono',monospace; letter-spacing:.04em; backdrop-filter:blur(4px); }
    .pipeline-strip b { color:var(--cyan); font-size:.9rem; }
    .section-rule { border-top:1px solid var(--line); margin:1.35rem 0 .2rem; }
    .note { color:var(--muted); font:400 .7rem 'IBM Plex Mono',monospace; }
    footer { border-top:1px solid var(--line); margin-top:2rem; padding:1rem 0 0; color:var(--muted); font:400 .68rem 'IBM Plex Mono',monospace; text-align:center; }
    [data-testid="stMetric"] { background:transparent; }
    [data-testid="stDataFrame"] { border: 1px solid rgba(92, 225, 230, 0.2); border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
    [data-testid="stDataFrame"] > div { background-color: rgba(16, 25, 36, 0.6) !important; backdrop-filter: blur(10px); }
    .control-room-btn { background: linear-gradient(90deg, rgba(92,225,230,0.15), rgba(92,225,230,0.02)); border: 1px solid rgba(92,225,230,0.5); border-radius: 12px; padding: 0.8rem 1rem; color: var(--cyan); font: 600 0.95rem 'Space Grotesk', sans-serif; display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 0 20px rgba(92,225,230,0.2), inset 0 0 10px rgba(92,225,230,0.1); margin-bottom: 1.5rem; text-shadow: 0 0 8px rgba(92,225,230,0.5); letter-spacing: .05em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; display: none;}
    .stDeployButton {display: none !important;}
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stCollapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stSidebar"][aria-expanded="false"] { display: block !important; transform: none !important; width: 310px !important; min-width: 310px !important; }
    [data-testid="stAlert"] { background-color: rgba(16, 25, 36, 0.75) !important; backdrop-filter: blur(12px) !important; border: 1px solid rgba(92, 225, 230, 0.3) !important; border-radius: 12px !important; color: var(--ink) !important; box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="brand"><div><div class="brand-mark">NETGUARD <span class="glow-cyan">AI</span></div><div class="brand-sub">AI NETWORK ATTACK FORECASTING</div></div><div class="status"><span class="status-dot"></span>ANALYSIS ACTIVE</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note">See the Next Move. Stop the Breach.</div>', unsafe_allow_html=True)
    _render_pipeline()

    with st.sidebar:
        st.markdown('<div class="control-room-btn">❖ CONTROL ROOM</div>', unsafe_allow_html=True)
        st.markdown("### DATA SOURCE")
        source_mode = st.radio("Data Source", ["Built-in Benchmark", "Upload CSV / Flow Telemetry", "Upload PCAP / Packet Telemetry"])
        datasets = discover_builtin_datasets()
        selected_dataset = None
        uploaded = None
        if source_mode == "Built-in Benchmark":
            st.caption("Built-in CTU-13 captures are benchmark examples in a dataset-agnostic forecasting pipeline.")
            if datasets:
                names = [item["dataset_name"] for item in datasets]
                selected_dataset = datasets[names.index(st.selectbox("Built-in benchmark", names))]
            else:
                st.warning("No compatible built-in benchmark captures are available.")
        elif source_mode == "Upload CSV / Flow Telemetry":
            st.caption("Upload compatible network flow telemetry. The same forecasting pipeline is used regardless of dataset source.")
            uploaded = st.file_uploader("CSV / Flow Telemetry", type=["csv", "txt"])
        else:
            st.caption("Upload an offline PCAP/PCAPNG capture. Packet-level features are extracted locally using the PCAP adapter.")
            uploaded = st.file_uploader("PCAP / Packet Telemetry", type=["pcap", "pcapng", "cap"])
        horizon = st.slider("Forecast horizon", 1, 5, 3)
        st.caption("Number of future network states to simulate.")
        top_k = st.slider("Top-K trajectories", 1, 10, 3)
        st.caption("Number of highest-scoring future trajectories to display.")
        beam_width = st.slider("Beam width", 1, 20, 8)
        st.caption("Maximum number of candidate trajectories retained at each step.")
        min_score = st.slider("Pruning threshold", 0.0, 0.1, 0.001, 0.001)
        st.caption("Discard trajectories whose model-derived score falls below this threshold.")
        graph_nodes = st.slider("Graph visualization nodes", 10, 80, 40)
        st.caption("Number of highest-activity entities shown visually; computation uses the full graph.")

    try:
        if source_mode == "Upload PCAP / Packet Telemetry":
            if uploaded is None:
                st.info("Upload an offline PCAP to activate packet telemetry mode.")
                return
            if uploaded.size > 500 * 1024 * 1024:
                st.error("Unable to parse PCAP. The uploaded file exceeds the configured 500 MB limit.")
                return
            with st.status("LOADING PCAP telemetry...", expanded=False) as status:
                capture = load_uploaded_pcap(uploaded.getvalue(), uploaded.name)
                capture["dataset_metadata"] = {
                    "dataset_name": uploaded.name, "dataset_family": "Uploaded packet telemetry",
                    "format": Path(uploaded.name).suffix.lstrip(".").lower(), "source_type": "Uploaded packet telemetry",
                }
                status.update(label="READY: PCAP telemetry loaded.", state="complete")
        else:
            if source_mode == "Built-in Benchmark" and selected_dataset is None:
                return
            if source_mode == "Upload CSV / Flow Telemetry" and uploaded is None:
                st.info("Upload CSV flow telemetry to activate this mode.")
                return
            source_name = uploaded.name if uploaded else selected_dataset["dataset_name"]
            with st.status(f"Loading {source_name}...", expanded=False) as status:
                capture = load_uploaded_capture(uploaded.getvalue(), uploaded.name) if uploaded else load_selected_scenario(selected_dataset)
                capture["input_mode"] = "CSV / Flow telemetry"
                capture["dataset_metadata"] = selected_dataset if selected_dataset else {
                    "dataset_name": uploaded.name, "dataset_family": "Uploaded telemetry", "format": "csv", "source_type": "Uploaded flow telemetry"
                }
                status.update(label=f"READY: {source_name} loaded.", state="complete")
    except Exception as exc:
        if source_mode == "Upload PCAP / Packet Telemetry":
            st.error("Unable to parse PCAP. Verify that the file is a supported offline PCAP capture.")
        elif uploaded:
            st.error(f"Invalid telemetry format. {exc}")
        else:
            st.error(f"Unable to load built-in benchmark: {exc}")
        return

    try:
        # The temporal checkpoint is the only model needed to construct H10 input.
        with st.status("Preparing H10 temporal history...", expanded=False) as status:
            temporal_assets = load_temporal_assets()
            X, raw_X, sample_meta = temporal_data_for_capture(capture, temporal_assets["scaler"])
            status.update(label="READY: temporal history prepared.", state="complete")
    except Exception as exc:
        st.error(f"Temporal Model checkpoint unavailable. {exc}")
        st.exception(exc)
        return

    packet_info = packet_summary(capture.get("packet_df", pd.DataFrame()), capture.get("flow_df", pd.DataFrame())) if source_mode == "Upload PCAP / Packet Telemetry" else None
    dataset_metadata = capture.get("dataset_metadata", {})
    timestamps = pd.to_numeric(capture["norm_df"].get("timestamp"), errors="coerce").dropna()
    time_range = "Unavailable"
    if len(timestamps):
        start = pd.to_datetime(float(timestamps.min()), unit="s").strftime("%Y-%m-%d %H:%M")
        end = pd.to_datetime(float(timestamps.max()), unit="s").strftime("%Y-%m-%d %H:%M")
        time_range = f"{start} to {end}"
    st.markdown('<div class="section-rule"></div><h2>Dataset</h2>', unsafe_allow_html=True)
    dataset_cols = st.columns(5)
    dataset_cols[0].metric("Dataset", dataset_metadata.get("dataset_name", capture["name"]))
    dataset_cols[1].metric("Dataset family", dataset_metadata.get("dataset_family", "Unknown"))
    dataset_cols[2].metric("Input format", dataset_metadata.get("format", "flow telemetry"))
    dataset_cols[3].metric("Flows / packets", packet_info["packets"] if packet_info is not None else f"{len(capture['norm_df']):,}")
    dataset_cols[4].metric("Time windows", len(capture["windows"]))
    st.caption(f"Time range: {time_range} · {dataset_metadata.get('source_type', 'Telemetry input')}")
    pipeline_status = [
        ("Input", "COMPLETED"),
        ("Feature Extraction", "COMPLETED" if packet_info is not None else "READY"),
        ("Network State", "COMPLETED"),
        ("Temporal Model", "READY"),
        ("World Model", "ON DEMAND"),
        ("Future Simulation", "READY"),
        ("PTAG + DSA", "READY"),
        ("MITRE / Explainability", "READY"),
    ]
    st.markdown('<div class="section-rule"></div><h2>Pipeline Status</h2>', unsafe_allow_html=True)
    status_cols = st.columns(len(pipeline_status))
    for col, (label, status) in zip(status_cols, pipeline_status):
        color = "#b6e35b" if status in {"READY", "COMPLETED"} else "#f4b860"
        col.markdown(f'<div class="note" style="color:{color};text-align:center">{label.upper()}<br><b>{status}</b></div>', unsafe_allow_html=True)

    if packet_info is not None:
        st.markdown('<div class="section-rule"></div><h2>Packet Telemetry</h2>', unsafe_allow_html=True)
        packet_cols = st.columns(6)
        packet_cols[0].metric("Packets parsed", packet_info["packets"])
        packet_cols[1].metric("Flows / sessions", packet_info["flows"])
        packet_cols[2].metric("TTL mean", f"{packet_info['ttl']['mean']:.2f}" if packet_info["ttl"] else "Unavailable")
        packet_cols[3].metric("TCP window", f"{packet_info['tcp_window']:.2f}" if packet_info["tcp_window"] is not None else "Unavailable")
        packet_cols[4].metric("Payload mean", f"{packet_info['payload']:.2f}" if packet_info["payload"] is not None else "Unavailable")
        packet_cols[5].metric("Port diversity", packet_info["port_diversity"])
        st.markdown(f'<div class="note">TCP flags SYN {packet_info["flags"]["SYN"]} · ACK {packet_info["flags"]["ACK"]} · FIN {packet_info["flags"]["FIN"]} · RST {packet_info["flags"]["RST"]} · IAT mean {packet_info["iat"] if packet_info["iat"] is not None else "Unavailable"} · retransmission heuristic {packet_info["retransmissions"]} · fragments {packet_info["fragmented"]}</div>', unsafe_allow_html=True)

    if len(X) == 0:
        st.warning("Insufficient temporal history for H10 forecasting.")
        st.info("Dataset information, packet/flow summaries, and NetworkState construction are available. Upload a longer capture for temporal inference.")
        return

    st.markdown(f'<div class="note">DATA SOURCE&nbsp;&nbsp; {capture["name"]} &nbsp;·&nbsp; {len(capture["windows"])} windows &nbsp;·&nbsp; INFERENCE ONLY</div>', unsafe_allow_html=True)
    selected = st.slider("Current forecast window", 0, len(X) - 1, len(X) - 1)
    meta = sample_meta[selected]
    anchor_index = meta["anchor_window_idx"]
    window = capture["windows"][anchor_index]
    state = capture["states_lookup"][anchor_index]
    score_series = temporal_scores(temporal_assets["temporal"], X)
    temporal_score = float(score_series[selected])
    try:
        graph_assets = load_graph_assets()
        combined_score = fusion_score({**temporal_assets, **graph_assets}, capture["norm_df"], window, X[selected])
    except Exception as exc:
        st.warning(f"GNN/fusion checkpoint unavailable; using the temporal forecast. {exc}")
        st.exception(exc)
        combined_score = None
    display_score = combined_score if combined_score is not None else temporal_score
    world_parameter_count = None
    try:
        world_assets = load_world_model_assets()
        world_parameter_count = world_assets.get("parameter_count")
        world_result = world_model_result(world_assets, X[selected], horizon, top_k, beam_width, min_score=min_score)
    except Exception as exc:
        st.error(f"World Model checkpoint unavailable. {exc}")
        st.exception(exc)
        world_result = None
    current_features = state.features
    history_features = [capture["states_lookup"][item["anchor_window_idx"]].features for item in sample_meta[max(0, selected - 5):selected]]
    velocity_map = {name: float(raw_X[selected, -1, 16 + index]) for index, name in enumerate(MACRO_FEATURE_NAMES)}
    interpretation = explain_observation(current_features, history_features, velocity=velocity_map, top_k=5)

    st.markdown('<div class="section-rule"></div><h2>Network Overview</h2>', unsafe_allow_html=True)
    current_df = capture["norm_df"].iloc[list(window.flow_indices)] if window.flow_indices else capture["norm_df"].iloc[0:0]
    cols = st.columns(6)
    overview = [("Flows", f"{window.total_flows:,}", "current window"), ("Packets", f"{current_features.get('total_packets', 0.0):,.0f}", "bidirectional"), ("Bytes", f"{current_features.get('total_bytes', 0.0):,.0f}", "bidirectional"), ("Source entities", f"{current_features.get('unique_src_ips', 0.0):,.0f}", "unique hosts"), ("Destination entities", f"{current_features.get('unique_dst_ips', 0.0):,.0f}", "unique hosts"), ("Attack score", f"{display_score:.4f}", "model-derived")]
    for col, (label, value, detail) in zip(cols, overview):
        with col:
            _render_card(label, value, detail)
    protocol_counts = current_df["protocol"].value_counts().reindex(["tcp", "udp", "icmp"], fill_value=0)
    st.markdown(f'<div class="note">PROTOCOL MIX&nbsp;&nbsp; TCP {int(protocol_counts.tcp)} &nbsp;·&nbsp; UDP {int(protocol_counts.udp)} &nbsp;·&nbsp; ICMP {int(protocol_counts.icmp)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    forecast_panel, graph_panel = st.columns([1, 1.35], gap="large")
    with forecast_panel:
        st.markdown('<div class="panel"><div class="panel-title">Attack Forecast</div><div class="panel-kicker">NEXT WINDOW · T+1</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-score">{display_score:.4f}</div><div class="hero-score-label">predicted attack risk · not calibrated probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:1rem;color:#dce8f0;font-weight:600">{"ATTACK-LIKELY" if display_score >= 0.5 else "BENIGN-LIKELY"}</div><div class="note">Current window {anchor_index} → target window {meta["target_window_idx"]}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="margin-top:1rem">Probability Timeline</div>', unsafe_allow_html=True)
        timeline = pd.DataFrame({"window": [item["anchor_window_idx"] for item in sample_meta], "model_score": score_series})
        st.line_chart(timeline.set_index("window"), height=220)
        st.caption("Temporal evolution of model-derived scores; current window is selected in the control room.")
    with graph_panel:
        st.markdown('<div class="panel"><div class="panel-title">Network Topology</div><div class="panel-kicker">DIRECTED COMMUNICATION GRAPH</div>', unsafe_allow_html=True)
        graph = build_communication_graphs(capture["norm_df"], [window])[0]
        dot, shown_nodes, edge_count = graph_dot(graph, graph_nodes)
        st.graphviz_chart(dot, use_container_width=True)
        st.markdown(f'<div class="note">Visualization subset: {shown_nodes} highest-activity nodes from {len(graph["node_ids"]):,} total. {edge_count:,} directed edges available for computation.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div><h2>World Model · Probable Futures</h2>', unsafe_allow_html=True)
    if world_result is not None:
        st.markdown('<div class="note">Learned-model-driven PTAG · deterministic checkpoint rollout · predicted attack risk is not calibrated probability</div>', unsafe_allow_html=True)
        risk_frame = pd.DataFrame({"future_step": list(range(1, horizon + 1)), "predicted_attack_risk": world_result.attack_risk_scores})
        st.line_chart(risk_frame.set_index("future_step"), height=180)
        trajectory_rows = [{"rank": item.rank, "trajectory": "Current → " + " → ".join(item.states), "model_derived_trajectory_score": item.cumulative_probability, "log_score": item.cumulative_log_probability, "horizon": horizon} for item in world_result.trajectories]
        ptag_graph = world_result.graph
        trajectories = world_result.trajectories
        st.markdown(f'<div class="note">Future states: {len(world_result.future_states)} · PTAG nodes: {len(ptag_graph.nodes)} · PTAG edges: {sum(len(edges) for edges in ptag_graph.edges.values())} · Beam width: {beam_width} · Pruning threshold: {min_score}</div>', unsafe_allow_html=True)
        st.caption("Model-derived trajectory score — not a probability. Scores are ranked in log space.")
    else:
        st.warning("Learned World Model checkpoint unavailable; future trajectory output is unavailable.")
        ptag_graph, trajectories = build_trajectory(display_score, horizon, top_k, beam_width)
        trajectory_rows = [{"rank": item.rank, "trajectory": "Current → " + " → ".join(item.states), "model_derived_trajectory_score": item.cumulative_probability, "log_score": item.cumulative_log_probability, "horizon": horizon} for item in trajectories]
        st.caption("Fallback simulation/demo PTAG only; values are not learned probabilities.")
    st.markdown('<h3>Top-K Future Trajectories</h3>', unsafe_allow_html=True)
    if trajectory_rows:
        st.dataframe(pd.DataFrame(trajectory_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No mathematically valid trajectory remained after trajectory search; the PTAG graph is shown above.")
    st.markdown(f'<div class="note">PTAG branch structure: {len(ptag_graph.nodes)} nodes · {sum(len(edges) for edges in ptag_graph.edges.values())} directed edges · horizon {horizon}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div><h2>Driving Indicators</h2>', unsafe_allow_html=True)
    velocity = raw_X[selected, -1, 16:]
    strongest = np.argsort(np.abs(velocity))[::-1][:5]
    indicators = pd.DataFrame({"indicator": [MACRO_FEATURE_NAMES[index] for index in strongest], "change": [float(velocity[index]) for index in strongest]})
    st.dataframe(pd.DataFrame(interpretation["driving_indicators"]), use_container_width=True, hide_index=True)
    st.caption("Important observed changes — not causal explanations.")
    st.markdown('<div class="section-rule"></div><h2>ATT&CK-Aligned Interpretation</h2>', unsafe_allow_html=True)
    tactic = interpretation.get("tactic")
    technique = interpretation.get("technique")
    st.write(f"Predicted behavior: {interpretation['behavior']}")
    st.write(f"ATT&CK tactic: {tactic['name']} ({tactic['id']})" if tactic else "ATT&CK tactic: Insufficient evidence for a defensible ATT&CK mapping")
    st.write(f"Technique candidate: {technique['name']} ({technique['id']})" if technique else "Technique candidate: Insufficient evidence for a defensible ATT&CK mapping")
    st.write(f"Evidence score: {interpretation['evidence_score']:.3f} (heuristic evidence, not calibrated confidence)")
    st.write("Supporting indicators:", interpretation["supporting_indicators"] or ["Insufficient evidence for a defensible ATT&CK mapping"])
    st.caption("ATT&CK-aligned interpretation from observable behavior; not ground-truth stage or causal intent.")
    st.markdown('<div class="section-rule"></div><h2>Security Context</h2>', unsafe_allow_html=True)
    context_left, context_right = st.columns(2, gap="large")
    with context_left:
        st.info("Behavior-to-MITRE mapping: integration point. No defensible learned technique mapping is enabled.")
    with context_right:
        world_parameter_text = f"{world_parameter_count:,} parameters" if world_parameter_count is not None else "checkpoint unavailable"
        st.markdown(f'<div class="panel"><div class="panel-title">Model Stack</div><div class="note" style="line-height:1.9">Temporal Transformer · H10 State + Velocity<br>GraphSAGE · 2 layers · hidden 32<br>Learned Latent World Model · {world_parameter_text}<br>PTAG · learned-model-driven future-state graph<br>Trajectory Search · beam search + pruning</div></div>', unsafe_allow_html=True)
    st.markdown('<footer>Research prototype · CTU-13 demonstration · Forecast scores are not calibrated probabilities</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
