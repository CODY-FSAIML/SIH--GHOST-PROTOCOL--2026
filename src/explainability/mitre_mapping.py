"""Conservative observable-behavior to ATT&CK interpretation mapping.

This module does not infer ground-truth attack stages from CTU-13 labels. It
maps aggregate network observations to candidate ATT&CK context only when the
telemetry supports the interpretation; otherwise it returns insufficient
 evidence. Scores are evidence heuristics, not calibrated confidence.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.features.network_state import MACRO_FEATURE_NAMES
from .feature_explanations import rank_driving_indicators

# Official ATT&CK names/IDs used only for conservative candidate context.
RECONNAISSANCE = ("Reconnaissance", "TA0043")
ACTIVE_SCANNING = ("Active Scanning", "T1595")
DISCOVERY = ("Discovery", "TA0007")
NETWORK_SERVICE_SCANNING = ("Network Service Scanning", "T1046")
COMMAND_AND_CONTROL = ("Command and Control", "TA0011")
EXFILTRATION = ("Exfiltration", "TA0010")


def _value(features: Mapping[str, Any], name: str) -> float:
    try:
        value = float(features.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _change(current: Mapping[str, Any], baseline: Mapping[str, Any] | None, name: str) -> float:
    return _value(current, name) - (_value(baseline or {}, name) if baseline else 0.0)


def _result(behavior: str, tactic: tuple[str, str] | None, technique: tuple[str, str] | None, score: float, indicators: list[str], explanation: str) -> dict[str, Any]:
    return {
        "behavior": behavior,
        "tactic": {"name": tactic[0], "id": tactic[1]} if tactic else None,
        "technique": {"name": technique[0], "id": technique[1]} if technique else None,
        "evidence_score": float(np.clip(score, 0.0, 1.0)),
        "supporting_indicators": indicators,
        "explanation": explanation,
        "score_semantics": "transparent evidence heuristic; not calibrated confidence",
        "mapping_basis": "observable aggregate network behavior only; not CTU-13 ground truth",
    }


def map_observable_behavior(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
    velocity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one conservative behavior/tactic/technique candidate."""
    destination_change = _change(current, baseline, "unique_dst_ips")
    flow_change = _change(current, baseline, "flow_count")
    tcp_change = _change(current, baseline, "tcp_ratio")
    byte_change = _change(current, baseline, "total_bytes")
    outbound_change = _change(current, baseline, "src_bytes_sum")
    destination_entropy_change = _change(current, baseline, "dst_ip_entropy")
    velocity = velocity or {}
    velocity_magnitude = max((abs(_value(velocity, name)) for name in MACRO_FEATURE_NAMES), default=0.0)

    # Aggregate-only telemetry can support a cautious Active Scanning candidate
    # when destination expansion and traffic/transport activity rise together.
    scan_signals = [destination_change > 0, destination_entropy_change > 0, flow_change > 0, tcp_change > 0]
    scan_strength = sum(scan_signals) / len(scan_signals)
    if scan_strength >= 0.75:
        return _result(
            "Reconnaissance-like network expansion",
            RECONNAISSANCE,
            ACTIVE_SCANNING,
            min(1.0, 0.45 + 0.1 * sum(scan_signals)),
            ["destination diversity increased", "destination entropy increased", "flow volume increased", "TCP activity increased"],
            "Observed expansion across destinations with increased aggregate transport activity is consistent with active-scanning-like behavior. Individual ports, SYN flags, and scan ground truth are unavailable.",
        )
    if destination_change > 0 or destination_entropy_change > 0:
        return _result(
            "Reconnaissance-like destination expansion",
            RECONNAISSANCE,
            None,
            0.35,
            ["destination diversity increased" if destination_change > 0 else "destination entropy increased"],
            "Destination diversity changed in an observation window, but the available features do not defensibly identify a specific scanning technique.",
        )
    if outbound_change > 0 and byte_change > 0:
        return _result(
            "Unusual outbound transfer pattern",
            EXFILTRATION,
            None,
            0.3,
            ["source-byte volume increased", "total-byte volume increased"],
            "Outbound byte volume increased relative to the baseline. This is an exfiltration-like indicator only; destinations, payloads, and channel semantics are unavailable.",
        )
    if flow_change > 0 and velocity_magnitude > 0:
        return _result(
            "Elevated communication activity",
            COMMAND_AND_CONTROL,
            None,
            0.25,
            ["flow volume increased", "traffic velocity changed"],
            "Increased communication activity is observable, but the current schema cannot defensibly identify command-and-control protocol or technique.",
        )
    return _result(
        "Insufficient evidence",
        None,
        None,
        0.0,
        [],
        "The available aggregate NetworkState features do not support a defensible ATT&CK tactic or technique candidate for this observation.",
    )


def explain_observation(
    current: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    velocity: Mapping[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    indicators = rank_driving_indicators(current, history, velocity=velocity, top_k=top_k)
    baseline = history[-1] if history else {}
    mapping = map_observable_behavior(current, baseline=baseline, velocity=velocity)
    mapping["driving_indicators"] = indicators
    mapping["velocity_indicators"] = [item for item in indicators if item["feature"] in (velocity or {})]
    return mapping
