"""Transparent driving-indicator extraction from observable NetworkState features."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.features.network_state import MACRO_FEATURE_NAMES


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _as_feature_map(values: Mapping[str, Any] | Sequence[float]) -> Dict[str, float]:
    if isinstance(values, Mapping):
        return {name: _finite(values.get(name, 0.0)) for name in MACRO_FEATURE_NAMES if name in values}
    array = np.asarray(values, dtype=float).reshape(-1)
    return {name: _finite(array[index]) for index, name in enumerate(MACRO_FEATURE_NAMES[: len(array)])}


def rank_driving_indicators(
    current: Mapping[str, Any] | Sequence[float],
    history: Iterable[Mapping[str, Any] | Sequence[float]],
    velocity: Mapping[str, Any] | Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rank important observed changes against a recent baseline.

    Importance is a transparent heuristic: absolute change divided by the
    historical standard deviation plus a small numerical floor. It is not a
    trained attribution method and must not be interpreted causally.
    """
    current_map = _as_feature_map(current)
    history_maps = [_as_feature_map(item) for item in history]
    if not history_maps:
        history_maps = [current_map]
    available = [name for name in MACRO_FEATURE_NAMES if name in current_map and any(name in item for item in history_maps)]
    velocity_map = _as_feature_map(velocity) if velocity is not None else {}
    rows = []
    for name in available:
        baseline_values = np.asarray([item.get(name, current_map[name]) for item in history_maps], dtype=float)
        baseline = _finite(np.mean(baseline_values))
        change = _finite(current_map[name] - baseline)
        relative_change = change / max(abs(baseline), 1e-6)
        spread = _finite(np.std(baseline_values), 0.0)
        importance = abs(change) / max(spread, 1e-6)
        if name in velocity_map:
            importance = max(importance, abs(_finite(velocity_map[name])))
        rows.append({
            "feature": name,
            "current_value": _finite(current_map[name]),
            "baseline_value": baseline,
            "change": change,
            "relative_change": _finite(relative_change),
            "direction": "increase" if change > 0 else "decrease" if change < 0 else "unchanged",
            "importance": _finite(importance),
            "available_in_schema": True,
            "interpretation": "important observed change; not a causal explanation",
        })
    rows.sort(key=lambda row: (-row["importance"], row["feature"]))
    return rows[:top_k] if top_k is not None else rows


def summarize_velocity(velocity: Mapping[str, Any] | Sequence[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Return the strongest finite first-difference indicators."""
    values = _as_feature_map(velocity)
    rows = [
        {
            "feature": name,
            "change": _finite(value),
            "direction": "increase" if value > 0 else "decrease" if value < 0 else "unchanged",
            "importance": abs(_finite(value)),
            "interpretation": "velocity indicator; not a causal explanation",
        }
        for name, value in values.items()
    ]
    rows.sort(key=lambda row: (-row["importance"], row["feature"]))
    return rows[:top_k]
