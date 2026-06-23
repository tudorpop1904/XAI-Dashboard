"""
xai_metrics.py — Engineering metrics for XAI method comparison.

Tracks runtime, memory, forward-pass count, and heatmap stability across runs.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class ResourceMetrics:
    elapsed_seconds: float
    peak_memory_mb: float
    forward_passes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def measure_call(fn: Callable[[], Any], forward_passes: int = 0) -> tuple[Any, ResourceMetrics]:
    tracemalloc.start()
    t0 = perf_counter()
    result = fn()
    elapsed = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, ResourceMetrics(
        elapsed_seconds=round(elapsed, 4),
        peak_memory_mb=round(peak / (1024 * 1024), 2),
        forward_passes=forward_passes,
    )


def heatmap_stability(heatmaps: list[np.ndarray]) -> float:
    """
    Mean pairwise Pearson correlation between flattened heatmaps.
    1.0 = identical maps across runs; lower = less stable.
    """
    if len(heatmaps) < 2:
        return 1.0
    flat = [h.astype(np.float64).ravel() for h in heatmaps]
    corrs: list[float] = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            a, b = flat[i], flat[j]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                corrs.append(1.0 if np.allclose(a, b) else 0.0)
            else:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return round(float(np.mean(corrs)), 4)


def fidelity_drop_score(baseline_prob: float, occluded_probs: list[float]) -> float:
    """
    Mean absolute probability drop when regions are occluded.
    Higher = method produces stronger localized drops (proxy for fidelity).
    """
    if not occluded_probs:
        return 0.0
    drops = [max(0.0, baseline_prob - p) for p in occluded_probs]
    return round(float(np.mean(drops)), 6)


def complexity_score(forward_passes: int, grid_cells: int = 0) -> dict[str, int | str]:
    """Theoretical complexity summary for thesis comparison tables."""
    if grid_cells > 0:
        return {
            "forward_passes": forward_passes,
            "asymptotic": f"O({grid_cells})",
            "grid_cells": grid_cells,
        }
    return {"forward_passes": forward_passes, "asymptotic": "O(1)", "grid_cells": 0}
