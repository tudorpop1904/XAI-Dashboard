"""
image_xai.py — Unified visual XAI for AI-generated image detection.

Black-box: Occlusion Sensitivity, Visual PMI, Visual Sobol.
White-box (same CNN): Grad-CAM, Saliency — included as comparison baselines.

All methods score regions by their influence on P(target_class | image).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from core.fake_detector import FakeDetectorCNN, grad_cam, saliency, target_probability
from core.xai_metrics import ResourceMetrics, measure_call

XAIMethod = Literal[
    "Occlusion",
    "Visual PMI",
    "Visual Sobol",
    "Grad-CAM",
    "Saliency",
]

BLACK_BOX_METHODS = ("Occlusion", "Visual PMI", "Visual Sobol")
WHITE_BOX_METHODS = ("Grad-CAM", "Saliency")


@dataclass
class XAIHeatmapResult:
    method: str
    heatmap: np.ndarray
    grid_rows: int
    grid_cols: int
    target_class: int
    baseline_prob: float
    metrics: ResourceMetrics
    category: str  # "black-box" or "white-box"
    cell_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _apply_grid_mask(
    x_tensor: torch.Tensor,
    mask: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    fill: float,
    mode: str,
) -> torch.Tensor:
    """
    mode='occlude': zero out cells where mask==0 (keep visible where mask==1).
    mode='reveal':  keep only cells where mask==1.
    """
    masked = x_tensor.clone()
    _, c, h, w = x_tensor.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    for r in range(grid_rows):
        for c_idx in range(grid_cols):
            active = mask[r, c_idx] == 1
            if (mode == "occlude" and active) or (mode == "reveal" and not active):
                continue
            r0 = r * cell_h
            r1 = (r + 1) * cell_h if r < grid_rows - 1 else h
            c0 = c_idx * cell_w
            c1 = (c_idx + 1) * cell_w if c_idx < grid_cols - 1 else w
            masked[:, :, r0:r1, c0:c1] = fill
    return masked


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    smin, smax = float(grid.min()), float(grid.max())
    if smax - smin > 1e-8:
        return (grid - smin) / (smax - smin)
    return np.zeros_like(grid)


def _upscale_grid(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    heatmap_pil = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), Image.BILINEAR)
    return np.asarray(heatmap_pil, dtype=np.float32) / 255.0


def occlusion_sensitivity(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    device: str = "cpu",
    fill: float = 0.5,
) -> XAIHeatmapResult:
    """Occlude each grid cell; importance = drop in target-class probability."""
    model.eval()
    x_tensor = x_tensor.to(device)
    model = model.to(device)
    _, _, h, w = x_tensor.shape
    total = grid_rows * grid_cols

    baseline = target_probability(model, x_tensor, target_class, device)
    scores = np.zeros((grid_rows, grid_cols))
    cell_scores: list[float] = []

    for idx in range(total):
        r, c = idx // grid_cols, idx % grid_cols
        mask = np.ones((grid_rows, grid_cols), dtype=np.int8)
        mask[r, c] = 0
        occluded = _apply_grid_mask(x_tensor, mask, grid_rows, grid_cols, fill, mode="occlude")
        p_occ = target_probability(model, occluded, target_class, device)
        drop = max(0.0, baseline - p_occ)
        scores[r, c] = drop
        cell_scores.append(drop)

    grid = _normalize_grid(scores)
    heatmap = _upscale_grid(grid, h, w)

    return XAIHeatmapResult(
        method="Occlusion",
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_class=target_class,
        baseline_prob=baseline,
        metrics=ResourceMetrics(
            elapsed_seconds=0.0,
            peak_memory_mb=0.0,
            forward_passes=total + 1,
        ),
        category="black-box",
        cell_scores=cell_scores,
        metadata={"fill_value": fill},
    )


def visual_pmi(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    device: str = "cpu",
) -> XAIHeatmapResult:
    """Reveal one region at a time; PMI = log2(P(reveal) / P(baseline occluded))."""
    model.eval()
    x_tensor = x_tensor.to(device)
    model = model.to(device)
    _, _, h, w = x_tensor.shape
    total = grid_rows * grid_cols

    baseline_tensor = torch.full_like(x_tensor, 0.5)
    p_baseline = target_probability(model, baseline_tensor, target_class, device)
    pmi_scores = np.zeros((grid_rows, grid_cols))
    cell_scores: list[float] = []

    for idx in range(total):
        r, c = idx // grid_cols, idx % grid_cols
        mask = np.zeros((grid_rows, grid_cols), dtype=np.int8)
        mask[r, c] = 1
        revealed = _apply_grid_mask(x_tensor, mask, grid_rows, grid_cols, 0.5, mode="reveal")
        p_reveal = target_probability(model, revealed, target_class, device)
        ratio = (p_reveal + 1e-7) / (p_baseline + 1e-7)
        score = float(np.log2(ratio))
        pmi_scores[r, c] = max(0.0, score)
        cell_scores.append(score)

    grid = _normalize_grid(pmi_scores)
    heatmap = _upscale_grid(grid, h, w)

    return XAIHeatmapResult(
        method="Visual PMI",
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_class=target_class,
        baseline_prob=p_baseline,
        metrics=ResourceMetrics(elapsed_seconds=0.0, peak_memory_mb=0.0, forward_passes=total + 1),
        category="black-box",
        cell_scores=cell_scores,
    )


def visual_sobol(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    n_samples: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> XAIHeatmapResult:
    """Monte Carlo mask sampling; Sobol-like variance reduction per cell."""
    model.eval()
    x_tensor = x_tensor.to(device)
    model = model.to(device)
    _, _, h, w = x_tensor.shape
    rng = np.random.default_rng(seed)

    masks: list[np.ndarray] = []
    predictions: list[float] = []

    for _ in range(n_samples):
        mask = rng.binomial(1, 0.5, size=(grid_rows, grid_cols)).astype(np.int8)
        if mask.sum() == 0:
            mask[rng.integers(grid_rows), rng.integers(grid_cols)] = 1
        masked = _apply_grid_mask(x_tensor, mask, grid_rows, grid_cols, 0.5, mode="occlude")
        predictions.append(target_probability(model, masked, target_class, device))
        masks.append(mask)

    var_total = float(np.var(predictions))
    sobol_grid = np.zeros((grid_rows, grid_cols))
    cell_scores: list[float] = []

    if var_total > 1e-8:
        for r in range(grid_rows):
            for c in range(grid_cols):
                subset = [p for idx, p in enumerate(predictions) if masks[idx][r, c] == 1]
                if len(subset) > 1:
                    var_cond = float(np.var(subset))
                    score = max(0.0, (var_total - var_cond) / var_total)
                else:
                    score = 0.0
                sobol_grid[r, c] = score
                cell_scores.append(score)
    else:
        cell_scores = [0.0] * (grid_rows * grid_cols)

    baseline = target_probability(model, x_tensor, target_class, device)
    grid = _normalize_grid(sobol_grid)
    heatmap = _upscale_grid(grid, h, w)

    return XAIHeatmapResult(
        method="Visual Sobol",
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_class=target_class,
        baseline_prob=baseline,
        metrics=ResourceMetrics(
            elapsed_seconds=0.0,
            peak_memory_mb=0.0,
            forward_passes=n_samples,
            extra={"n_samples": n_samples, "seed": seed},
        ),
        category="black-box",
        cell_scores=cell_scores,
    )


def run_grad_cam(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    img_hw: tuple[int, int],
    device: str = "cpu",
) -> XAIHeatmapResult:
    def _run():
        return grad_cam(model, x_tensor, target_class, img_hw, device)

    heatmap, metrics = measure_call(_run, forward_passes=2)
    baseline = target_probability(model, x_tensor, target_class, device)
    return XAIHeatmapResult(
        method="Grad-CAM",
        heatmap=heatmap,
        grid_rows=1,
        grid_cols=1,
        target_class=target_class,
        baseline_prob=baseline,
        metrics=metrics,
        category="white-box",
    )


def run_saliency(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    device: str = "cpu",
) -> XAIHeatmapResult:
    def _run():
        return saliency(model, x_tensor, target_class, device)

    heatmap, metrics = measure_call(_run, forward_passes=2)
    baseline = target_probability(model, x_tensor, target_class, device)
    return XAIHeatmapResult(
        method="Saliency",
        heatmap=heatmap,
        grid_rows=1,
        grid_cols=1,
        target_class=target_class,
        baseline_prob=baseline,
        metrics=metrics,
        category="white-box",
    )


def _measure_blackbox(fn) -> XAIHeatmapResult:
    result, metrics = measure_call(fn, forward_passes=0)
    result.metrics = ResourceMetrics(
        elapsed_seconds=metrics.elapsed_seconds,
        peak_memory_mb=metrics.peak_memory_mb,
        forward_passes=result.metrics.forward_passes,
        extra=result.metrics.extra,
    )
    return result


def run_method(
    method: XAIMethod,
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    *,
    grid_rows: int = 4,
    grid_cols: int = 4,
    n_sobol_samples: int = 64,
    device: str = "cpu",
) -> XAIHeatmapResult:
    _, _, h, w = x_tensor.shape
    if method == "Occlusion":
        return _measure_blackbox(
            lambda: occlusion_sensitivity(
                model, x_tensor, target_class, grid_rows, grid_cols, device
            )
        )
    if method == "Visual PMI":
        return _measure_blackbox(
            lambda: visual_pmi(model, x_tensor, target_class, grid_rows, grid_cols, device)
        )
    if method == "Visual Sobol":
        return _measure_blackbox(
            lambda: visual_sobol(
                model, x_tensor, target_class, grid_rows, grid_cols, n_sobol_samples, device
            )
        )
    if method == "Grad-CAM":
        return run_grad_cam(model, x_tensor, target_class, (h, w), device)
    if method == "Saliency":
        return run_saliency(model, x_tensor, target_class, device)
    raise ValueError(f"Unknown XAI method: {method}")


def run_all_methods(
    model: FakeDetectorCNN,
    x_tensor: torch.Tensor,
    target_class: int,
    methods: list[XAIMethod] | None = None,
    **kwargs,
) -> dict[str, XAIHeatmapResult]:
    if methods is None:
        methods = list(BLACK_BOX_METHODS) + list(WHITE_BOX_METHODS)
    return {m: run_method(m, model, x_tensor, target_class, **kwargs) for m in methods}
