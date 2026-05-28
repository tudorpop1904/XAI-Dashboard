"""
visual_xai.py — CNN Visual PMI + Visual Sobol implementations.
"""

from __future__ import annotations

import numpy as np
import torch


def _apply_cnn_mask(
    x_tensor: torch.Tensor,
    mask: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    fill: float = 0.0,
) -> torch.Tensor:
    """Apply a grid binary mask to x_tensor of shape (1, 1, H, W)."""
    masked = x_tensor.clone()
    _, _, h, w = x_tensor.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    for r in range(grid_rows):
        for c in range(grid_cols):
            if mask[r, c] == 0:
                r0 = r * cell_h
                r1 = (r + 1) * cell_h if r < grid_rows - 1 else h
                c0 = c * cell_w
                c1 = (c + 1) * cell_w if c < grid_cols - 1 else w
                masked[0, 0, r0:r1, c0:c1] = fill
    return masked


def visual_pmi_attribution(
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    target_class: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """
    CNN Visual PMI — "Reveal one region, occlude everything else".
    Measures the log-ratio of target class probability when ONLY a region is visible
    compared to when the entire image is occluded (baseline).
    """
    model.eval()
    x_tensor = x_tensor.to(device)
    model = model.to(device)

    _, _, h, w = x_tensor.shape
    total_cells = grid_rows * grid_cols

    # Get baseline prediction (fully occluded image, filled with 0.0)
    baseline_tensor = torch.full_like(x_tensor, 0.0)
    with torch.no_grad():
        baseline_logits = model(baseline_tensor)
        p_baseline = torch.softmax(baseline_logits, dim=1)[0, target_class].item()

    pmi_scores = np.zeros((grid_rows, grid_cols))

    for idx in range(total_cells):
        r = idx // grid_cols
        c = idx % grid_cols

        mask = np.zeros((grid_rows, grid_cols))
        mask[r, c] = 1

        revealed_tensor = _apply_cnn_mask(x_tensor, mask, grid_rows, grid_cols)
        with torch.no_grad():
            revealed_logits = model(revealed_tensor)
            p_reveal = torch.softmax(revealed_logits, dim=1)[0, target_class].item()

        # PMI: log2((p_reveal + eps) / (p_baseline + eps))
        pmi_scores[r, c] = np.log2((p_reveal + 1e-7) / (p_baseline + 1e-7))

    # Postprocess PMI scores
    pmi_scores = np.clip(pmi_scores, 0.0, None)  # Only keep positive contributions

    smin, smax = pmi_scores.min(), pmi_scores.max()
    if smax - smin > 1e-8:
        pmi_scores = (pmi_scores - smin) / (smax - smin)
    else:
        pmi_scores = np.zeros_like(pmi_scores)

    # Upscale to shape (H, W)
    from PIL import Image as _PILImage
    heatmap_pil = _PILImage.fromarray((pmi_scores * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), _PILImage.BILINEAR)
    heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0
    return heatmap


def visual_sobol_attribution(
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    target_class: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    n_samples: int = 100,
    device: str = "cpu",
) -> np.ndarray:
    """
    CNN Visual Sobol — Monte Carlo variance reduction.
    Samples random binary masks, computes target class prediction, and decomposes
    variance to assign a Sobol sensitivity index to each region.
    """
    model.eval()
    x_tensor = x_tensor.to(device)
    model = model.to(device)

    _, _, h, w = x_tensor.shape
    rng = np.random.default_rng(42)

    masks = []
    predictions = []

    # Sample N random masks and run model
    for _ in range(n_samples):
        mask = rng.binomial(1, 0.5, size=(grid_rows, grid_cols))
        # Ensure at least one active region
        if mask.sum() == 0:
            mask[rng.integers(grid_rows), rng.integers(grid_cols)] = 1

        masked_tensor = _apply_cnn_mask(x_tensor, mask, grid_rows, grid_cols)
        with torch.no_grad():
            logits = model(masked_tensor)
            p = torch.softmax(logits, dim=1)[0, target_class].item()

        masks.append(mask)
        predictions.append(p)

    # Compute Sobol sensitivity indices
    var_total = float(np.var(predictions))
    sobol_grid = np.zeros((grid_rows, grid_cols))

    if var_total > 1e-8:
        for r in range(grid_rows):
            for c in range(grid_cols):
                subset = [p for idx, p in enumerate(predictions) if masks[idx][r, c] == 1]
                if len(subset) > 1:
                    var_cond = float(np.var(subset))
                    sobol_grid[r, c] = max(0.0, (var_total - var_cond) / var_total)
                else:
                    sobol_grid[r, c] = 0.0
    else:
        sobol_grid = np.zeros((grid_rows, grid_cols))

    # Postprocess and interpolate to original H, W
    smin, smax = sobol_grid.min(), sobol_grid.max()
    if smax - smin > 1e-8:
        sobol_grid = (sobol_grid - smin) / (smax - smin)
    else:
        sobol_grid = np.zeros_like(sobol_grid)

    from PIL import Image as _PILImage
    heatmap_pil = _PILImage.fromarray((sobol_grid * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), _PILImage.BILINEAR)
    heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0
    return heatmap
