"""
visual_xai_vlm.py — VLM Visual PMI + Visual Sobol implementations.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from core.vlm_engine import (
    VLM_MODEL,
    XAI_SURROGATE_MODEL,
    OcclusionResult,
    _normalised_levenshtein,
    _transcribe_for_xai,
)


def _reveal_only_region(
    img: Image.Image,
    row: int,
    col: int,
    grid_rows: int,
    grid_cols: int,
    fill: int = 200,
) -> Image.Image:
    """Return a copy of img with ONLY the (row, col) grid cell visible, all else filled with fill."""
    w, h = img.size
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    x0 = col * cell_w
    y0 = row * cell_h
    x1 = x0 + cell_w if col < grid_cols - 1 else w
    y1 = y0 + cell_h if row < grid_rows - 1 else h

    # Create background image filled with gray
    revealed = Image.new(img.mode, img.size, fill)
    # Paste only the region from the original image
    region = img.crop((x0, y0, x1, y1))
    revealed.paste(region, (x0, y0))
    return revealed


def _apply_mask(
    img: Image.Image,
    mask: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    fill: int = 200,
) -> Image.Image:
    """Return a copy of img with regions where mask == 0 filled with fill."""
    w, h = img.size
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    masked = img.copy()
    from PIL import ImageDraw as _IDraw

    draw = _IDraw.Draw(masked)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if mask[r, c] == 0:
                x0 = c * cell_w
                y0 = r * cell_h
                x1 = x0 + cell_w if c < grid_cols - 1 else w
                y1 = y0 + cell_h if r < grid_rows - 1 else h
                draw.rectangle([x0, y0, x1, y1], fill=fill)
    return masked


def vlm_pmi_sensitivity(
    images: list[Image.Image],
    baseline_text: str,
    grid_rows: int = 3,
    grid_cols: int = 3,
    use_surrogate: bool = True,
    progress_callback=None,
) -> OcclusionResult:
    """
    VLM Visual PMI — "Reveal one region, occlude everything else".
    Measures the log-ratio of the transcription similarity when ONLY a region is visible,
    compared to when the ENTIRE image is occluded (baseline).
    """
    total_cells = grid_rows * grid_cols
    img0 = images[0]
    if img0.mode not in ("RGB", "L"):
        img0 = img0.convert("RGB")

    probe_model = XAI_SURROGATE_MODEL if use_surrogate else VLM_MODEL

    if use_surrogate:
        surrogate_baseline = _transcribe_for_xai(images, probe_model)
    else:
        surrogate_baseline = baseline_text

    t0 = time.time()

    # Get transcription for fully occluded image
    fully_occluded_img = Image.new(img0.mode, img0.size, 200)
    fully_occluded_text = _transcribe_for_xai([fully_occluded_img] + images[1:], probe_model)
    fully_occluded_sim = _normalised_levenshtein(surrogate_baseline, fully_occluded_text)

    cell_texts: list[str] = []
    pmi_scores: list[float] = []

    for idx in range(total_cells):
        r = idx // grid_cols
        c = idx % grid_cols
        revealed = _reveal_only_region(img0, r, c, grid_rows, grid_cols)
        query_images = [revealed] + images[1:]
        cell_text = _transcribe_for_xai(query_images, probe_model)

        sim = _normalised_levenshtein(surrogate_baseline, cell_text)
        # Compute PMI: log2((sim + eps) / (baseline_sim + eps))
        pmi = float(np.log2((sim + 1e-5) / (fully_occluded_sim + 1e-5)))
        cell_texts.append(cell_text)
        pmi_scores.append(pmi)

        if progress_callback:
            progress_callback(idx + 1, total_cells)

    elapsed = time.time() - t0

    # Build heatmap
    importance = np.array(pmi_scores).reshape(grid_rows, grid_cols)
    # Clip negative PMI to 0 (meaning no positive information contribution)
    importance = np.clip(importance, 0, None)

    # Normalise to [0, 1]
    imin, imax = float(importance.min()), float(importance.max())
    if imax - imin > 1e-8:
        importance = (importance - imin) / (imax - imin)
    else:
        importance = np.zeros_like(importance)

    # Upscale to image dimensions
    w, h = img0.size
    from PIL import Image as _PILImage

    heatmap_pil = _PILImage.fromarray((importance * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), _PILImage.BILINEAR)
    heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    return OcclusionResult(
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        baseline_text=surrogate_baseline,
        cell_texts=cell_texts,
        cell_similarities=pmi_scores,  # store raw pmi scores here
        surrogate_model=probe_model,
        elapsed_seconds=elapsed,
    )


def vlm_sobol_sensitivity(
    images: list[Image.Image],
    baseline_text: str,
    grid_rows: int = 3,
    grid_cols: int = 3,
    n_samples: int = 100,
    use_surrogate: bool = True,
    progress_callback=None,
) -> OcclusionResult:
    """
    VLM Visual Sobol — Monte Carlo variance reduction.
    Samples random binary masks, computes VLM transcription similarity, and decomposes
    variance to assign a Sobol sensitivity index to each region.
    """
    img0 = images[0]
    if img0.mode not in ("RGB", "L"):
        img0 = img0.convert("RGB")

    probe_model = XAI_SURROGATE_MODEL if use_surrogate else VLM_MODEL

    if use_surrogate:
        surrogate_baseline = _transcribe_for_xai(images, probe_model)
    else:
        surrogate_baseline = baseline_text

    t0 = time.time()

    masks: list[np.ndarray] = []
    similarities: list[float] = []

    # Seed for reproducibility in the Monte Carlo process
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        # 50% probability for each region to be visible (1) or occluded (0)
        mask = rng.binomial(1, 0.5, size=(grid_rows, grid_cols))

        # Ensure mask is not completely empty to avoid VLM reading nothing/errors
        if mask.sum() == 0:
            mask[rng.integers(grid_rows), rng.integers(grid_cols)] = 1

        masked_img = _apply_mask(img0, mask, grid_rows, grid_cols)
        query_images = [masked_img] + images[1:]
        cell_text = _transcribe_for_xai(query_images, probe_model)

        sim = _normalised_levenshtein(surrogate_baseline, cell_text)
        masks.append(mask)
        similarities.append(sim)

        if progress_callback:
            progress_callback(i + 1, n_samples)

    elapsed = time.time() - t0

    # Calculate Sobol indices
    var_total = float(np.var(similarities))
    sobol_grid = np.zeros((grid_rows, grid_cols))

    if var_total > 1e-8:
        for r in range(grid_rows):
            for c in range(grid_cols):
                # Filter similarities where the target region was visible (mask == 1)
                subset = [sims for idx, sims in enumerate(similarities) if masks[idx][r, c] == 1]
                if len(subset) > 1:
                    var_cond = float(np.var(subset))
                    # Sobol main effect index analogue: variance reduction
                    sobol_grid[r, c] = max(0.0, (var_total - var_cond) / var_total)
                else:
                    sobol_grid[r, c] = 0.0
    else:
        sobol_grid = np.zeros((grid_rows, grid_cols))

    # Normalise to [0, 1]
    smin, smax = float(sobol_grid.min()), float(sobol_grid.max())
    if smax - smin > 1e-8:
        sobol_grid = (sobol_grid - smin) / (smax - smin)
    else:
        sobol_grid = np.zeros_like(sobol_grid)

    # Upscale to image dimensions
    w, h = img0.size
    from PIL import Image as _PILImage

    heatmap_pil = _PILImage.fromarray((sobol_grid * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), _PILImage.BILINEAR)
    heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    return OcclusionResult(
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        baseline_text=surrogate_baseline,
        cell_texts=[f"Sample {i}: sim={s:.2f}" for i, s in enumerate(similarities[:10])],  # summary
        cell_similarities=similarities,
        surrogate_model=probe_model,
        elapsed_seconds=elapsed,
    )
