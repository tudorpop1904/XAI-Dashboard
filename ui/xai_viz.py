"""xai_viz.py — Heatmap overlay helpers for XAI comparison UI."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from core.image_xai import XAIHeatmapResult


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.45,
) -> Image.Image:
    """Blend heatmap onto RGB image."""
    rgb = base_image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]), Image.BICUBIC)
    base = np.asarray(rgb, dtype=np.float32) / 255.0
    colored = plt.get_cmap(cmap)(heatmap)[..., :3]
    blended = np.clip(base * (1 - alpha) + colored * alpha, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def show_xai_result(result: XAIHeatmapResult, base_image: Image.Image):
    """Display side-by-side original and overlay."""
    overlay = overlay_heatmap(base_image, result.heatmap)
    c1, c2 = st.columns(2)
    with c1:
        st.image(base_image, caption="Original", use_container_width=True)
    with c2:
        st.image(
            overlay,
            caption=f"{result.method} ({result.category})",
            use_container_width=True,
        )
    m = result.metrics
    st.caption(
        f"⏱ {m.elapsed_seconds}s · 💾 {m.peak_memory_mb} MB · "
        f"🔄 {m.forward_passes} forward passes · "
        f"P(target)={result.baseline_prob:.3f}"
    )


def metrics_table(rows: list[dict]) -> None:
    st.dataframe(rows, use_container_width=True, hide_index=True)
