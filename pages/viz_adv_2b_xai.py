"""
viz_adv_2b_xai.py — Advanced Math Tutor: Occlusion Sensitivity XAI.

This page implements Visual XAI for the VLM transcription step using
Occlusion Sensitivity — a model-agnostic technique that works with any
black-box vision model (including Ollama-served VLMs).

The technique: systematically gray-out regions of the image, re-run
the VLM on each occluded version, and measure how much the transcription
changes.  Regions where masking causes the biggest output degradation
are the most important for the model's reading.

Speed optimisation: by default, a lightweight *surrogate* model
(e.g. minicpm-v) is used for the probing calls.  For XAI, we need
consistency, not accuracy — a fast model that reliably changes its
output when regions are masked produces equally valid heatmaps.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ui.state import init_state, require, nav_buttons
from core.vlm_engine import (
    check_vlm_available,
    check_surrogate_available,
    occlusion_sensitivity,
    XAI_SURROGATE_MODEL,
    VLM_MODEL,
)

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "viz_advanced":
    st.warning("This page is part of the **Advanced Math Tutor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("adv_transcription", "Run VLM transcription first.")

nav_buttons(back_page="pages/viz_adv_2_transcribe.py")

# ---- Header ----
st.header("🌡️ Explain · Occlusion Sensitivity")
st.write(
    "**Occlusion Sensitivity** is a model-agnostic Visual XAI technique. "
    "We systematically mask regions of your image and re-run a VLM to "
    "see which parts are most critical for accurate transcription.\n\n"
    "Bright/warm regions = the model relies heavily on that area. "
    "Dark/cool regions = masking them barely changes the output."
)

st.divider()

# ---- Model status ----
vlm_ok, vlm_msg = check_vlm_available()
surr_ok, surr_msg = check_surrogate_available()

if surr_ok:
    st.success(surr_msg)
elif vlm_ok:
    st.warning(
        f"Surrogate model not available ({surr_msg}). "
        f"Will fall back to `{VLM_MODEL}` (slower)."
    )
else:
    st.error(vlm_msg)

can_run = surr_ok or vlm_ok

# ---- Configuration ----
st.subheader("⚙️ XAI Configuration")

c1, c2 = st.columns(2)
with c1:
    grid_rows = st.slider(
        "Grid rows", min_value=2, max_value=6, value=3,
        help="More rows = finer heatmap but more VLM calls."
    )
with c2:
    grid_cols = st.slider(
        "Grid columns", min_value=2, max_value=6, value=3,
        help="More columns = finer heatmap but more VLM calls."
    )

total_cells = grid_rows * grid_cols

use_surrogate = st.checkbox(
    f"Use fast surrogate model (`{XAI_SURROGATE_MODEL}`) for probing",
    value=surr_ok,
    disabled=not surr_ok,
    help=(
        "For XAI, we only need to detect *change*, not produce accurate "
        "transcriptions. A fast lightweight model produces equally valid "
        "heatmaps in a fraction of the time."
    ),
)

probe_name = XAI_SURROGATE_MODEL if use_surrogate else VLM_MODEL
st.caption(
    f"**{total_cells + (1 if use_surrogate else 0)} calls** to `{probe_name}` "
    f"(+1 surrogate baseline if using surrogate)."
)

# ---- Run button ----
if st.button(
    f"Run Occlusion Sensitivity ({total_cells} cells)",
    type="primary",
    disabled=not can_run,
):
    images = st.session_state["adv_images"]
    baseline = st.session_state["adv_transcription"]

    progress_bar = st.progress(0, text="Starting occlusion analysis…")

    def _progress(step, total):
        pct = step / total
        progress_bar.progress(pct, text=f"Cell {step}/{total} — probing with `{probe_name}`…")

    with st.spinner("Running Occlusion Sensitivity analysis…"):
        result = occlusion_sensitivity(
            images=images,
            baseline_text=baseline,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            use_surrogate=use_surrogate,
            progress_callback=_progress,
        )

    progress_bar.progress(1.0, text="✅ Done!")

    st.session_state["adv_occlusion"] = result
    st.success(
        f"✅ Occlusion analysis complete in {result.elapsed_seconds:.1f}s "
        f"({total_cells} cells, model: `{result.surrogate_model}`)."
    )

# ---- Display results ----
if st.session_state.get("adv_occlusion") is not None:
    result = st.session_state["adv_occlusion"]
    images = st.session_state["adv_images"]

    st.divider()
    st.subheader("Occlusion Sensitivity Heatmap")

    # ---- Build matplotlib figure ----
    img0 = images[0]
    if img0.mode not in ("RGB",):
        img0 = img0.convert("RGB")
    img_arr = np.asarray(img0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap overlay
    axes[1].imshow(img_arr)
    axes[1].imshow(
        result.heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Occlusion Sensitivity Overlay")
    axes[1].axis("off")

    # Raw heatmap
    im = axes[2].imshow(result.heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Importance Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # ---- Interpretation guide ----
    st.markdown(
        "> **How to read this:** Warm/bright regions (red/yellow) are areas where "
        "masking caused the biggest change in the VLM's transcription — meaning "
        "the model relies heavily on those pixels. Cool regions (blue) had little "
        "effect when masked."
    )

    st.caption(f"Probing model: `{result.surrogate_model}`")

    # ---- Per-cell detail ----
    st.divider()
    st.subheader("📊 Per-Cell Similarity Scores")

    # Show grid as a table
    sim_grid = np.array(result.cell_similarities).reshape(
        result.grid_rows, result.grid_cols
    )

    col_labels = [f"Col {c+1}" for c in range(result.grid_cols)]
    row_labels = [f"Row {r+1}" for r in range(result.grid_rows)]

    import pandas as pd
    sim_df = pd.DataFrame(sim_grid, index=row_labels, columns=col_labels)
    st.dataframe(
        sim_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
        .format("{:.3f}"),
        use_container_width=True,
    )
    st.caption(
        "Similarity = how close the occluded transcription is to the original "
        "(1.0 = identical, 0.0 = completely different). "
        "Low similarity = that region is important."
    )

    # ---- Expandable per-cell transcriptions ----
    with st.expander("🔍 View individual occluded transcriptions"):
        for idx, (text, sim) in enumerate(
            zip(result.cell_texts, result.cell_similarities)
        ):
            r = idx // result.grid_cols
            c = idx % result.grid_cols
            importance = 1.0 - sim
            label = "🔴" if importance > 0.5 else "🟡" if importance > 0.2 else "🟢"
            st.markdown(
                f"**Cell ({r+1}, {c+1})** — similarity: {sim:.3f} "
                f"— importance: {importance:.3f} {label}"
            )
            st.text(text[:500] if text else "(empty / VLM returned nothing)")
            st.markdown("---")

    # ---- Metrics ----
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Grid", f"{result.grid_rows}×{result.grid_cols}")
    with c2:
        avg_sim = float(np.mean(result.cell_similarities))
        st.metric("Avg Similarity", f"{avg_sim:.3f}")
    with c3:
        st.metric("XAI Time", f"{result.elapsed_seconds:.1f}s")
    with c4:
        st.metric("Probe Model", result.surrogate_model.split(":")[0])

    # ---- Proceed ----
    st.divider()
    if st.button("Evaluate solution →", type="primary"):
        st.switch_page("pages/viz_adv_3_evaluate.py")
