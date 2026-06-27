"""4_xai_compare.py — Run and compare all visual XAI methods on the detection."""

import streamlit as st

from core.fake_detector import load_model_from_bytes
from core.image_xai import BLACK_BOX_METHODS, WHITE_BOX_METHODS, run_method
from core.xai_metrics import complexity_score, heatmap_stability
from ui.state import init_state, nav_buttons, require
from ui.xai_viz import metrics_table, show_xai_result

init_state()
require("detection_result", "Upload an image and run detection first.")
require("input_tensor", "Missing input tensor — re-upload the image.")
nav_buttons(back_page="pages/3_upload.py")

st.header("🌡️ Compare XAI Methods")
det = st.session_state["detection_result"]
st.info(f"Explaining prediction: **{det.label}** (class {det.class_index}, {det.confidence:.1%})")

st.subheader("Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    grid_rows = st.slider("Grid rows", 2, 8, 4)
with c2:
    grid_cols = st.slider("Grid cols", 2, 8, 4)
with c3:
    sobol_samples = st.slider("Sobol samples", 16, 200, 64)

stability_runs = st.slider(
    "Stability runs (black-box only)",
    1,
    5,
    2,
    help="Re-run stochastic methods to measure heatmap stability.",
)

use_blackbox = st.multiselect(
    "Black-box methods",
    list(BLACK_BOX_METHODS),
    default=list(BLACK_BOX_METHODS),
)
use_whitebox = st.multiselect(
    "White-box methods (same CNN)",
    list(WHITE_BOX_METHODS),
    default=list(WHITE_BOX_METHODS),
)
methods = use_blackbox + use_whitebox

if not methods:
    st.warning("Select at least one method.")
    st.stop()

total_bb_passes = grid_rows * grid_cols * len(use_blackbox)
if "Visual Sobol" in use_blackbox:
    total_bb_passes += sobol_samples - grid_rows * grid_cols
st.caption(f"Estimated forward passes (black-box): ~{total_bb_passes + len(use_blackbox)}")

if st.button("Run XAI comparison", type="primary"):
    device = st.session_state["detector_device"]
    features = st.session_state.get("detector_features", {})
    model = load_model_from_bytes(
        st.session_state["detector_model_state"],
        device,
        add_fft=features.get("add_fft", True),
        add_lbp=features.get("add_lbp", True),
        add_magnitude=features.get("add_magnitude", True),
    )
    x = st.session_state["input_tensor"]
    target = det.class_index
    img = st.session_state["uploaded_image"]

    results = {}
    stability_map = {}

    progress = st.progress(0, text="Starting…")
    for i, method in enumerate(methods):
        progress.progress((i) / len(methods), text=f"Running {method}…")
        if method in BLACK_BOX_METHODS and stability_runs > 1:
            heatmaps = []
            last = None
            for run_idx in range(stability_runs):
                last = run_method(
                    method,
                    model,
                    x,
                    target,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    n_sobol_samples=sobol_samples,
                    device=device,
                )
                heatmaps.append(last.heatmap)
            results[method] = last
            stability_map[method] = heatmap_stability(heatmaps)
        else:
            results[method] = run_method(
                method,
                model,
                x,
                target,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                n_sobol_samples=sobol_samples,
                device=device,
            )
            stability_map[method] = 1.0

    progress.progress(1.0, text="Done!")
    st.session_state["xai_results"] = results
    st.session_state["xai_stability"] = stability_map
    st.session_state["xai_config"] = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "sobol_samples": sobol_samples,
        "methods": methods,
    }

if st.session_state.get("xai_results"):
    results = st.session_state["xai_results"]
    stability_map = st.session_state.get("xai_stability", {})
    img = st.session_state["uploaded_image"]

    st.divider()
    st.subheader("Engineering comparison")

    rows = []
    for name, res in results.items():
        cells = res.grid_rows * res.grid_cols if res.category == "black-box" else 0
        cx = complexity_score(res.metrics.forward_passes, cells)
        rows.append(
            {
                "Method": name,
                "Category": res.category,
                "Time (s)": res.metrics.elapsed_seconds,
                "Peak RAM (MB)": res.metrics.peak_memory_mb,
                "Forward passes": res.metrics.forward_passes,
                "Complexity": cx["asymptotic"],
                "Stability": stability_map.get(name, "—"),
            }
        )
    metrics_table(rows)

    st.subheader("Heatmap overlays")
    for name, res in results.items():
        with st.expander(f"{name} ({res.category})", expanded=True):
            show_xai_result(res, img)

    st.subheader("Forensic feature visualization")
    features = st.session_state.get("detector_features", {})
    from core.image_features import fft_channel, lbp_channel, magnitude_channel

    x_tensor = st.session_state["input_tensor"]
    fc1, fc2, fc3 = st.columns(3)
    if features.get("add_fft", True):
        with fc1:
            st.write("**FFT Magnitude**")
            st.image(fft_channel(x_tensor[0]).squeeze(0).cpu().numpy(), use_container_width=True, clamp=True)
    if features.get("add_lbp", True):
        with fc2:
            st.write("**LBP Texture**")
            st.image(lbp_channel(x_tensor[0]).squeeze(0).cpu().numpy(), use_container_width=True, clamp=True)
    if features.get("add_magnitude", True):
        with fc3:
            st.write("**Gradient Magnitude**")
            st.image(magnitude_channel(x_tensor[0]).squeeze(0).cpu().numpy(), use_container_width=True, clamp=True)

    if st.button("Generate report →", type="primary"):
        st.switch_page("pages/5_report.py")
