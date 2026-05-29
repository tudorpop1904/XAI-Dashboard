"""
acc_3_xai.py — Accessible Writing Instructor: Grad-CAM & Saliency.

Shows the child (or their instructor) WHERE the AI looked when
recognizing the character — reuses the white-box XAI methods.
"""

import matplotlib.pyplot as plt
import streamlit as st
import torch

from core.accessible_cnn import (
    grad_cam_accessible,
    load_model_from_bytes,
    saliency_accessible,
)
from ui.accessibility import inject_accessible_theme
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "accessible_writing":
    st.warning("This page is part of the **Accessible Writing Instructor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("acc_last_x", "Draw or upload a character first.")

inject_accessible_theme()
nav_buttons(back_page="pages/acc_2_practice.py")

# ---- Header ----
st.header("🌡️ XAI · How Does the AI Read?")
st.write(
    "These heatmaps show **which parts of your drawing** the AI focused on "
    "when deciding what character you wrote. Bright/warm areas = important strokes."
)

st.divider()

# ---- Load model and data ----
label_map = st.session_state["acc_label_map"]
num_classes = st.session_state["acc_num_classes"]
device = st.session_state.get("acc_device", "cpu")
model = load_model_from_bytes(st.session_state["acc_model_state"], num_classes, device)

x_np = st.session_state["acc_last_x"]
x_tensor = torch.from_numpy(x_np)
pred_class = st.session_state["acc_last_pred_class"]
pred_char = st.session_state["acc_last_char"]

st.info(f"🔤 Recognized character: **{pred_char}** (class {pred_class})")

# ---- Generate XAI maps ----
if st.button("Generate Grad-CAM & Saliency maps", type="primary"):
    with st.spinner("Computing XAI visualizations…"):
        cam = grad_cam_accessible(model, x_tensor, pred_class, device=device)
        sal = saliency_accessible(model, x_tensor, pred_class, device=device)

    st.session_state["acc_grad_cam"] = cam
    st.session_state["acc_saliency"] = sal
    st.success("✅ XAI maps ready!")

# ---- Display ----
if st.session_state.get("acc_grad_cam") is not None:
    cam = st.session_state["acc_grad_cam"]
    sal = st.session_state["acc_saliency"]
    img = x_np[0, 0]  # (28, 28)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Your Drawing: '{pred_char}'", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Overlay", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Saliency overlay
    axes[2].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(sal, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Saliency Map", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Raw Grad-CAM
    im = axes[3].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Grad-CAM (raw)", fontsize=14, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # ---- Explanation ----
    st.markdown(
        "> **Grad-CAM** highlights the regions the CNN's convolutional layers "
        "focused on — typically the most distinctive strokes of the character.\n\n"
        "> **Saliency** shows which individual pixels, if changed, would most "
        "affect the prediction — the fine edges and stroke boundaries."
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Back to practice", use_container_width=True):
            st.switch_page("pages/acc_2_practice.py")
    with c2:
        if st.button("📝 Get writing exercises →", type="primary", use_container_width=True):
            st.switch_page("pages/acc_4_tutor.py")
