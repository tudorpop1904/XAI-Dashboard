"""
viz_4_explain.py — Grad-CAM and saliency map overlays for the last prediction image.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from core.viz_math_cnn import (
    grad_cam_for_image,
    input_saliency,
    load_model_from_bytes,
)
from ui.state import init_state, nav_buttons, require

init_state()

require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "visualization":
    st.warning("This page is part of the Visualization flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("viz_model_state", "Train the model first.")
require("viz_last_x_np", "Run a prediction on your photo first.")

nav_buttons(back_page="pages/viz_3_predict.py")

st.header("💡 Explain · Grad-CAM & Saliency")
st.write(
    "**Grad-CAM** highlights regions of the last convolutional feature map that "
    "support the predicted class. **Saliency** (input-gradient magnitude) shows "
    "which input pixels most sensitively affect the logit."
)

st.divider()

cm = st.session_state["viz_class_map"]
n_cls = len(cm)
device = st.session_state.get("viz_device") or ("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_bytes(st.session_state["viz_model_state"], n_cls, device)
x_np = st.session_state["viz_last_x_np"]
pred_cls = int(st.session_state["viz_last_pred_class"])
x = torch.from_numpy(x_np).float()

img_size = int(x.shape[-1])
h, w = img_size, img_size

with st.spinner("Computing Grad-CAM and saliency…"):
    cam = grad_cam_for_image(model, x, pred_cls, (h, w), device=device)
    sal = input_saliency(model, x, pred_cls, device=device)

plane = np.clip(x_np[0, 0], 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].imshow(plane, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input")
axes[0].axis("off")

axes[1].imshow(plane, cmap="gray", vmin=0, vmax=1)
axes[1].imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
axes[1].set_title("Grad-CAM")
axes[1].axis("off")

axes[2].imshow(plane, cmap="gray", vmin=0, vmax=1)
axes[2].imshow(sal, cmap="magma", alpha=0.42, vmin=0, vmax=1)
axes[2].set_title("Saliency (|∂logit/∂input|)")
axes[2].axis("off")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

st.caption(f"Target class index: **{pred_cls}** (`{st.session_state.get('viz_last_expression')}`)")

st.divider()

if st.button("Tutor interpretation (LLM) →", type="primary"):
    st.switch_page("pages/viz_5_counselor.py")
