"""
viz_1_data.py — Visualization XAI: synthetic handwritten equation dataset.

Builds a reproducible synthetic image set (finite expression classes) for CNN training.
"""

import numpy as np
import streamlit as st

from core.viz_math_cnn import generate_synthetic_bundle
from ui.state import init_state, nav_buttons, require

init_state()

require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "visualization":
    st.warning("This wizard is for **Visualization XAI**. Return to Home and choose “Photomath-style” there.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

nav_buttons(back_page="pages/1_home.py")

st.header("📷 Dataset · Handwritten math (demo)")
st.write(
    "We generate **synthetic** images of simple expressions (digits and + − ×). "
    "A CNN classifies *which* expression was written; the app then shows the "
    "**numeric answer** for that expression (closed vocabulary — like a tutor "
    "flashcard deck, not full arbitrary LaTeX)."
)

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    num_classes = st.slider("Number of expression classes", 8, 40, 24, 1)
with c2:
    samples_per_class = st.slider("Samples per class", 40, 200, 90, 10)
with c3:
    img_size = st.select_slider("Image size (px square)", options=[48, 56, 64, 72], value=56)

seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

if st.button("Prepare dataset", type="primary"):
    with st.spinner("Generating synthetic images…"):
        bundle = generate_synthetic_bundle(
            num_classes=int(num_classes),
            samples_per_class=int(samples_per_class),
            img_size=int(img_size),
            val_fraction=0.12,
            seed=int(seed),
        )

    st.session_state["viz_class_map"] = bundle.class_map
    st.session_state["viz_img_size"] = bundle.img_size
    st.session_state["viz_train_x_np"] = bundle.train_images.numpy()
    st.session_state["viz_train_y_np"] = bundle.train_labels.numpy()
    st.session_state["viz_val_x_np"] = bundle.val_images.numpy()
    st.session_state["viz_val_y_np"] = bundle.val_labels.numpy()
    st.session_state["viz_bundle_meta"] = {
        "num_classes": len(bundle.class_map),
        "n_train": int(bundle.train_labels.shape[0]),
        "n_val": int(bundle.val_labels.shape[0]),
        "seed": int(seed),
    }
    st.session_state["viz_model_state"] = None
    st.session_state["viz_val_accuracy"] = None
    st.session_state["viz_train_history"] = None
    st.session_state["viz_last_x_np"] = None
    st.session_state["viz_last_pred_class"] = None
    st.session_state["viz_last_expression"] = None
    st.session_state["viz_last_answer"] = None
    st.session_state["viz_user_correction"] = None
    st.session_state["viz_llm_note"] = None

    st.success("✅ Dataset ready.")
    st.caption(
        f"{len(bundle.class_map)} classes · "
        f"{bundle.train_labels.shape[0]} train · "
        f"{bundle.val_labels.shape[0]} validation images"
    )

if st.session_state.get("viz_train_x_np") is not None:
    st.divider()
    st.subheader("Preview")

    x = st.session_state["viz_train_x_np"]
    y = st.session_state["viz_train_y_np"]
    cmap = st.session_state["viz_class_map"]
    idx = 0
    st.image(
        np.clip(x[idx, 0], 0, 1),
        caption=f"Example label {y[idx]} → `{cmap[int(y[idx])][0]}` = {cmap[int(y[idx])][1]}",
        width=220,
    )

    if st.button("Train CNN →", type="primary"):
        st.switch_page("pages/viz_2_train.py")
