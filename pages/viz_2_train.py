"""
viz_2_train.py — Train the handwriting CNN on the prepared synthetic bundle.
"""

import streamlit as st
import torch

from core.viz_math_cnn import (
    VizMathBundle,
    model_state_bytes,
    train_model,
)
from ui.state import init_state, nav_buttons, require

init_state()

require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "visualization":
    st.warning("This page is part of the Visualization flow. Go back to Home.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("viz_train_x_np", "Prepare the dataset on the previous step first.")

nav_buttons(back_page="pages/viz_1_data.py")

st.header("🧪 Train · Handwriting CNN")
st.write("Train a small convolutional network on your synthetic expression images.")

st.divider()

epochs = st.slider("Epochs", 4, 40, 14)
batch_size = st.slider("Batch size", 16, 256, 64, 16)
lr = st.number_input("Learning rate", min_value=1e-5, max_value=0.1, value=1e-3, format="%.5f")

if st.button("Train model", type="primary"):
    cm = st.session_state["viz_class_map"]
    img_size = int(st.session_state["viz_img_size"])
    bundle = VizMathBundle(
        class_map=cm,
        img_size=img_size,
        train_images=torch.from_numpy(st.session_state["viz_train_x_np"]),
        train_labels=torch.from_numpy(st.session_state["viz_train_y_np"]),
        val_images=torch.from_numpy(st.session_state["viz_val_x_np"]),
        val_labels=torch.from_numpy(st.session_state["viz_val_y_np"]),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Training on **{device}** (this may take a minute)…"):
        model, val_acc, history = train_model(
            bundle,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            device=device,
        )

    st.session_state["viz_model_state"] = model_state_bytes(model)
    st.session_state["viz_val_accuracy"] = float(val_acc)
    st.session_state["viz_train_history"] = history
    st.session_state["viz_device"] = device

    st.success(f"✅ Training complete. Validation accuracy: **{val_acc:.1%}**")
    if history.get("loss"):
        st.line_chart(history["loss"])

if st.session_state.get("viz_model_state") is not None:
    st.divider()
    st.caption(f"Validation accuracy: {st.session_state['viz_val_accuracy']:.1%}")
    if st.button("Predict from your photo →", type="primary"):
        st.switch_page("pages/viz_3_predict.py")
