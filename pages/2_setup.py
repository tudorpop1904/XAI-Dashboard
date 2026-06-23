"""2_setup.py — Train or reload the Real vs AI-generated detector CNN."""

import streamlit as st
import torch

from core.cache import save_model_cache, load_model_cache
from core.fake_data import CLASS_NAMES, generate_synthetic_bundle
from core.fake_detector import model_state_bytes, train_detector
from ui.state import init_state, nav_buttons, reset_detector

init_state()
nav_buttons(back_page="pages/1_home.py")

st.header("🧪 Train Detector")
st.write(
    "Train a lightweight CNN on **synthetic Real vs AI-generated** image pairs. "
    "For production forensic accuracy, fine-tune on CiFAKE or similar datasets."
)

img_size = st.slider("Image size (px)", 64, 224, st.session_state["detector_img_size"], step=32)
samples = st.slider("Samples per class", 50, 500, 200, step=50)
epochs = st.slider("Training epochs", 3, 30, 10)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device: `{device}`")

c1, c2 = st.columns(2)

with c1:
    if st.button("Train new model", type="primary", use_container_width=True):
        reset_detector()
        with st.spinner("Generating synthetic data and training…"):
            bundle = generate_synthetic_bundle(
                samples_per_class=samples,
                img_size=img_size,
            )
            model, acc, history = train_detector(
                bundle, epochs=epochs, device=device
            )
            state = model_state_bytes(model)
            st.session_state["detector_model_state"] = state
            st.session_state["detector_val_accuracy"] = acc
            st.session_state["detector_train_history"] = history
            st.session_state["detector_img_size"] = img_size
            st.session_state["detector_device"] = device
            save_model_cache("fake_detector", state)
        st.success(f"Training complete — validation accuracy: **{acc:.1%}**")
        st.line_chart(history["loss"])

with c2:
    if st.button("Load cached model", use_container_width=True):
        cached = load_model_cache("fake_detector")
        if cached:
            st.session_state["detector_model_state"] = cached
            st.session_state["detector_device"] = device
            st.success("Loaded detector from disk cache.")
        else:
            st.warning("No cached model found. Train first.")

if st.session_state.get("detector_model_state"):
    st.divider()
    acc = st.session_state.get("detector_val_accuracy")
    if acc is not None:
        st.info(f"Model ready · classes: {', '.join(CLASS_NAMES)} · val accuracy: {acc:.1%}")
    else:
        st.info(f"Model ready · classes: {', '.join(CLASS_NAMES)}")
    if st.button("Continue to Upload →", type="primary"):
        st.switch_page("pages/3_upload.py")
