"""
acc_1_setup.py — Accessible Writing Instructor: Setup & Training.

Downloads EMNIST dataset and trains the character recognition CNN.
"""

import streamlit as st
import torch

from ui.state import init_state, require, nav_buttons
from ui.accessibility import inject_accessible_theme, accessible_metric
from core.accessible_cnn import (
    load_emnist_bundle,
    train_accessible_model,
    model_state_bytes,
    NUM_CLASSES,
)

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "accessible_writing":
    st.warning("This page is part of the **Accessible Writing Instructor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

inject_accessible_theme()
nav_buttons(back_page="pages/1_home.py")

# ---- Header ----
st.header("✍️ Writing Instructor · Setup")
st.write(
    "We'll download real handwriting samples (EMNIST — 62 characters: "
    "letters A–Z, a–z, and digits 0–9) and train a small neural network "
    "to recognize your handwriting."
)

st.divider()

# ---- Configuration ----
c1, c2 = st.columns(2)
with c1:
    max_per_class = st.slider(
        "Samples per character",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="More samples = better accuracy but slower training.",
    )
with c2:
    epochs = st.slider(
        "Training epochs",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"🖥️ Device: **{device}** · 📊 Classes: **{NUM_CLASSES}**")

# ---- Load + Train ----
if st.button("Download data & train model", type="primary"):
    with st.spinner("Downloading EMNIST dataset (first time may take a minute)…"):
        bundle = load_emnist_bundle(max_per_class=max_per_class)

    st.success(
        f"✅ Loaded {len(bundle.train_labels)} training + "
        f"{len(bundle.val_labels)} validation images."
    )

    progress_bar = st.progress(0, text="Training…")

    def _progress(ep, total, loss, acc):
        pct = ep / total
        progress_bar.progress(pct, text=f"Epoch {ep}/{total} — loss: {loss:.4f} — val acc: {acc:.1%}")

    with st.spinner("Training CNN…"):
        model, val_acc, history = train_accessible_model(
            bundle,
            epochs=epochs,
            device=device,
            progress_callback=_progress,
        )

    progress_bar.progress(1.0, text="✅ Training complete!")

    # Save to session state
    st.session_state["acc_model_state"] = model_state_bytes(model)
    st.session_state["acc_label_map"] = bundle.label_map
    st.session_state["acc_num_classes"] = bundle.num_classes
    st.session_state["acc_val_accuracy"] = val_acc
    st.session_state["acc_train_history"] = history
    st.session_state["acc_device"] = device
    st.session_state["acc_attempts"] = 0
    st.session_state["acc_correct"] = 0

# ---- Results ----
if st.session_state.get("acc_model_state") is not None:
    st.divider()
    st.subheader("Model Ready")

    c1, c2, c3 = st.columns(3)
    with c1:
        accessible_metric("Classes", str(st.session_state.get("acc_num_classes", 62)))
    with c2:
        acc = st.session_state.get("acc_val_accuracy", 0)
        accessible_metric("Accuracy", f"{acc:.1%}")
    with c3:
        accessible_metric("Device", st.session_state.get("acc_device", "cpu"))

    if st.button("Start practicing →", type="primary"):
        st.switch_page("pages/acc_2_practice.py")
