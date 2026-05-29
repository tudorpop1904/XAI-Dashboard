"""
acc_2_practice.py — Accessible Writing Instructor: Drawing & Recognition.

The child draws a character on the canvas (or uploads a photo).
The CNN recognizes it and speaks the result aloud via TTS.
"""

import io

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from ui.state import init_state, require, nav_buttons
from ui.accessibility import (
    inject_accessible_theme,
    show_big_character,
    show_result,
    accessible_metric,
)
from core.accessible_cnn import (
    load_model_from_bytes,
    canvas_to_tensor,
    pil_to_emnist_tensor,
    predict_character,
)
from core.tts import speak, speak_character_result

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "accessible_writing":
    st.warning("This page is part of the **Accessible Writing Instructor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("acc_model_state", "Train the model first.")

inject_accessible_theme()
nav_buttons(back_page="pages/acc_1_setup.py")

# ---- Load model ----
label_map = st.session_state["acc_label_map"]
num_classes = st.session_state["acc_num_classes"]
device = st.session_state.get("acc_device", "cpu")
model = load_model_from_bytes(st.session_state["acc_model_state"], num_classes, device)

# ---- Header ----
st.header("✍️ Practice · Draw a Character")
st.write(
    "Draw a **single letter or digit** on the canvas below. "
    "The AI will recognize it and read it back to you. "
    "You can also upload a photo instead."
)

st.divider()

# ---- Input mode selection ----
mode = st.radio(
    "Input method",
    ["✏️ Draw on canvas", "📷 Upload photo"],
    horizontal=True,
)

x_tensor = None

if mode == "✏️ Draw on canvas":
    st.subheader("Draw here")
    st.caption("Use your finger or mouse. Draw one character, big and clear.")

    # Large canvas for accessibility
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="acc_canvas",
    )

    if canvas_result.image_data is not None:
        # Check if anything was drawn (not just blank canvas)
        alpha = canvas_result.image_data[:, :, 3]
        if alpha.sum() > 100:  # Some ink detected
            x_tensor = canvas_to_tensor(canvas_result.image_data)

else:
    upload = st.file_uploader(
        "Upload a photo of a single character",
        type=["png", "jpg", "jpeg", "webp"],
    )
    if upload is not None:
        pil = Image.open(io.BytesIO(upload.getvalue()))
        st.image(pil, caption="Your character", width=200)
        x_tensor = pil_to_emnist_tensor(pil)

# ---- Recognize ----
st.divider()

if st.button("🔍 Recognize my character", type="primary", disabled=(x_tensor is None)):
    if x_tensor is not None:
        results = predict_character(model, x_tensor, label_map, device=device, top_k=3)
        best_char, best_conf = results[0]

        # Store for XAI page
        st.session_state["acc_last_x"] = x_tensor.detach().cpu().numpy()
        st.session_state["acc_last_pred_class"] = list(label_map.values()).index(best_char)
        st.session_state["acc_last_char"] = best_char
        st.session_state["acc_last_conf"] = best_conf
        st.session_state["acc_last_top3"] = results

        # Display
        show_big_character(best_char)
        show_result(
            f"I see: <strong>{best_char}</strong> (confidence: {best_conf:.0%})",
            "neutral",
        )

        # Top-3 predictions
        st.caption("Top predictions:")
        for char, conf in results:
            st.caption(f"  {char} — {conf:.1%}")

        # Speak it
        speak_character_result(best_char, best_conf)

# ---- Assessment ----
if st.session_state.get("acc_last_char") is not None:
    st.divider()
    st.subheader("Was that correct?")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes, correct!", use_container_width=True):
            st.session_state["acc_attempts"] = st.session_state.get("acc_attempts", 0) + 1
            st.session_state["acc_correct"] = st.session_state.get("acc_correct", 0) + 1
            show_result("Great job! Keep practicing! 🌟", "correct")
            speak("Great job! Keep practicing!", rate=0.75)
    with c2:
        if st.button("❌ No, wrong", use_container_width=True):
            st.session_state["acc_attempts"] = st.session_state.get("acc_attempts", 0) + 1
            show_result("No worries! Let's try again. Practice makes perfect! 💪", "incorrect")
            speak("No worries. Let's try again. Practice makes perfect!", rate=0.75)

    # Progress
    st.divider()
    attempts = st.session_state.get("acc_attempts", 0)
    correct = st.session_state.get("acc_correct", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        accessible_metric("Attempts", str(attempts))
    with c2:
        accessible_metric("Correct", str(correct))
    with c3:
        rate = f"{correct/attempts:.0%}" if attempts > 0 else "—"
        accessible_metric("Accuracy", rate)

    # Navigation
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 See how the AI reads →", type="primary", use_container_width=True):
            st.switch_page("pages/acc_3_xai.py")
    with c2:
        if st.button("📝 Get writing exercises →", use_container_width=True):
            st.switch_page("pages/acc_4_tutor.py")
