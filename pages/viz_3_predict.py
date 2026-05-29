"""
viz_3_predict.py — Upload a handwritten expression photo and get read-out + answer.
"""

import io

import streamlit as st
import torch
from PIL import Image

from core.viz_math_cnn import (
    class_to_solution,
    load_model_from_bytes,
    pil_to_tensor_gray,
    predict_class,
)
from ui.state import init_state, nav_buttons, require

init_state()

require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "visualization":
    st.warning("This page is part of the Visualization flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("viz_model_state", "Train the CNN on the previous step first.")

nav_buttons(back_page="pages/viz_2_train.py")

st.header("🔮 Predict · Read my handwriting")
st.write(
    "Upload a **grayscale or color** photo of a **simple expression** "
    "(digits and + − ×), similar to the training style. "
    "The model only knows the vocabulary it was trained on."
)

st.divider()

upload = st.file_uploader("Your photo", type=["png", "jpg", "jpeg", "webp"])

if upload is not None:
    pil = Image.open(io.BytesIO(upload.getvalue()))
    st.image(pil, caption="Uploaded image", width=320)

if st.button("Run model", type="primary") and upload is not None:
    cm = st.session_state["viz_class_map"]
    img_size = int(st.session_state["viz_img_size"])
    n_cls = len(cm)
    device = st.session_state.get("viz_device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_bytes(st.session_state["viz_model_state"], n_cls, device)
    x = pil_to_tensor_gray(pil, img_size)
    pred = predict_class(model, x, device=device)
    expr, ans = class_to_solution(cm, pred)

    st.session_state["viz_last_x_np"] = x.detach().cpu().numpy()
    st.session_state["viz_last_pred_class"] = pred
    st.session_state["viz_last_expression"] = expr
    st.session_state["viz_last_answer"] = ans
    st.session_state["viz_user_correction"] = None
    st.session_state["viz_llm_note"] = None

    st.success(f"**Read expression:** `{expr}`  →  **Answer:** {ans}")

st.divider()
st.subheader("Your assessment (skill tracking)")

st.write("If the read-out was wrong, say so — this feeds the tutor step and a rough **self-reported skill** tally.")

assessment = st.radio(
    "Was the model’s reading correct for your photo?",
    ("Not yet judged", "Yes, correct", "No, it misread"),
    horizontal=True,
)

correction = st.text_input(
    "If it misread, what is the right expression or answer?",
    value="",
    placeholder="e.g. 7−2 or answer 5",
)

if st.button("Save assessment", type="secondary"):
    if assessment == "Not yet judged":
        st.warning("Pick **Yes** or **No** first.")
    else:
        st.session_state["viz_skill_attempts"] = int(st.session_state.get("viz_skill_attempts") or 0) + 1
        if assessment == "Yes, correct":
            st.session_state["viz_skill_marked_correct"] = (
                int(st.session_state.get("viz_skill_marked_correct") or 0) + 1
            )
            st.session_state["viz_user_correction"] = None
            st.info("Recorded as **correct**.")
        else:
            st.session_state["viz_user_correction"] = correction.strip() or None
            st.info("Recorded as **incorrect** (optional correction stored).")

if st.session_state.get("viz_last_expression"):
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Attempts (self-reported)", st.session_state.get("viz_skill_attempts", 0))
    with c2:
        st.metric("Marked correct", st.session_state.get("viz_skill_marked_correct", 0))

    if st.button("Explain with Grad-CAM & Saliency →", type="primary"):
        st.switch_page("pages/viz_4_explain.py")
