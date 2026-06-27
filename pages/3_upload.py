"""3_upload.py — Upload an image for forensic analysis."""

import streamlit as st

from core.fake_detector import load_model_from_bytes, pil_to_tensor, predict
from ui.state import init_state, nav_buttons, require, reset_analysis

init_state()
require("detector_model_state", "Train or load a detector first (Setup page).")
nav_buttons(back_page="pages/2_setup.py")

st.header("📤 Upload Image")
st.write("Upload a JPEG or PNG image to classify as **Real** or **AI-Generated**.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    if st.session_state.get("uploaded_image_name") != uploaded.name:
        reset_analysis()

    from PIL import Image

    img = Image.open(uploaded)
    st.session_state["uploaded_image"] = img
    st.session_state["uploaded_image_name"] = uploaded.name

    img_size = st.session_state["detector_img_size"]
    device = st.session_state["detector_device"]
    features = st.session_state.get("detector_features", {})
    model = load_model_from_bytes(
        st.session_state["detector_model_state"],
        device,
        add_fft=features.get("add_fft", True),
        add_lbp=features.get("add_lbp", True),
        add_magnitude=features.get("add_magnitude", True),
    )
    x = pil_to_tensor(img, img_size)
    st.session_state["input_tensor"] = x

    st.image(img, caption=uploaded.name, use_container_width=True)

    if st.button("Run detection", type="primary"):
        result = predict(model, x, device)
        st.session_state["detection_result"] = result

        label_color = "🟢" if result.label == "Real" else "🔴"
        st.metric(
            f"{label_color} Prediction",
            result.label,
            f"{result.confidence:.1%} confidence",
        )
        st.bar_chart(result.probabilities)

        if st.button("Compare XAI methods →"):
            st.switch_page("pages/4_xai_compare.py")

if st.session_state.get("detection_result") and uploaded:
    st.divider()
    r = st.session_state["detection_result"]
    st.success(f"Last detection: **{r.label}** ({r.confidence:.1%})")
    if st.button("Continue to XAI comparison →", type="primary"):
        st.switch_page("pages/4_xai_compare.py")
