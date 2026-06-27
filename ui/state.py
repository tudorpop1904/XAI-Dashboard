"""
state.py — Session state for the AI-generated image detection XAI playground.
"""

import streamlit as st

_DEFAULTS = {
    "detector_model_state": None,
    "detector_val_accuracy": None,
    "detector_train_history": None,
    "detector_img_size": 128,
    "detector_device": "cpu",
    "uploaded_image": None,
    "uploaded_image_name": None,
    "input_tensor": None,
    "detection_result": None,
    "xai_results": None,
    "xai_stability": None,
    "xai_config": None,
    "llm_report": None,
    "deployment_env": "local",
}


def init_state():
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis():
    for k in (
        "uploaded_image",
        "uploaded_image_name",
        "input_tensor",
        "detection_result",
        "xai_results",
        "xai_config",
        "xai_stability",
        "llm_report",
    ):
        st.session_state[k] = _DEFAULTS.get(k)


def reset_detector():
    reset_analysis()
    for k in ("detector_model_state", "detector_val_accuracy", "detector_train_history"):
        st.session_state[k] = _DEFAULTS.get(k)


def require(key, message="Please complete the previous step first."):
    val = st.session_state.get(key)
    if val is None or (isinstance(val, list) and len(val) == 0):
        st.warning(message)
        if st.button("🏠 Home"):
            st.switch_page("pages/1_home.py")
        st.stop()


def nav_buttons(back_page=None):
    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("🏠 Home"):
            st.switch_page("pages/1_home.py")
    with cols[1]:
        if back_page and st.button("⬅ Back"):
            st.switch_page(back_page)
