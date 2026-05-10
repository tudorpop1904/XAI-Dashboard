"""
state.py — Centralised Streamlit session-state management.

Provides init_state() for default values, reset helpers, and
page guard functions that redirect when prerequisites are unmet.
"""

import streamlit as st


# ---------------------
# DEFAULT STATE
# ---------------------

_DEFAULTS = {
    # XAI category selection
    "xai_category": None,

    # Dataset
    "raw_df": None,
    "clean_df": None,
    "schema": None,
    "dropped_columns": [],
    "dataset_name": None,
    "dataset_path": None,

    # Training
    "target_column": None,
    "task_type": None,
    "model": None,
    "model_name": None,
    "encoders": {},
    "X": None,
    "y": None,

    # Prediction
    "last_input_df": None,
    "last_prediction": None,
    "last_encoded_input_df": None,

    # Explanation
    "explanation_runs": [],
    "xai_methods_used": [],
    "num_runs": 5,
    "llm_explanation": None,

    # --- Visualization XAI (handwritten math / CNN) ---
    "viz_class_map": None,
    "viz_img_size": None,
    "viz_bundle_meta": None,
    "viz_model_state": None,
    "viz_val_accuracy": None,
    "viz_train_history": None,
    "viz_last_x_np": None,
    "viz_last_pred_class": None,
    "viz_last_expression": None,
    "viz_last_answer": None,
    "viz_user_correction": None,
    "viz_skill_attempts": 0,
    "viz_skill_marked_correct": 0,
    "viz_llm_note": None,
    "viz_device": "cpu",

    # --- Advanced Math Tutor (VLM pipeline) ---
    "adv_images": None,
    "adv_image_names": None,
    "adv_transcription": None,
    "adv_transcription_raw": None,
    "adv_transcription_time": None,
    "adv_occlusion": None,
    "adv_evaluation": None,
    "adv_evaluation_raw": None,
    "adv_llm_feedback": None,

    # --- Counterfactual Explanations (DiCE) ---
    "cf_input_df": None,
    "cf_encoded_input_df": None,
    "cf_current_pred": None,
    "cf_desired_career": None,
    "cf_desired_encoded": None,
    "cf_features_to_vary": None,
    "cf_immutable": None,
    "cf_results": None,
    "cf_llm_feedback": None,

    # --- Accessible Writing Instructor ---
    "acc_model_state": None,
    "acc_label_map": None,
    "acc_num_classes": None,
    "acc_val_accuracy": None,
    "acc_train_history": None,
    "acc_device": "cpu",
    "acc_attempts": 0,
    "acc_correct": 0,
    "acc_last_x": None,
    "acc_last_pred_class": None,
    "acc_last_char": None,
    "acc_last_conf": None,
    "acc_last_top3": None,
    "acc_grad_cam": None,
    "acc_saliency": None,
    "acc_exercises": None,
}


def init_state():
    """Populate session_state with defaults (only keys that are missing)."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_training_state():
    """Clear model-related state when a new dataset is loaded."""
    keys = [
        "target_column", "task_type", "model", "model_name",
        "encoders", "X", "y",
        "last_input_df", "last_prediction", "last_encoded_input_df",
        "explanation_runs", "xai_methods_used", "llm_explanation",
    ]
    for k in keys:
        st.session_state[k] = _DEFAULTS.get(k)


def reset_prediction_state():
    """Clear prediction & explanation state when training is redone."""
    keys = [
        "last_input_df", "last_prediction", "last_encoded_input_df",
        "explanation_runs", "xai_methods_used", "llm_explanation",
    ]
    for k in keys:
        st.session_state[k] = _DEFAULTS.get(k)


def reset_feature_attribution_flow():
    """Clear tabular FA pipeline when switching to another XAI track."""
    reset_training_state()
    for k in (
        "raw_df",
        "clean_df",
        "schema",
        "dropped_columns",
        "dataset_name",
        "dataset_path",
    ):
        st.session_state[k] = _DEFAULTS.get(k)


def reset_visualization_flow():
    """Clear CNN / viz math pipeline."""
    for k, v in _DEFAULTS.items():
        if k.startswith("viz_"):
            st.session_state[k] = v


def reset_advanced_tutor_flow():
    """Clear advanced VLM math tutor pipeline."""
    for k, v in _DEFAULTS.items():
        if k.startswith("adv_"):
            st.session_state[k] = v


def reset_counterfactual_flow():
    """Clear counterfactual (DiCE) pipeline."""
    for k, v in _DEFAULTS.items():
        if k.startswith("cf_"):
            st.session_state[k] = v


def reset_accessible_flow():
    """Clear accessible writing instructor pipeline."""
    for k, v in _DEFAULTS.items():
        if k.startswith("acc_"):
            st.session_state[k] = v


# ---------------------
# PAGE GUARDS
# ---------------------

def require(key, message="Please complete the previous steps first."):
    """
    Guard: if session_state[key] is None / empty, show a warning
    with a link to Home and stop execution of the current page.
    """
    val = st.session_state.get(key)
    if val is None or (isinstance(val, list) and len(val) == 0):
        st.warning(message)
        if st.button("🏠 Go to Home"):
            st.switch_page("pages/1_home.py")
        st.stop()


# ---------------------
# NAVIGATION HELPERS
# ---------------------

def nav_buttons(back_page=None):
    """
    Render Back + Home buttons at the top of a page.
    """
    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("🏠 Home"):
            st.switch_page("pages/1_home.py")
    with cols[1]:
        if back_page and st.button("⬅ Back"):
            st.switch_page(back_page)
