"""
3_train.py — Model election and training.

Lets the user pick a target column and an ML predictor,
trains the model, and shows the accuracy / R² score.
"""

import streamlit as st
from ui.state import init_state, require, nav_buttons, reset_prediction_state
from core.preprocessing import preprocess_data, detect_task
from core.models import get_model_names, get_available_models, train_model

init_state()

# ---- Guard ----
require("clean_df", "Please load a dataset first.")
if st.session_state.get("xai_category") != "feature_attribution":
    st.warning("This step belongs to **Feature Attribution**.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

# ---- Navigation ----
nav_buttons(back_page="pages/2_dataset.py")

st.header("🧪 Model Training")
st.write("Select a target column and a machine-learning model, then train it.")

st.divider()

clean_df = st.session_state["clean_df"]

# ---- Target Column ----

target_column = st.selectbox(
    "Target column (what to predict)",
    clean_df.columns,
    key="target_selector",
)

# Detect task type from the raw target column
task_type = detect_task(clean_df[target_column])

st.info(f"Detected task type: **{task_type}**")

# ---- Model selector ----

available_models = get_available_models(task_type)

model_name = st.selectbox(
    "Predictor model",
    available_models,
    key="model_selector",
    help="Models are filtered to those supporting the detected task type.",
)

# ---- Train button ----

if st.button("Train Model", type="primary"):

    with st.spinner(f"Training **{model_name}** on `{target_column}`…"):
        X, y, encoders, task = preprocess_data(clean_df, target_column)
        model, score = train_model(X, y, task, model_name)

    # Persist everything
    st.session_state["target_column"] = target_column
    st.session_state["task_type"] = task
    st.session_state["model"] = model
    st.session_state["model_name"] = model_name
    st.session_state["encoders"] = encoders
    st.session_state["X"] = X
    st.session_state["y"] = y

    reset_prediction_state()

    if task == "classification":
        st.success(f"✅ Model trained!  **Accuracy = {score:.4f}**")
    else:
        st.success(f"✅ Model trained!  **R² = {score:.4f}**")

# ---- Show current training info ----

if st.session_state.get("model") is not None:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Active model:** {st.session_state['model_name']}")
    with c2:
        st.write(f"**Target:** `{st.session_state['target_column']}`")

    if st.button("Proceed to Prediction →", type="primary"):
        st.switch_page("pages/4_predict.py")
