"""
3_train.py — Model election and training.

Lets the user pick a target column and an ML predictor,
trains the model, and shows the accuracy / R² score.
"""

import streamlit as st

from core.models import get_available_models, train_model
from core.preprocessing import detect_task, preprocess_data
from ui.state import init_state, nav_buttons, require, reset_prediction_state

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

selected_model_names = st.multiselect(
    "Predictor model(s)",
    available_models,
    default=[available_models[0]] if available_models else [],
    help="Select one or more models to train and compare.",
)

# ---- Train button ----

if st.button("Train Model(s)", type="primary"):
    if not selected_model_names:
        st.error("Please select at least one model.")
    else:
        trained_models = {}
        with st.spinner("Training selected models on target…"):
            X, y, encoders, task = preprocess_data(clean_df, target_column)
            for name in selected_model_names:
                model, score = train_model(X, y, task, name)
                trained_models[name] = {"model": model, "score": score}

        # Save to session state
        st.session_state["trained_models"] = trained_models
        st.session_state["target_column"] = target_column
        st.session_state["task_type"] = task
        st.session_state["encoders"] = encoders
        st.session_state["X"] = X
        st.session_state["y"] = y

        # Default active model to the first one trained
        first_name = selected_model_names[0]
        st.session_state["model"] = trained_models[first_name]["model"]
        st.session_state["model_name"] = first_name

        reset_prediction_state()
        st.success(f"✅ Trained {len(trained_models)} model(s) successfully!")

# ---- Show current training info ----

if st.session_state.get("trained_models") is not None:
    st.divider()
    st.subheader("📊 Model Comparison")

    metric_name = "Accuracy" if st.session_state["task_type"] == "classification" else "R² Score"
    comparison_data = []
    for name, info in st.session_state["trained_models"].items():
        comparison_data.append({"Model Name": name, metric_name: info["score"]})

    import pandas as pd

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comp_df.style.background_gradient(cmap="Blues", subset=[metric_name]).format({metric_name: "{:.4f}"}),
        use_container_width=True,
    )

    st.subheader("🎯 Active Model Selection")
    active_model_name = st.selectbox(
        "Choose which model to use for Prediction & Explanation:",
        list(st.session_state["trained_models"].keys()),
        index=list(st.session_state["trained_models"].keys()).index(st.session_state["model_name"]),
    )

    # Update active model in session state
    st.session_state["model"] = st.session_state["trained_models"][active_model_name]["model"]
    st.session_state["model_name"] = active_model_name

    st.write(f"**Current Active Model:** `{st.session_state['model_name']}`")
    st.write(f"**Target Column:** `{st.session_state['target_column']}`")

    if st.button("Proceed to Prediction →", type="primary", use_container_width=True):
        st.switch_page("pages/4_predict.py")
