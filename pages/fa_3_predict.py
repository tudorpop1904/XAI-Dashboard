"""
4_predict.py — Custom data insertion and prediction.

Generates a dynamic form based on the dataset's feature columns,
runs the trained model, and displays the prediction.
"""

import pandas as pd
import streamlit as st

from core.preprocessing import encode_input
from ui.forms import generate_input_form
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("model", "Please train a model first.")
if st.session_state.get("xai_category") != "feature_attribution":
    st.warning("This step belongs to **Feature Attribution**.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

# ---- Navigation ----
nav_buttons(back_page="pages/3_train.py")

st.header("🔮 Predict Your Career Path")
st.write("Fill in your details below and let the model suggest a career field.")

st.divider()

clean_df = st.session_state["clean_df"]
target_column = st.session_state["target_column"]
task_type = st.session_state["task_type"]
model = st.session_state["model"]
encoders = st.session_state["encoders"]

feature_columns = [col for col in clean_df.columns if col != target_column]

# ---- Input form ----

st.subheader("Your Profile")

inputs = generate_input_form(clean_df, feature_columns)

if st.button("Predict my Career", type="primary"):
    input_df = pd.DataFrame([inputs])
    encoded_input_df = encode_input(input_df, encoders)

    st.session_state["last_input_df"] = input_df
    st.session_state["last_encoded_input_df"] = encoded_input_df
    st.session_state["explanation_runs"] = []
    st.session_state["llm_explanation"] = None

    prediction = model.predict(encoded_input_df)[0]

    # Decode if the target was label-encoded
    if task_type == "classification" and target_column in encoders:
        prediction_display = encoders[target_column].inverse_transform([int(prediction)])[0]
    else:
        prediction_display = prediction

    st.session_state["last_prediction"] = prediction_display

# ---- Show prediction ----

if st.session_state.get("last_prediction") is not None:
    st.divider()
    st.subheader("Prediction Result")

    pred = st.session_state["last_prediction"]
    st.success(f"🎯  **Recommended Career Field:** {pred}")

    with st.expander("Your input data"):
        st.dataframe(st.session_state["last_input_df"], use_container_width=True)

    if st.button("Explain with XAI →", type="primary"):
        st.switch_page("pages/5_explain.py")
