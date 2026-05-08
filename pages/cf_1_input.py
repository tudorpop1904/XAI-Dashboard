"""
cf_1_input.py — Counterfactual Explanations: Input & Desired Outcome.

The student enters their current profile (skills, grades, interests)
AND selects the career they WANT.  DiCE then shows what changes are
needed to reach that goal.

Prerequisite: the FA pipeline must have been run first (dataset loaded,
model trained) — we reuse the same model and encoders.
"""

import streamlit as st
import pandas as pd

from ui.state import init_state, require, nav_buttons
from ui.forms import generate_input_form
from core.preprocessing import encode_input

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "counterfactual":
    st.warning("This page is part of the **Counterfactual** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("model", "Please train a model first (use Feature Attribution flow).")

nav_buttons(back_page="pages/1_home.py")

# ---- Header ----
st.header("🔄 Counterfactual · Define Your Goal")
st.write(
    "Enter your **current profile** and pick the **career you want**. "
    "DiCE will show you what changes in your profile would lead the AI "
    "to predict that career — in other words, *what should you change "
    "and by how much?*"
)

st.divider()

# ---- Pull state from FA ----
clean_df = st.session_state["clean_df"]
target_column = st.session_state["target_column"]
task_type = st.session_state["task_type"]
model = st.session_state["model"]
encoders = st.session_state["encoders"]

feature_columns = [col for col in clean_df.columns if col != target_column]

# ---- Current profile ----
st.subheader("📋 Your Current Profile")
inputs = generate_input_form(clean_df, feature_columns)

# ---- Desired career ----
st.divider()
st.subheader("🎯 Desired Outcome")

if target_column in encoders:
    career_options = list(encoders[target_column].classes_)
else:
    career_options = sorted(clean_df[target_column].unique().tolist())

desired_career = st.selectbox(
    "Which career do you want the AI to recommend?",
    career_options,
    help="DiCE will find what minimal changes to your profile would "
         "make the model predict this career.",
)

# ---- Feature constraints ----
st.divider()
st.subheader("🔒 Feature Constraints (optional)")
st.write(
    "Mark features as **immutable** if you can't or won't change them "
    "(e.g., age, gender). DiCE will only suggest changes to the remaining features."
)

immutable = st.multiselect(
    "Immutable features (won't be changed)",
    feature_columns,
    default=[],
    help="Select features that should NOT be modified in the counterfactuals.",
)

features_to_vary = [f for f in feature_columns if f not in immutable]

# ---- Current prediction preview ----
st.divider()

input_df = pd.DataFrame([inputs])
encoded_input_df = encode_input(input_df, encoders)

current_pred_raw = model.predict(encoded_input_df)[0]
if task_type == "classification" and target_column in encoders:
    current_pred = encoders[target_column].inverse_transform([int(current_pred_raw)])[0]
else:
    current_pred = current_pred_raw

st.info(f"📌 **Current prediction for your profile:** {current_pred}")

if str(current_pred) == str(desired_career):
    st.success(
        "✅ The model already predicts your desired career! "
        "You can still generate CFs to see alternative paths."
    )

# ---- Save & proceed ----
if st.button("Generate Counterfactuals →", type="primary"):
    # Encode the desired class for DiCE
    if target_column in encoders:
        desired_encoded = int(
            encoders[target_column].transform([desired_career])[0]
        )
    else:
        desired_encoded = desired_career

    st.session_state["cf_input_df"] = input_df
    st.session_state["cf_encoded_input_df"] = encoded_input_df
    st.session_state["cf_current_pred"] = current_pred
    st.session_state["cf_desired_career"] = desired_career
    st.session_state["cf_desired_encoded"] = desired_encoded
    st.session_state["cf_features_to_vary"] = features_to_vary
    st.session_state["cf_immutable"] = immutable
    st.session_state["cf_results"] = None
    st.session_state["cf_llm_feedback"] = None

    st.switch_page("pages/cf_2_explain.py")
