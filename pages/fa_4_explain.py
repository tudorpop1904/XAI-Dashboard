"""
5_explain.py — XAI method selection, run configuration, and execution.

Lets the user pick SHAP and/or LIME, set the number of runs,
and displays paired side-by-side bar charts per run.
"""

import pandas as pd
import streamlit as st

from core.lime_engine import build_lime_explainer, explain_with_lime
from core.models import get_shap_explainer_type
from core.pmi import pmi_attribution
from core.shap_engine import (
    build_shap_explainer,
    compute_shap_values,
    get_base_value,
    get_predicted_class_index,
    shap_values_to_dataframe,
)
from core.sobol import variance_attribution
from ui.plots import plot_lime_bar, plot_pmi_bar, plot_shap_bar, plot_sobol_bar
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("last_encoded_input_df", "Please make a prediction first.")
if st.session_state.get("xai_category") != "feature_attribution":
    st.warning("This step belongs to **Feature Attribution**.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

# ---- Navigation ----
nav_buttons(back_page="pages/4_predict.py")

st.header("💡 Explain the Prediction")

pred = st.session_state["last_prediction"]
st.info(f"Current prediction: **{pred}**")

# ---- Model Selection ----
if st.session_state.get("trained_models") is not None and len(st.session_state["trained_models"]) > 1:
    st.subheader("🎯 Model Selection")
    active_model_name = st.selectbox(
        "Choose which model to explain:",
        list(st.session_state["trained_models"].keys()),
        index=list(st.session_state["trained_models"].keys()).index(st.session_state["model_name"]),
        key="explain_model_select"
    )
    # Update active model in session state
    st.session_state["model"] = st.session_state["trained_models"][active_model_name]["model"]
    st.session_state["model_name"] = active_model_name
    st.info(f"Explaining prediction with **{active_model_name}**")
    st.divider()

# ---- XAI Configuration ----

st.subheader("XAI Configuration")

c1, c2 = st.columns(2)

with c1:
    use_shap = st.checkbox("SHAP (Game Theory)", value=True, key="use_shap")
    use_lime = st.checkbox("LIME (Surrogate)", value=True, key="use_lime")
    use_pmi = st.checkbox("PMI (Information Theory)", value=True, key="use_pmi")
    use_sobol = st.checkbox("Variance Reduction (Sobol)", value=True, key="use_sobol")

with c2:
    num_runs = st.number_input(
        "Number of runs",
        min_value=1,
        max_value=20,
        value=st.session_state.get("num_runs", 5),
        step=1,
        key="num_runs_input",
        help="SHAP is deterministic for the same model/input. "
             "LIME uses random perturbations so results vary between runs.",
    )

if not use_shap and not use_lime and not use_pmi and not use_sobol:
    st.warning("Please select at least one XAI method.")
    st.stop()

# ---- Run button ----

if st.button("Run Explanations", type="primary"):

    model = st.session_state["model"]
    model_name = st.session_state["model_name"]
    X = st.session_state["X"]
    y = st.session_state["y"]
    task_type = st.session_state["task_type"]
    encoded_input_df = st.session_state["last_encoded_input_df"]

    methods_used = []
    if use_shap:
        methods_used.append("SHAP")
    if use_lime:
        methods_used.append("LIME")
    if use_pmi:
        methods_used.append("PMI")
    if use_sobol:
        methods_used.append("Sobol")

    with st.spinner(f"Running {', '.join(methods_used)} for {num_runs} run(s)…"):

        # Build explainers once
        shap_explainer = None
        lime_explainer = None

        if use_shap:
            explainer_type = get_shap_explainer_type(model_name)
            shap_explainer = build_shap_explainer(model, X, explainer_type)

        if use_lime:
            lime_explainer = build_lime_explainer(X, y, task_type)

        predicted_class_idx = 0
        if task_type == "classification":
            predicted_class_idx = get_predicted_class_index(model, encoded_input_df)

        explanation_runs = []

        for run_idx in range(num_runs):
            run_data = {"run_number": run_idx + 1}

            if use_shap and shap_explainer is not None:
                shap_values = compute_shap_values(shap_explainer, encoded_input_df)
                run_data["shap_df"] = shap_values_to_dataframe(
                    shap_values, encoded_input_df, predicted_class_idx
                )
                run_data["shap_base_value"] = get_base_value(
                    shap_explainer, predicted_class_idx
                )
            else:
                run_data["shap_df"] = None
                run_data["shap_base_value"] = None

            if use_lime and lime_explainer is not None:
                run_data["lime_df"] = explain_with_lime(
                    lime_explainer, model, encoded_input_df, task_type
                )
            else:
                run_data["lime_df"] = None

            if use_pmi:
                pmi_scores = pmi_attribution(model, X, encoded_input_df)
                run_data["pmi_df"] = pd.DataFrame(pmi_scores, columns=["feature", "pmi_score"])
            else:
                run_data["pmi_df"] = None

            if use_sobol:
                sobol_scores = variance_attribution(model, X, encoded_input_df, task_type)
                run_data["sobol_df"] = pd.DataFrame(sobol_scores, columns=["feature", "variance_reduction"])
            else:
                run_data["sobol_df"] = None

            explanation_runs.append(run_data)

    st.session_state["explanation_runs"] = explanation_runs
    st.session_state["xai_methods_used"] = methods_used
    st.session_state["num_runs"] = num_runs
    st.session_state["llm_explanation"] = None  # reset on new runs
    st.success(f"✅ {num_runs} run(s) completed!")

# ---- Render results ----

if st.session_state.get("explanation_runs"):
    st.divider()
    st.subheader("Explanation Results")

    methods_used = st.session_state.get("xai_methods_used", ["SHAP", "LIME"])

    # Calculate how many columns we need
    num_cols = len(methods_used)

    for run in st.session_state["explanation_runs"]:
        st.markdown(f"#### Run {run['run_number']}")

        if num_cols > 0:
            cols = st.columns(num_cols)
        else:
            cols = []

        col_idx = 0
        if "SHAP" in methods_used and run.get("shap_df") is not None:
            with cols[col_idx]:
                st.markdown("**SHAP**")
                fig = plot_shap_bar(run["shap_df"], top_n=10)
                st.pyplot(fig)
                with st.expander("SHAP data table"):
                    st.dataframe(run["shap_df"], use_container_width=True)
                st.caption(f"Base value: {run['shap_base_value']:.6f}")
            col_idx += 1

        # LIME
        if "LIME" in methods_used and run.get("lime_df") is not None:
            with cols[col_idx]:
                st.markdown("**LIME**")
                fig = plot_lime_bar(run["lime_df"], top_n=10)
                st.pyplot(fig)
                with st.expander("LIME data table"):
                    st.dataframe(run["lime_df"], use_container_width=True)
            col_idx += 1

        # PMI
        if "PMI" in methods_used and run.get("pmi_df") is not None:
            with cols[col_idx]:
                st.markdown("**PMI**")
                fig = plot_pmi_bar(run["pmi_df"], top_n=10)
                st.pyplot(fig)
                with st.expander("PMI data table"):
                    st.dataframe(run["pmi_df"], use_container_width=True)
            col_idx += 1

        # Sobol
        if "Sobol" in methods_used and run.get("sobol_df") is not None:
            with cols[col_idx]:
                st.markdown("**Variance Reduction (Sobol)**")
                fig = plot_sobol_bar(run["sobol_df"], top_n=10)
                st.pyplot(fig)
                with st.expander("Sobol data table"):
                    st.dataframe(run["sobol_df"], use_container_width=True)
            col_idx += 1

        st.divider()

    if st.button("Interpret with LLM →", type="primary"):
        st.switch_page("pages/6_interpret.py")
