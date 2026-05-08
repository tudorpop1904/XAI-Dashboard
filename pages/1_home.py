"""
1_home.py — Landing page.

Lets the user choose the XAI task category.
"""

import streamlit as st
from ui.state import (
    init_state,
    reset_feature_attribution_flow,
    reset_visualization_flow,
    reset_advanced_tutor_flow,
    reset_counterfactual_flow,
)

init_state()

st.header("Welcome to Career Navigator AI 🎓")
st.write(
    "An interactive playground for **Explainable AI** — "
    "helping students understand how AI makes career suggestions, "
    "and enabling researchers to compare feature-attribution methods."
)

st.divider()

# ---- XAI Category Cards ----

st.subheader("Choose an XAI Task")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.markdown("### 📊 Feature Attribution")
    st.write(
        "Analyse which features (skills, grades, interests) "
        "drive the model's career prediction using **SHAP** and **LIME**."
    )
    if st.button("Start Feature Attribution", type="primary", use_container_width=True):
        reset_visualization_flow()
        reset_advanced_tutor_flow()
        reset_counterfactual_flow()
        st.session_state["xai_category"] = "feature_attribution"
        st.switch_page("pages/2_dataset.py")

with col2:
    st.markdown("### 🖼️ Visualization (Simple CNN)")
    st.caption("CNN · Grad-CAM · Saliency on handwritten expressions")
    st.write(
        "Train a small CNN on **synthetic handwritten** math expressions, upload your "
        "own photo, get a read-out + answer, then inspect **Grad-CAM** and **saliency** "
        "maps with an LLM tutor note."
    )
    if st.button("Start visualization XAI", type="primary", use_container_width=True):
        reset_feature_attribution_flow()
        reset_visualization_flow()
        reset_advanced_tutor_flow()
        reset_counterfactual_flow()
        st.session_state["xai_category"] = "visualization"
        st.switch_page("pages/viz_1_data.py")

with col3:
    st.markdown("### 📐 Advanced Math Tutor")
    st.caption("VLM · Full-page review · LaTeX transcription · Step-by-step grading")
    st.write(
        "Upload **full notebook pages** of handwritten math solutions. "
        "A Vision-Language Model reads, transcribes to LaTeX, evaluates "
        "correctness step-by-step, and provides personalised feedback "
        "with practice problems."
    )
    if st.button("Start Advanced Math Tutor", type="primary", use_container_width=True):
        reset_feature_attribution_flow()
        reset_visualization_flow()
        reset_advanced_tutor_flow()
        reset_counterfactual_flow()
        st.session_state["xai_category"] = "viz_advanced"
        st.switch_page("pages/viz_adv_1_upload.py")

with col4:
    st.markdown("### 🔄 Counterfactual Explanations")
    st.caption("DiCE · What-if · Actionable career changes")
    st.write(
        "Given a career prediction, ask *\"what would I need to change "
        "to get a different career?\"* DiCE generates diverse counterfactual "
        "paths with **immutability constraints** and an LLM counselor."
    )
    if st.button("Start Counterfactual XAI", type="primary", use_container_width=True):
        if st.session_state.get("model") is None:
            st.warning(
                "⚠️ Counterfactual explanations need a trained model. "
                "Please run **Feature Attribution** first to load a dataset "
                "and train a model."
            )
        else:
            reset_visualization_flow()
            reset_advanced_tutor_flow()
            reset_counterfactual_flow()
            st.session_state["xai_category"] = "counterfactual"
            st.switch_page("pages/cf_1_input.py")

st.divider()

# ---- Quick-start info ----

with st.expander("ℹ️  How does this app work?"):
    st.markdown(
        """
        1. **Select an XAI category** above — all four are now live.
        2. **Pick a dataset** from Kaggle's career-guidance collections (FA/CF), or upload photos (Viz / Tutor).
        3. **Choose and train** a machine-learning predictor, or let the VLM handle it.
        4. **Enter your own data** and get a prediction or transcription.
        5. **Explain** the prediction with SHAP/LIME, Grad-CAM/saliency, occlusion sensitivity, or DiCE counterfactuals.
        6. **Interpret** the results with an LLM-powered "Guidance Counselor".

        Each step lives on its own page — use the sidebar or the
        **Back** / **Home** buttons to navigate.
        """
    )

