"""1_home.py — Landing page for AI-generated image detection XAI playground."""

import streamlit as st

from ui.state import init_state

init_state()

st.header("🔍 AI Image Forensics · XAI Playground")

st.write(
    "A research-oriented dashboard for **detecting AI-generated images** and "
    "**comparing visual explainability methods** on a single binary classification task."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Detection task")
    st.markdown(
        """
        - **Input:** uploaded RGB image
        - **Output:** Real vs AI-Generated (+ confidence)
        - **Model:** Local PyTorch CNN incorporating the **Faster-Than-Lies (FTL)** methodology (FFT + LBP + Magnitude channels)
        - **Data:** Kaggle integration (e.g. CIFAKE) or offline synthetic pairs
        """
    )

with col2:
    st.subheader("XAI methods compared")
    st.markdown(
        """
        **Black-box (perturbation):**
        Occlusion Sensitivity · Visual PMI · Visual Sobol
        **White-box (gradient, same CNN):**
        Grad-CAM · Saliency
        """
    )

st.divider()

st.subheader("Research axes")
st.markdown(
    """
    1. **Theoretical** — fidelity, stability, asymptotic complexity across methods
    2. **Methodological** — validating the **Faster-Than-Lies** feature engineering approach
    3. **Engineering** — CPU time, peak RAM, forward-pass count (local vs Azure VM)
    4. **MLOps** — Jenkins CI/CD pipeline (lint, test, Docker build, deploy)
    """
)

if st.button("Get started → Train detector", type="primary", use_container_width=True):
    st.switch_page("pages/2_setup.py")

if st.session_state.get("detector_model_state") is not None:
    st.success("Detector model is loaded in session. You can skip to upload.")
    if st.button("Go to Upload →", use_container_width=True):
        st.switch_page("pages/3_upload.py")
