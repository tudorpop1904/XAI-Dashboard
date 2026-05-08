"""
cf_1_data.py — Counterfactual Explanations: load dataset and build DataFrame.

This module implements the first step of the DiCE counterfactual pipeline:
- Loads the training dataset (iris) and appends metadata.
- Optionally generates a small synthetic dataset for demonstration.
- Creates the combined :py:attr:``
"""

import pandas as pd;
import numpy as np;
import streamlit as st;
from ui.state import init_state, require, nav_buttons;

init_state();

require("cf_category", "Please start from the Home page.");
if st.session_state.get("cf_category") != "counterfactual":
    st.warning(
        "This wizard is for **Counterfactual Explanations**. "
        "Return to Home and choose “Counterfactual Explanations” there."
    );
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py");
    st.stop();


st.header("📖Select a Dataset:");

c1, c2 = st.columns(2);
with c1:
    dataset = st.selectbox("Choose a dataset:", options=["Iris", "Synthetic"]);
with c2:
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1);

