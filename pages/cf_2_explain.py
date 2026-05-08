"""
cf_2_explain.py — Counterfactual Explanations: DiCE generation & visualization.

Runs DiCE with the selected method, displays the counterfactual
examples as a diff table (what changed), and visualizes the changes.
"""

import streamlit as st
import pandas as pd
import numpy as np

from ui.state import init_state, require, nav_buttons
from core.dice_engine import (
    generate_counterfactuals,
    CF_METHODS,
)

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "counterfactual":
    st.warning("This page is part of the **Counterfactual** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("cf_input_df", "Please fill in your profile first.")

nav_buttons(back_page="pages/cf_1_input.py")

# ---- Header ----
st.header("🔄 Counterfactual · DiCE Explanations")
st.write(
    "DiCE generates **diverse counterfactual examples** — minimal changes "
    "to your profile that would make the model predict your desired career."
)

st.divider()

# ---- Pull state ----
clean_df = st.session_state["clean_df"]
target_column = st.session_state["target_column"]
model = st.session_state["model"]
encoders = st.session_state["encoders"]
input_df = st.session_state["cf_input_df"]
encoded_input_df = st.session_state["cf_encoded_input_df"]
desired_career = st.session_state["cf_desired_career"]
desired_encoded = st.session_state["cf_desired_encoded"]
features_to_vary = st.session_state["cf_features_to_vary"]
current_pred = st.session_state["cf_current_pred"]

feature_columns = [c for c in clean_df.columns if c != target_column]

# Identify continuous features for DiCE
continuous_features = [
    col for col in feature_columns
    if pd.api.types.is_numeric_dtype(clean_df[col])
]

# ---- Configuration ----
st.subheader("⚙️ DiCE Configuration")

c1, c2 = st.columns(2)

with c1:
    method_name = st.selectbox(
        "CF Generation Method",
        list(CF_METHODS.keys()),
        help="Different algorithms explore the feature space differently.",
    )

with c2:
    num_cfs = st.slider(
        "Number of counterfactuals",
        min_value=1,
        max_value=10,
        value=4,
        help="How many alternative profiles to generate.",
    )

method_info = CF_METHODS[method_name]
st.caption(f"**{method_name}**: {method_info['description']}")

if st.session_state.get("cf_immutable"):
    st.info(
        f"🔒 **Immutable features:** {', '.join(st.session_state['cf_immutable'])}. "
        f"DiCE will only modify the remaining {len(features_to_vary)} features."
    )

# ---- Context cards ----
c1, c2 = st.columns(2)
with c1:
    st.metric("Current prediction", current_pred)
with c2:
    st.metric("Desired career", desired_career)

st.divider()

# ---- Generate button ----
if st.button(
    f"Generate {num_cfs} counterfactuals ({method_name})",
    type="primary",
):
    # Prepare encoded training data for DiCE
    from core.preprocessing import encode_input

    # Build a training df with encoded features + target
    training_encoded = clean_df.copy()
    for col, le in encoders.items():
        if col in training_encoded.columns:
            training_encoded[col] = le.transform(
                training_encoded[col].astype(str)
            )

    with st.spinner(f"Generating counterfactuals with {method_name}…"):
        result = generate_counterfactuals(
            model=model,
            training_df=training_encoded,
            target_column=target_column,
            input_row=encoded_input_df,
            desired_class=desired_encoded,
            continuous_features=continuous_features,
            method=method_info["method"],
            num_cfs=num_cfs,
            features_to_vary=features_to_vary,
        )

    st.session_state["cf_results"] = result
    st.session_state["cf_llm_feedback"] = None

    if result.num_cfs_found > 0:
        st.success(
            f"✅ Found {result.num_cfs_found} counterfactual(s) "
            f"in {result.elapsed_seconds:.1f}s."
        )
    else:
        st.warning(
            "⚠️ DiCE could not find counterfactuals for this configuration. "
            "Try a different method, relax constraints, or increase the count."
        )


# ---- Display results ----
if st.session_state.get("cf_results") is not None:
    result = st.session_state["cf_results"]

    if result.num_cfs_found == 0:
        st.warning("No counterfactuals were found.")
        st.stop()

    st.divider()
    st.subheader("📊 Counterfactual Explanations")

    # ---- Decode CFs back to human-readable form ----
    cf_display = result.cf_df.copy()
    orig_display = result.original_input.copy()

    for col, le in encoders.items():
        if col in cf_display.columns and col != target_column:
            try:
                cf_display[col] = le.inverse_transform(
                    cf_display[col].astype(int)
                )
            except (ValueError, IndexError):
                pass
        if col in orig_display.columns and col != target_column:
            try:
                orig_display[col] = le.inverse_transform(
                    orig_display[col].astype(int)
                )
            except (ValueError, IndexError):
                pass
        if col == target_column and col in cf_display.columns:
            try:
                cf_display[col] = le.inverse_transform(
                    cf_display[col].astype(int)
                )
            except (ValueError, IndexError):
                pass

    # ---- Diff table: highlight what changed ----
    st.markdown("**Your profile vs. counterfactual suggestions:**")

    # Build a combined comparison table
    rows = []
    orig_row = orig_display.iloc[0]
    display_cols = [c for c in orig_display.columns]

    for cf_idx in range(len(cf_display)):
        cf_row = cf_display.iloc[cf_idx]
        row_data = {"CF #": cf_idx + 1}
        for col in display_cols:
            orig_v = orig_row[col]
            cf_v = cf_row[col]
            if str(orig_v) != str(cf_v):
                row_data[col] = f"{orig_v} → **{cf_v}**"
            else:
                row_data[col] = str(orig_v)
        if target_column in cf_display.columns:
            row_data["Predicted Career"] = cf_row[target_column]
        rows.append(row_data)

    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ---- Per-CF change cards ----
    st.divider()
    st.subheader("🔍 What Needs to Change")

    for i, changes in enumerate(result.changes_summary):
        with st.expander(f"Counterfactual #{i + 1} — {len(changes)} change(s)", expanded=(i == 0)):
            if not changes:
                st.write("No changes needed (already achievable).")
            else:
                for feat, (old_val, new_val) in changes.items():
                    # Decode back to human-readable
                    if feat in encoders and feat != target_column:
                        try:
                            old_display = encoders[feat].inverse_transform([int(old_val)])[0]
                            new_display = encoders[feat].inverse_transform([int(new_val)])[0]
                        except (ValueError, IndexError):
                            old_display, new_display = old_val, new_val
                    else:
                        old_display, new_display = old_val, new_val

                    st.markdown(f"- **{feat}**: `{old_display}` → `{new_display}`")

    # ---- Feature change frequency chart ----
    st.divider()
    st.subheader("📈 Which Features Change Most Often?")

    change_counts = {}
    for changes in result.changes_summary:
        for feat in changes:
            change_counts[feat] = change_counts.get(feat, 0) + 1

    if change_counts:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(3, len(change_counts) * 0.4)))
        features_sorted = sorted(change_counts.items(), key=lambda x: x[1], reverse=True)
        feat_names = [f[0] for f in features_sorted]
        feat_counts = [f[1] for f in features_sorted]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
        ax.barh(feat_names, feat_counts, color=colors)
        ax.set_xlabel("Times changed across CFs")
        ax.set_title("Feature Change Frequency")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Features that change in **every** CF are likely the most "
            "impactful levers for reaching your desired career."
        )

    # ---- Metrics ----
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CFs Found", result.num_cfs_found)
    with c2:
        st.metric("Method", method_name)
    with c3:
        avg_changes = (
            sum(len(c) for c in result.changes_summary) / len(result.changes_summary)
            if result.changes_summary else 0
        )
        st.metric("Avg Changes", f"{avg_changes:.1f}")
    with c4:
        st.metric("Time", f"{result.elapsed_seconds:.1f}s")

    # ---- Proceed ----
    st.divider()
    if st.button("Get counselor interpretation →", type="primary"):
        st.switch_page("pages/cf_3_counselor.py")
