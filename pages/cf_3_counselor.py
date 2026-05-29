"""
cf_3_counselor.py — Counterfactual Explanations: LLM Counselor.

The AI guidance counselor interprets the DiCE counterfactual results
in plain language — turning abstract feature changes into actionable
career advice.
"""

import streamlit as st

from core.llm import check_ollama_available, stream_llm_explanation
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "counterfactual":
    st.warning("This page is part of the **Counterfactual** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("cf_results", "Please generate counterfactuals first.")

nav_buttons(back_page="pages/cf_2_explain.py")

# ---- Header ----
st.header("🗣️ Counselor · Counterfactual Guidance")
st.write(
    "Your AI guidance counselor explains the counterfactual results "
    "in plain language — translating feature changes into **concrete, "
    "actionable advice** for reaching your desired career."
)

st.divider()


# ---- Build the counselor prompt ----
def _build_cf_counselor_prompt() -> str:
    result = st.session_state["cf_results"]
    current_pred = st.session_state["cf_current_pred"]
    desired_career = st.session_state["cf_desired_career"]
    input_df = st.session_state["cf_input_df"]
    encoders = st.session_state.get("encoders", {})
    target_column = st.session_state["target_column"]
    immutable = st.session_state.get("cf_immutable", [])

    parts = [
        "You are a warm, encouraging, and highly knowledgeable academic "
        "guidance counselor. A student used an AI career prediction system "
        "and wants to know what they need to change to reach a different "
        "career path.\n\n",
        "## Context\n",
        f"- **Current prediction:** {current_pred}\n",
        f"- **Desired career:** {desired_career}\n",
        f"- **DiCE method used:** {result.method}\n",
        f"- **Counterfactuals found:** {result.num_cfs_found}\n",
    ]

    if immutable:
        parts.append(f"- **Locked features (can't change):** {', '.join(immutable)}\n")

    parts.append("\n**Student's current profile:**\n")
    for col in input_df.columns:
        parts.append(f"  - {col}: {input_df.iloc[0][col]}\n")

    parts.append("\n**Counterfactual changes suggested by DiCE:**\n")
    for i, changes in enumerate(result.changes_summary):
        parts.append(f"\nPath #{i + 1}:\n")
        if not changes:
            parts.append("  (No changes needed)\n")
        else:
            for feat, (old_val, new_val) in changes.items():
                # Decode to readable
                if feat in encoders and feat != target_column:
                    try:
                        old_d = encoders[feat].inverse_transform([int(old_val)])[0]
                        new_d = encoders[feat].inverse_transform([int(new_val)])[0]
                    except (ValueError, IndexError):
                        old_d, new_d = old_val, new_val
                else:
                    old_d, new_d = old_val, new_val
                parts.append(f"  - {feat}: {old_d} → {new_d}\n")

    parts.append(
        "\n## Your Task\n"
        "1. Explain EACH suggested path in concrete, actionable terms.\n"
        "   - What does each feature change MEAN practically?\n"
        "   - How feasible/difficult is each change?\n"
        "   - Which path seems most realistic?\n"
        "2. If multiple paths exist, compare them: which is easiest, "
        "   which is most impactful, which gives the broadest advantage?\n"
        "3. Point out any features that change consistently across ALL "
        "   paths — these are the critical levers.\n"
        "4. Give 2-3 specific, concrete steps the student can take "
        "   TODAY to start moving toward their desired career.\n"
        "5. Be encouraging but realistic. Under 350 words.\n"
    )

    return "".join(parts)


# ---- LLM status ----
ok, msg = check_ollama_available()
if ok:
    st.success(msg)
else:
    st.warning(msg)

# ---- Quick summary ----
result = st.session_state["cf_results"]
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Current", st.session_state["cf_current_pred"])
with c2:
    st.metric("Desired", st.session_state["cf_desired_career"])
with c3:
    st.metric("Paths found", result.num_cfs_found)

st.divider()

# ---- Generate feedback ----
if st.button("Get Counselor's Advice", type="primary", disabled=not ok):
    prompt = _build_cf_counselor_prompt()

    with st.spinner("The counselor is preparing your guidance…"):
        response = st.write_stream(stream_llm_explanation(prompt))

    st.session_state["cf_llm_feedback"] = response
    st.success("✅ Guidance ready!")

elif st.session_state.get("cf_llm_feedback"):
    st.markdown(st.session_state["cf_llm_feedback"])

# ---- Footer ----
st.divider()

c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Try different career", use_container_width=True):
        st.session_state["cf_results"] = None
        st.session_state["cf_llm_feedback"] = None
        st.switch_page("pages/cf_1_input.py")
with c2:
    if st.button("🏠 Start over", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("cf_"):
                del st.session_state[key]
        st.switch_page("pages/1_home.py")
