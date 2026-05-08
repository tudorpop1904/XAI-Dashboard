"""
viz_adv_3_evaluate.py — Advanced Math Tutor: Evaluation & Grading.

The text LLM reads the LaTeX transcription, checks every mathematical step
for correctness, and produces:
  - A verdict (correct / incorrect / partially correct / unclear).
  - A step-by-step analysis.
  - Error identification and correction.
  - Practice problems for the student.
"""

import streamlit as st

from ui.state import init_state, require, nav_buttons
from core.vlm_engine import (
    check_eval_available,
    evaluate_solution,
    Verdict,
)

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "viz_advanced":
    st.warning("This page is part of the **Advanced Math Tutor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("adv_transcription", "Run VLM transcription first.")

nav_buttons(back_page="pages/viz_adv_2b_xai.py")

# ---- Header ----
st.header("📝 Evaluate · Solution Grading")
st.write(
    "The AI tutor carefully checks each mathematical step in your solution "
    "for correctness, identifies errors (if any), and provides detailed feedback."
)

st.divider()

# ---- Eval LLM status ----
eval_ok, eval_msg = check_eval_available()
if eval_ok:
    st.success(eval_msg)
else:
    st.error(eval_msg)

# ---- Show the transcription being evaluated ----
with st.expander("📄 Transcription being evaluated", expanded=False):
    st.markdown(st.session_state["adv_transcription"])

# ---- Evaluate button ----
if st.button("Evaluate my solution", type="primary", disabled=not eval_ok):
    latex = st.session_state["adv_transcription"]
    with st.spinner("The tutor is checking your work step by step…"):
        result = evaluate_solution(latex)

    st.session_state["adv_evaluation"] = result
    st.session_state["adv_evaluation_raw"] = result.step_by_step
    st.session_state["adv_llm_feedback"] = None  # Reset downstream

    st.success(f"✅ Evaluation complete in {result.elapsed_seconds:.1f}s.")

# ---- Display evaluation ----
if st.session_state.get("adv_evaluation") is not None:
    result = st.session_state["adv_evaluation"]

    st.divider()

    # ---- Verdict banner ----
    verdict_icons = {
        Verdict.CORRECT: ("✅", "success"),
        Verdict.INCORRECT: ("❌", "error"),
        Verdict.PARTIALLY_CORRECT: ("⚠️", "warning"),
        Verdict.UNCLEAR: ("❓", "info"),
    }
    icon, method = verdict_icons.get(result.verdict, ("❓", "info"))
    getattr(st, method)(f"{icon}  **Verdict: {result.verdict.value.replace('_', ' ').title()}**")

    # ---- Summary ----
    st.subheader("Summary")
    st.markdown(result.summary)

    # ---- Step-by-step ----
    st.subheader("Step-by-Step Analysis")
    st.markdown(result.step_by_step)

    # ---- Errors ----
    if result.errors:
        st.subheader("🔴 Errors Found")
        for i, err in enumerate(result.errors, 1):
            st.markdown(f"**{i}.** {err}")

    # ---- Corrected solution ----
    if result.corrected_latex and result.corrected_latex.lower() not in ("n/a", "none", ""):
        st.subheader("✏️ Corrected Solution")
        st.markdown(result.corrected_latex)
        with st.expander("Raw corrected LaTeX"):
            st.code(result.corrected_latex, language="latex")

    # ---- Practice problems ----
    if result.practice_problems:
        st.subheader("🎯 Practice Problems")
        st.markdown(result.practice_problems)

    # ---- Metrics ----
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Verdict", result.verdict.value.replace("_", " ").title())
    with c2:
        st.metric("Errors", len(result.errors))
    with c3:
        st.metric("Eval time", f"{result.elapsed_seconds:.1f}s")

    # ---- Proceed to counselor ----
    st.divider()
    if st.button("Get detailed counselor feedback →", type="primary"):
        st.switch_page("pages/viz_adv_4_counselor.py")
