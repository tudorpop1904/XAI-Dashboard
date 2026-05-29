"""
viz_5_counselor.py — LLM tutor note for the visualization + XAI track.
"""

import streamlit as st

from core.llm import (
    build_viz_counselor_prompt,
    check_ollama_available,
    stream_llm_explanation,
)
from ui.state import init_state, nav_buttons, require

init_state()

require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "visualization":
    st.warning("This page is part of the Visualization flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("viz_last_expression", "Complete prediction (and optionally XAI) first.")

nav_buttons(back_page="pages/viz_4_explain.py")

st.header("🗣️ Tutor’s note")
st.write(
    "A short, student-friendly interpretation of the prediction and how "
    "**Grad-CAM** vs **saliency** relate to trusting the read-out."
)

st.divider()

ok, msg = check_ollama_available()
if ok:
    st.success(msg)
else:
    st.warning(msg)

if st.button("Ask the tutor (LLM)", type="primary", disabled=not ok):
    prompt = build_viz_counselor_prompt(
        expression=st.session_state["viz_last_expression"],
        predicted_answer=str(st.session_state.get("viz_last_answer") or ""),
        pred_class=int(st.session_state.get("viz_last_pred_class") or 0),
        user_correction=st.session_state.get("viz_user_correction"),
        skill_attempts=int(st.session_state.get("viz_skill_attempts") or 0),
        skill_marked_correct=int(st.session_state.get("viz_skill_marked_correct") or 0),
        val_accuracy=st.session_state.get("viz_val_accuracy"),
        xai_methods="Grad-CAM overlay + input saliency map",
    )
    with st.spinner("Thinking…"):
        response = st.write_stream(stream_llm_explanation(prompt))
    st.session_state["viz_llm_note"] = response
    st.success("✅ Done.")

elif st.session_state.get("viz_llm_note"):
    st.markdown(st.session_state["viz_llm_note"])
