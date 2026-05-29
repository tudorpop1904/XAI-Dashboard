"""
6_interpret.py — LLM-powered natural-language explanation.

Sends the SHAP/LIME run data to an Ollama-hosted LLM and
streams a plain-English "guidance counselor" interpretation.
"""

import streamlit as st

from core.llm import (
    build_explanation_prompt,
    check_ollama_available,
    stream_llm_explanation,
)
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("explanation_runs", "Please run XAI explanations first.")
if st.session_state.get("xai_category") != "feature_attribution":
    st.warning(
        "This counselor page is for **SHAP/LIME** (Feature Attribution). "
        "For the handwriting tutor, open **Viz · Tutor** from the sidebar."
    )
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

# ---- Navigation ----
nav_buttons(back_page="pages/5_explain.py")

st.header("🗣️ Guidance Counselor's Note")

st.write(
    "Our AI Guidance Counselor explains the results in plain language. "
    "This helps you understand which of your strengths led to this recommendation."
)

st.divider()

# ---- Ollama status ----

ok, msg = check_ollama_available()

if ok:
    st.success(msg)
else:
    st.warning(msg)
    st.info(
        "💡 Running out of memory? Try switching from `phi3:mini` to `tinyllama` "
        "in your `.env` or `docker-compose.yml`."
    )

# ---- Trigger / Display ----

if st.button("Get Counselor's Explanation", type="primary", disabled=not ok):
    methods_used = st.session_state.get("xai_methods_used", ["SHAP", "LIME"])

    prompt = build_explanation_prompt(
        dataset_name=st.session_state["dataset_name"],
        target_column=st.session_state["target_column"],
        task_type=st.session_state["task_type"],
        prediction=st.session_state["last_prediction"],
        explanation_runs=st.session_state["explanation_runs"],
        model_name=st.session_state.get("model_name", "Unknown"),
        xai_methods=methods_used,
    )

    with st.spinner("Thinking…"):
        response = st.write_stream(stream_llm_explanation(prompt))

    st.session_state["llm_explanation"] = response
    st.success("✅ Interpretation complete!")

elif st.session_state.get("llm_explanation") is not None:
    st.markdown(st.session_state["llm_explanation"])

# ---- Summary footer ----

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.caption(f"**Dataset:** {st.session_state.get('dataset_name', '—')}")
with c2:
    st.caption(f"**Model:** {st.session_state.get('model_name', '—')}")
with c3:
    st.caption(f"**Prediction:** {st.session_state.get('last_prediction', '—')}")
