"""5_report.py — LLM forensic narrative + deployment metrics summary."""

import os

import streamlit as st

from core.llm import build_forensic_prompt, check_ollama_available, stream_explanation
from ui.state import init_state, nav_buttons, require

init_state()
require("xai_results", "Run XAI comparison first.")
nav_buttons(back_page="pages/4_xai_compare.py")

st.header("📊 Forensic Report")

det = st.session_state["detection_result"]
results = st.session_state["xai_results"]
stability = st.session_state.get("xai_stability", {})

deployment_env = st.radio(
    "Deployment environment (for report context)",
    ["local", "azure-vm"],
    horizontal=True,
    index=0 if st.session_state.get("deployment_env") == "local" else 1,
)
st.session_state["deployment_env"] = deployment_env

if deployment_env == "azure-vm":
    st.caption(
        "Azure VM metrics can be collected via the same dashboard after cloud deploy "
        "(Jenkins `Deploy (Cloud)` stage → `deploy/deploy.sh`)."
    )

st.subheader("Detection summary")
st.write(f"**{det.label}** — {det.confidence:.1%} confidence")
st.json(det.probabilities)

st.subheader("XAI metrics table")
rows = []
for name, res in results.items():
    rows.append(
        {
            "method": name,
            "category": res.category,
            "elapsed_s": res.metrics.elapsed_seconds,
            "peak_memory_mb": res.metrics.peak_memory_mb,
            "forward_passes": res.metrics.forward_passes,
            "stability": stability.get(name, 1.0),
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()
st.subheader("LLM interpretation (optional)")

llm_ok, llm_msg = check_ollama_available()
if llm_ok:
    st.success(llm_msg)
else:
    st.warning(f"{llm_msg} — metrics table above is still valid without LLM.")

if st.button("Generate LLM report", type="primary", disabled=not llm_ok):
    prompt = build_forensic_prompt(
        det.label,
        det.confidence,
        det.probabilities,
        rows,
        deployment_env=deployment_env,
    )
    st.session_state["llm_report"] = st.write_stream(stream_explanation(prompt))

if st.session_state.get("llm_report"):
    st.markdown("### Saved report")
    st.markdown(st.session_state["llm_report"])

st.divider()
st.caption(f"OLLAMA_HOST={os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")
