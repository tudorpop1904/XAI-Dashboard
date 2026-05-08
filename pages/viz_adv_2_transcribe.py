"""
viz_adv_2_transcribe.py — Advanced Math Tutor: VLM transcription step.

The VLM (e.g. Qwen2.5-VL via Ollama) reads the uploaded handwritten pages
and produces a LaTeX transcription.  The student sees the rendered LaTeX
side-by-side with the original photos to confirm the read-out is faithful.
"""

import streamlit as st

from ui.state import init_state, require, nav_buttons
from core.vlm_engine import (
    check_vlm_available,
    transcribe_images,
)

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "viz_advanced":
    st.warning("This page is part of the **Advanced Math Tutor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("adv_images", "Upload your notebook pages first.")

nav_buttons(back_page="pages/viz_adv_1_upload.py")

# ---- Header ----
st.header("🔬 Transcribe · VLM Reading")
st.write(
    "The Vision-Language Model reads your handwriting and transcribes it "
    "into **LaTeX + plain text**.  Review the output below to make sure "
    "the interpretation matches what you wrote."
)

st.divider()

# ---- VLM status ----
vlm_ok, vlm_msg = check_vlm_available()
if vlm_ok:
    st.success(vlm_msg)
else:
    st.error(vlm_msg)
    st.info(
        "💡 Make sure the multimodal model is pulled:\n"
        "```\n"
        "docker exec -it xai-ollama ollama pull qwen2.5vl:7b\n"
        "```"
    )

# ---- Transcription button ----
images = st.session_state["adv_images"]

if st.button("Run transcription", type="primary", disabled=not vlm_ok):
    with st.spinner(
        f"Reading {len(images)} page(s) with VLM — this may take a minute…"
    ):
        result = transcribe_images(images)

    st.session_state["adv_transcription"] = result.latex
    st.session_state["adv_transcription_raw"] = result.raw_text
    st.session_state["adv_transcription_time"] = result.elapsed_seconds

    # Clear downstream state
    st.session_state["adv_occlusion"] = None
    st.session_state["adv_evaluation"] = None
    st.session_state["adv_evaluation_raw"] = None
    st.session_state["adv_llm_feedback"] = None

    st.success(
        f"✅ Transcription complete in {result.elapsed_seconds:.1f}s "
        f"({result.page_count} page(s))."
    )

# ---- Display transcription ----
if st.session_state.get("adv_transcription"):
    latex = st.session_state["adv_transcription"]

    st.divider()
    st.subheader("Transcription Result")

    # Side-by-side: original images vs. LaTeX rendering
    col_img, col_latex = st.columns([1, 1])

    with col_img:
        st.markdown("**Original Pages**")
        for idx, img in enumerate(images):
            st.image(img, caption=f"Page {idx + 1}", use_container_width=True)

    with col_latex:
        st.markdown("**LaTeX Transcription**")
        # Render LaTeX in Streamlit (st.latex for display math)
        st.markdown(latex)

        with st.expander("📋 Raw LaTeX source"):
            st.code(latex, language="latex")

    # ---- Manual correction ----
    st.divider()
    st.subheader("✏️ Manual Corrections (optional)")
    st.write(
        "If the VLM misread something, you can edit the transcription below "
        "before proceeding to evaluation."
    )

    corrected = st.text_area(
        "Editable LaTeX transcription",
        value=latex,
        height=300,
        key="adv_latex_editor",
    )

    if st.button("Save corrections"):
        st.session_state["adv_transcription"] = corrected.strip()
        st.success("Corrections saved.")

    # ---- Proceed ----
    st.divider()

    elapsed = st.session_state.get("adv_transcription_time", 0)
    st.caption(f"Transcription time: {elapsed:.1f}s")

    if st.button("Explain with Occlusion Sensitivity →", type="primary"):
        st.switch_page("pages/viz_adv_2b_xai.py")
