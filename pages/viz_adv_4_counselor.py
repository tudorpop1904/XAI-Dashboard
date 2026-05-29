"""
viz_adv_4_counselor.py — Advanced Math Tutor: LLM Counselor feedback.

The final step — an encouraging, student-friendly interpretation of the
evaluation results.  This mirrors the existing counselor pages but is
tailored to the full-page math review workflow.
"""

import streamlit as st

from core.llm import stream_llm_explanation
from core.vlm_engine import Verdict, check_eval_available
from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "viz_advanced":
    st.warning("This page is part of the **Advanced Math Tutor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("adv_evaluation", "Evaluate your solution first.")

nav_buttons(back_page="pages/viz_adv_3_evaluate.py")

# ---- Header ----
st.header("🗣️ Counselor · Personalized Feedback")
st.write(
    "Your AI math tutor provides a warm, personalized interpretation of "
    "the evaluation results — explaining what you did well, where to "
    "improve, and how to approach similar problems in the future."
)

st.divider()

# ---- Build counselor prompt ----


def _build_adv_counselor_prompt() -> str:
    """Construct the counselor prompt from evaluation state."""
    result = st.session_state["adv_evaluation"]
    latex = st.session_state.get("adv_transcription", "")

    parts = [
        "You are a warm, encouraging, and highly knowledgeable mathematics "
        "tutor and academic counselor. A student submitted a handwritten "
        "mathematical solution which has been transcribed and evaluated by "
        "an AI grader. Your job is to provide a personalised, student-friendly "
        "explanation of the evaluation results.\n\n",
        "Guidelines:\n"
        "- Be encouraging even if the work has errors.\n"
        "- Explain mathematical concepts at an undergraduate level.\n"
        "- Reference specific parts of the student's work when giving feedback.\n"
        "- If the solution is correct, congratulate warmly and suggest ways "
        "  to deepen understanding.\n"
        "- If incorrect, focus on the learning opportunity — explain WHY the "
        "  error is a common mistake and HOW to avoid it.\n"
        "- Keep your response under 400 words.\n"
        "- Use LaTeX (with $ delimiters) for any mathematical expressions.\n\n",
        f"**Verdict:** {result.verdict.value.replace('_', ' ').title()}\n\n",
        f"**Summary from grader:** {result.summary}\n\n",
    ]

    if result.errors:
        parts.append("**Errors found:**\n")
        for i, err in enumerate(result.errors, 1):
            parts.append(f"  {i}. {err}\n")
        parts.append("\n")

    if result.corrected_latex and result.corrected_latex.lower() not in ("n/a", "none"):
        parts.append(f"**Corrected solution:** {result.corrected_latex}\n\n")

    if result.practice_problems:
        parts.append(f"**Practice problems suggested:** {result.practice_problems}\n\n")

    parts.append(f"**Student's original work (LaTeX transcription):**\n{latex}\n\n")

    parts.append(
        "Now write your personalised counselor feedback for the student. "
        "Remember: be encouraging, specific, and educational."
    )

    return "".join(parts)


# ---- LLM status ----
eval_ok, eval_msg = check_eval_available()
if eval_ok:
    st.success(eval_msg)
else:
    st.warning(eval_msg)

# ---- Generate feedback ----
if st.button("Ask the counselor", type="primary", disabled=not eval_ok):
    prompt = _build_adv_counselor_prompt()
    with st.spinner("The counselor is preparing your feedback…"):
        response = st.write_stream(stream_llm_explanation(prompt))

    st.session_state["adv_llm_feedback"] = response
    st.success("✅ Feedback ready!")

elif st.session_state.get("adv_llm_feedback"):
    st.markdown(st.session_state["adv_llm_feedback"])

# ---- Summary footer ----
st.divider()

result = st.session_state.get("adv_evaluation")
if result:
    verdict_icons = {
        Verdict.CORRECT: "✅",
        Verdict.INCORRECT: "❌",
        Verdict.PARTIALLY_CORRECT: "⚠️",
        Verdict.UNCLEAR: "❓",
    }
    icon = verdict_icons.get(result.verdict, "❓")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"**Verdict:** {icon} {result.verdict.value.replace('_', ' ').title()}")
    with c2:
        st.caption(f"**Errors:** {len(result.errors)}")
    with c3:
        page_count = len(st.session_state.get("adv_images", []))
        st.caption(f"**Pages reviewed:** {page_count}")

# ---- Start over ----
st.divider()
if st.button("🏠 Start a new review", use_container_width=True):
    # Clear advanced tutor state
    for key in list(st.session_state.keys()):
        if key.startswith("adv_"):
            del st.session_state[key]
    st.switch_page("pages/1_home.py")
