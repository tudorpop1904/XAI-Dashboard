"""
acc_4_tutor.py — Accessible Writing Instructor: LLM Writing Exercises.

The LLM generates personalized writing exercises based on the child's
progress and recognized characters. Results are read aloud via TTS.
"""

import streamlit as st

from ui.state import init_state, require, nav_buttons
from ui.accessibility import inject_accessible_theme, accessible_metric
from core.llm import stream_llm_explanation, check_ollama_available
from core.tts import speak_exercise

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "accessible_writing":
    st.warning("This page is part of the **Accessible Writing Instructor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

require("acc_model_state", "Train the model first.")

inject_accessible_theme()
nav_buttons(back_page="pages/acc_2_practice.py")

# ---- Header ----
st.header("📝 Writing Tutor · Exercises")
st.write(
    "Your AI writing tutor will suggest personalized exercises "
    "based on your progress. Each exercise will be read aloud."
)

st.divider()


# ---- Build prompt ----
def _build_tutor_prompt() -> str:
    attempts = st.session_state.get("acc_attempts", 0)
    correct = st.session_state.get("acc_correct", 0)
    last_char = st.session_state.get("acc_last_char", None)
    top3 = st.session_state.get("acc_last_top3", [])
    val_acc = st.session_state.get("acc_val_accuracy", None)

    parts = [
        "You are a patient, encouraging writing tutor helping a visually impaired "
        "child learn to write letters and numbers by hand. The child practices by "
        "drawing characters on a digital canvas, which an AI recognizes.\n\n",

        "## Rules for your response\n",
        "- Use SIMPLE, CLEAR language (age 7-12)\n",
        "- Be VERY encouraging and supportive\n",
        "- Give specific, actionable tips about handwriting (stroke order, size, shape)\n",
        "- Generate 3-5 practice exercises (single characters or short words)\n",
        "- Each exercise should be a clear instruction like 'Write the letter B'\n",
        "- Consider the child's visual impairment — suggest writing BIG and BOLD\n\n",

        "## Current session data\n",
    ]

    if last_char:
        parts.append(f"- Last character attempted: **{last_char}**\n")
        if top3:
            conf_str = ", ".join([f"{c}: {p:.0%}" for c, p in top3])
            parts.append(f"- AI confidence: {conf_str}\n")

    parts.append(f"- Total attempts this session: {attempts}\n")
    parts.append(f"- Correctly recognized: {correct}\n")

    if attempts > 0:
        rate = correct / attempts
        parts.append(f"- Success rate: {rate:.0%}\n")

        if rate < 0.3:
            parts.append(
                "\nThe child is struggling. Focus on the BASICS: simple letters "
                "like I, L, O, T. Emphasize drawing SLOWLY and LARGE.\n"
            )
        elif rate < 0.7:
            parts.append(
                "\nThe child is progressing well. Introduce slightly harder characters "
                "and encourage consistency.\n"
            )
        else:
            parts.append(
                "\nThe child is doing great! Challenge them with similar-looking "
                "characters (e.g., b/d, p/q, 6/9) to build distinction skills.\n"
            )

    if val_acc is not None:
        parts.append(f"\nNote: The AI model's overall accuracy is {val_acc:.1%}. ")
        parts.append("Some misrecognitions may be the AI's fault, not the child's.\n")

    parts.append(
        "\nProvide your response in this format:\n"
        "1. A short encouraging comment about their progress\n"
        "2. A specific tip about handwriting\n"
        "3. 3-5 practice exercises (numbered)\n"
        "4. A fun motivational closing\n"
        "\nKeep it under 200 words."
    )

    return "".join(parts)


# ---- LLM status ----
ok, msg = check_ollama_available()
if ok:
    st.success(msg)
else:
    st.warning(msg)

# ---- Progress summary ----
attempts = st.session_state.get("acc_attempts", 0)
correct = st.session_state.get("acc_correct", 0)

c1, c2, c3 = st.columns(3)
with c1:
    accessible_metric("Attempts", str(attempts))
with c2:
    accessible_metric("Correct", str(correct))
with c3:
    rate = f"{correct/attempts:.0%}" if attempts > 0 else "—"
    accessible_metric("Accuracy", rate)

st.divider()

# ---- Generate exercises ----
if st.button("🎯 Get Writing Exercises", type="primary", disabled=not ok):
    prompt = _build_tutor_prompt()

    with st.spinner("Your tutor is preparing exercises…"):
        response = st.write_stream(stream_llm_explanation(prompt))

    st.session_state["acc_exercises"] = response
    st.success("✅ Exercises ready!")

    # Read aloud
    speak_exercise(response[:300] if len(response) > 300 else response)

elif st.session_state.get("acc_exercises"):
    st.markdown(st.session_state["acc_exercises"])

    if st.button("🔊 Read exercises aloud"):
        text = st.session_state["acc_exercises"]
        speak_exercise(text[:300] if len(text) > 300 else text)

# ---- Navigation ----
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("✏️ Back to practice", type="primary", use_container_width=True):
        st.switch_page("pages/acc_2_practice.py")
with c2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("pages/1_home.py")
