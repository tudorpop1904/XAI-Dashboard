"""
app.py — Application entrypoint / router.

Uses Streamlit's st.navigation API to define the multi-page layout.
All page logic lives in the pages/ directory; this file is only
responsible for routing and shared setup.
"""

from dotenv import load_dotenv

# Load .env for local (non-Docker) runs — sets KAGGLE_USERNAME, KAGGLE_KEY, etc.
load_dotenv()

import streamlit as st
from ui.state import init_state
from core.llm import check_ollama_available
from core.vlm_engine import check_vlm_available, check_surrogate_available

# ---- Page-level config (must be first Streamlit call) ----
st.set_page_config(
    page_title="Career Navigator AI 🎓",
    page_icon="🎓",
    layout="wide",
)

# ---- Initialise session state ----
init_state()

# ---- Background model polling ----
if "models_pulled" not in st.session_state:
    st.session_state["models_pulled"] = False

@st.fragment(run_every="5s")
def poll_ollama_models():
    """Poll Ollama in the background to notify when all models are pulled."""
    if st.session_state.get("models_pulled"):
        return

    try:
        llm_ok, _ = check_ollama_available()
        vlm_ok, _ = check_vlm_available()
        surr_ok, _ = check_surrogate_available()

        if llm_ok and vlm_ok and surr_ok:
            st.session_state["models_pulled"] = True
            st.toast("All models pulled. Ollama is ready!", icon="✅")
            st.rerun()
    except Exception:
        pass

if not st.session_state["models_pulled"]:
    with st.sidebar:
        poll_ollama_models()

# ---- Define pages ----
pg = st.navigation(
    [
        st.Page("pages/1_home.py", title="Home", icon="🏠"),
        st.Page("pages/2_dataset.py", title="FA · Dataset", icon="📊"),
        st.Page("pages/3_train.py", title="FA · Train", icon="🧪"),
        st.Page("pages/4_predict.py", title="FA · Predict", icon="🔮"),
        st.Page("pages/5_explain.py", title="FA · Explain", icon="💡"),
        st.Page("pages/6_interpret.py", title="FA · Counselor", icon="🗣️"),
        st.Page("pages/viz_1_data.py", title="Viz · Data", icon="📷"),
        st.Page("pages/viz_2_train.py", title="Viz · Train", icon="🧠"),
        st.Page("pages/viz_3_predict.py", title="Viz · Predict", icon="✏️"),
        st.Page("pages/viz_4_explain.py", title="Viz · XAI maps", icon="🌡️"),
        st.Page("pages/viz_5_counselor.py", title="Viz · Tutor", icon="📚"),
        st.Page("pages/viz_adv_1_upload.py", title="Tutor · Upload", icon="📸"),
        st.Page("pages/viz_adv_2_transcribe.py", title="Tutor · Transcribe", icon="🔬"),
        st.Page("pages/viz_adv_2b_xai.py", title="Tutor · XAI", icon="🌡️"),
        st.Page("pages/viz_adv_3_evaluate.py", title="Tutor · Evaluate", icon="📝"),
        st.Page("pages/viz_adv_4_counselor.py", title="Tutor · Counselor", icon="🎓"),
        st.Page("pages/cf_1_input.py", title="CF · Input", icon="🔄"),
        st.Page("pages/cf_2_explain.py", title="CF · Explain", icon="🎲"),
        st.Page("pages/cf_3_counselor.py", title="CF · Counselor", icon="🗣️"),
        st.Page("pages/acc_1_setup.py", title="Write · Setup", icon="✍️"),
        st.Page("pages/acc_2_practice.py", title="Write · Practice", icon="✏️"),
        st.Page("pages/acc_3_xai.py", title="Write · XAI", icon="🌡️"),
        st.Page("pages/acc_4_tutor.py", title="Write · Tutor", icon="📝"),
    ]
)

# ---- Run selected page ----
pg.run()