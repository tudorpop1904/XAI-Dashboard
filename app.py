"""
app.py — AI Image Forensics XAI Playground entrypoint.
"""

import streamlit as st
from dotenv import load_dotenv

from core.llm import check_ollama_available
from ui.state import init_state

# Load .env for local (non-Docker) runs — sets KAGGLE_USERNAME, KAGGLE_KEY, etc.
load_dotenv()

# ---- Page-level config (must be first Streamlit call) ----
st.set_page_config(
    page_title="AI Image Forensics · XAI Playground",
    page_icon="🔍",
    layout="wide",
)

# ---- Initialise session state ----
init_state()

if "llm_ready" not in st.session_state:
    st.session_state["llm_ready"] = False


@st.fragment(run_every="10s")
def poll_ollama():
    if st.session_state.get("llm_ready"):
        return

    try:
        ok, _ = check_ollama_available()
        if ok:
            st.session_state["llm_ready"] = True
            st.toast("Ollama LLM ready for reports.", icon="✅")
            st.rerun()
    except Exception:
        pass


if not st.session_state["llm_ready"]:
    with st.sidebar:
        poll_ollama()
        st.caption("Waiting for Ollama LLM (optional, for Report page)...")


# ---- Define pages ----
pg = st.navigation(
    [
        st.Page("pages/1_home.py", title="Home", icon="🏠"),
        st.Page("pages/2_setup.py", title="Setup", icon="🧪"),
        st.Page("pages/3_upload.py", title="Upload", icon="📤"),
        st.Page("pages/4_xai_compare.py", title="XAI Compare", icon="🌡️"),
        st.Page("pages/5_report.py", title="Report", icon="📊"),
    ]
)

# ---- Run selected page ----
pg.run()
