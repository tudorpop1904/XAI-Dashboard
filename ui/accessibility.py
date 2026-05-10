"""
accessibility.py — UI helpers for visually impaired users.

Provides high-contrast CSS injection and large-font formatting
tailored for severe blurriness and tunnel vision.
"""

import streamlit as st


# ─────────────────────────────────────────────
# High-contrast theme (injected via CSS)
# ─────────────────────────────────────────────

_ACCESSIBLE_CSS = """
<style>
/* === Accessible Writing Instructor Theme === */
[data-testid="stAppViewContainer"] .acc-theme {
    /* Already inherits Streamlit dark mode — we amplify contrast */
}

/* Large, clear fonts for all text in accessible pages */
.acc-theme h1, .acc-theme h2, .acc-theme h3 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}
.acc-theme p, .acc-theme span, .acc-theme div, .acc-theme label {
    font-size: 1.35rem !important;
    line-height: 1.6 !important;
}

/* High-contrast colors */
.acc-result-correct {
    background: #1a472a;
    border-left: 6px solid #4ade80;
    padding: 1.2rem;
    border-radius: 8px;
    font-size: 1.8rem !important;
    font-weight: 600;
    margin: 1rem 0;
}
.acc-result-incorrect {
    background: #4a1a1a;
    border-left: 6px solid #f87171;
    padding: 1.2rem;
    border-radius: 8px;
    font-size: 1.8rem !important;
    font-weight: 600;
    margin: 1rem 0;
}
.acc-result-neutral {
    background: #1a3a4a;
    border-left: 6px solid #60a5fa;
    padding: 1.2rem;
    border-radius: 8px;
    font-size: 1.8rem !important;
    font-weight: 600;
    margin: 1rem 0;
}

/* Large character display */
.acc-big-char {
    font-size: 6rem !important;
    font-weight: 800;
    text-align: center;
    padding: 1rem;
    background: #1e293b;
    border-radius: 16px;
    border: 3px solid #94a3b8;
    margin: 1rem 0;
    line-height: 1.2;
}

/* Large buttons */
.acc-theme button[kind="primary"] {
    font-size: 1.4rem !important;
    padding: 0.8rem 2rem !important;
    min-height: 60px !important;
}
</style>
"""


def inject_accessible_theme():
    """Inject the high-contrast, large-font CSS into the current page."""
    st.markdown(_ACCESSIBLE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="acc-theme">', unsafe_allow_html=True)


def show_big_character(char: str):
    """Display a character in very large, high-contrast text."""
    st.markdown(f'<div class="acc-big-char">{char}</div>', unsafe_allow_html=True)


def show_result(text: str, status: str = "neutral"):
    """
    Show a result message with appropriate color coding.
    status: "correct", "incorrect", or "neutral"
    """
    css_class = f"acc-result-{status}"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def accessible_metric(label: str, value: str):
    """Large, high-contrast metric display."""
    st.markdown(
        f'<div style="text-align:center; padding:0.8rem; '
        f'background:#1e293b; border-radius:12px; margin:0.5rem 0;">'
        f'<div style="font-size:1rem; color:#94a3b8;">{label}</div>'
        f'<div style="font-size:2.4rem; font-weight:700; color:#f8fafc;">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
