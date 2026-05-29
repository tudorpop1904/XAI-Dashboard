"""
viz_adv_1_upload.py — Advanced Math Tutor: Upload handwritten solution pages.

Step 1 of the advanced visualization pipeline. The student uploads one or
more photos of their handwritten math work.  Images are stored in session
state and displayed in order so the student can verify the page sequence.
"""

import io

import streamlit as st
from PIL import Image

from ui.state import init_state, nav_buttons, require

init_state()

# ---- Guard ----
require("xai_category", "Please start from the Home page.")
if st.session_state.get("xai_category") != "viz_advanced":
    st.warning("This page is part of the **Advanced Math Tutor** flow.")
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

nav_buttons(back_page="pages/1_home.py")

# ---- Header ----
st.header("📸 Upload · Handwritten Solution Pages")
st.write(
    "Upload **one or more photos** of your handwritten math solution. "
    "If the solution spans multiple pages, upload them in reading order — "
    "the tutor will stitch them into one continuous argument."
)

st.divider()

# ---- File uploader (multiple) ----
uploads = st.file_uploader(
    "Drop your notebook page(s) here",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
    help="Supported: PNG, JPG, JPEG, WebP, BMP. Max ~20 MB per file.",
)

if uploads:
    images = []
    for f in uploads:
        img = Image.open(io.BytesIO(f.getvalue()))
        images.append(img)

    st.session_state["adv_images"] = images
    st.session_state["adv_image_names"] = [f.name for f in uploads]

    # Clear downstream state when new images are uploaded
    for key in (
        "adv_transcription",
        "adv_transcription_raw",
        "adv_evaluation",
        "adv_evaluation_raw",
        "adv_llm_feedback",
    ):
        st.session_state[key] = None

    st.success(f"✅ {len(images)} page(s) loaded.")

    # ---- Preview grid ----
    st.subheader("Page Preview")
    cols_per_row = min(len(images), 3)
    for row_start in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(images):
                with col:
                    st.image(
                        images[idx],
                        caption=f"Page {idx + 1}: {st.session_state['adv_image_names'][idx]}",
                        use_container_width=True,
                    )

    st.divider()

    if st.button("Transcribe with VLM →", type="primary"):
        st.switch_page("pages/viz_adv_2_transcribe.py")

elif st.session_state.get("adv_images"):
    # Show previously uploaded images
    images = st.session_state["adv_images"]
    names = st.session_state.get("adv_image_names", [])
    st.info(f"{len(images)} page(s) already loaded.")

    cols_per_row = min(len(images), 3)
    for row_start in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(images):
                with col:
                    name = names[idx] if idx < len(names) else f"Page {idx + 1}"
                    st.image(images[idx], caption=name, use_container_width=True)

    st.divider()
    if st.button("Transcribe with VLM →", type="primary"):
        st.switch_page("pages/viz_adv_2_transcribe.py")
