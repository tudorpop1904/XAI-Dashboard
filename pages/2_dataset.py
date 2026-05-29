"""
2_dataset.py — Dataset selection and preview.

Downloads the chosen Kaggle dataset, cleans it, and
displays an overview (shape, dropped columns, preview table).
"""

import streamlit as st

from data.cleaner import clean_dataset
from data.datasets import DATASETS
from data.loader import download_dataset, load_csv_from_dataset
from ui.schema import infer_schema
from ui.state import init_state, nav_buttons, require, reset_training_state

init_state()

# ---- Guard ----
require("xai_category", "Please select an XAI category on the Home page first.")
if st.session_state.get("xai_category") != "feature_attribution":
    st.warning(
        "This page is for **Feature Attribution** (tabular career data). "
        "For handwriting + CNN visualization, use **Home → Visualization**."
    )
    if st.button("🏠 Home"):
        st.switch_page("pages/1_home.py")
    st.stop()

# ---- Navigation ----
nav_buttons(back_page="pages/1_home.py")

st.header("📊 Dataset Selection")
st.write("Choose a career-guidance dataset from Kaggle and preview its contents.")

st.divider()

# ---- Dataset picker ----

dataset_name = st.selectbox(
    "Choose dataset",
    list(DATASETS.keys()),
    key="dataset_selector",
)

if st.button("Load Dataset", type="primary"):
    slug = DATASETS[dataset_name]

    with st.spinner("Downloading dataset from Kaggle…"):
        path = download_dataset(slug)
        raw_df = load_csv_from_dataset(path)

    schema = infer_schema(raw_df)
    clean_df, dropped_columns = clean_dataset(raw_df, schema)

    st.session_state["raw_df"] = raw_df
    st.session_state["clean_df"] = clean_df
    st.session_state["schema"] = schema
    st.session_state["dropped_columns"] = dropped_columns
    st.session_state["dataset_name"] = dataset_name
    st.session_state["dataset_path"] = path

    reset_training_state()

    st.success(f"✅ Loaded dataset: **{dataset_name}**")

# ---- Dataset overview ----

if st.session_state.get("clean_df") is not None:
    raw_df = st.session_state["raw_df"]
    clean_df = st.session_state["clean_df"]
    dropped_columns = st.session_state["dropped_columns"]

    st.subheader("Dataset Overview")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Original shape", f"{raw_df.shape[0]} × {raw_df.shape[1]}")
    with c2:
        st.metric("Cleaned shape", f"{clean_df.shape[0]} × {clean_df.shape[1]}")

    if dropped_columns:
        with st.expander(f"Dropped columns ({len(dropped_columns)})"):
            st.write(dropped_columns)

    st.write("### Cleaned Dataset Preview")
    st.dataframe(clean_df.head(10), use_container_width=True)

    with st.expander("Column Types"):
        st.write(clean_df.dtypes.astype(str))

    st.divider()

    if st.button("Proceed to Training →", type="primary"):
        st.switch_page("pages/3_train.py")
