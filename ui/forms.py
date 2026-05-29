"""
forms.py — Dynamic input-form generators for prediction.
"""

import pandas as pd
import streamlit as st


def generate_input_form(df, feature_columns):
    """
    Build a Streamlit form with one widget per feature column.

    For categoricals with ≤ 20 unique values  → selectbox
    For integers                               → number_input (step=1)
    For floats                                 → number_input (float)
    Fallback                                   → selectbox of first 20 unique values

    Returns
    -------
    inputs : dict   {column_name: user_value}
    """
    inputs = {}

    for col in feature_columns:
        series = df[col]
        unique_vals = series.dropna().unique().tolist()

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ) and len(unique_vals) <= 20:
            value = st.selectbox(col, unique_vals, key=f"form_{col}")

        elif pd.api.types.is_integer_dtype(series):
            value = st.number_input(
                col,
                value=int(series.median()),
                step=1,
                key=f"form_{col}",
            )

        elif pd.api.types.is_float_dtype(series):
            value = st.number_input(
                col,
                value=float(series.median()),
                key=f"form_{col}",
            )

        else:
            # Fallback: categorical with many values
            value = st.selectbox(col, unique_vals[:20], key=f"form_{col}")

        inputs[col] = value

    return inputs
