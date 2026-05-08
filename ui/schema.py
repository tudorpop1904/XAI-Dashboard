"""
schema.py — Automatic schema inference for DataFrames.

Classifies columns into: numeric, categorical, boolean-like,
datetime, id-like, and empty.  Also collects basic statistics.
"""

import pandas as pd


def infer_schema(df):
    """
    Walk every column of *df* and classify it.

    Returns a dict with keys:
        all_columns, id_like_columns, numeric_columns, categorical_columns,
        datetime_columns, boolean_like_columns, empty_columns,
        shape, size, memory_usage, missing_values, unique_values, value_counts
    """
    schema = {
        "all_columns": df.columns.tolist(),
        "id_like_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "boolean_like_columns": [],
        "empty_columns": [],
        "shape": df.shape,
        "size": df.size,
        "memory_usage": df.memory_usage().sum(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
        "value_counts": {col: df[col].value_counts().to_dict() for col in df.columns},
    }

    for col in df.columns:
        series = df[col]

        if series.isnull().all():
            schema["empty_columns"].append(col)
            continue

        if (
            col.lower() in ["index", "key"]
            or col.lower().endswith("_id")
            or col.lower().startswith("id")
        ):
            schema["id_like_columns"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            schema["numeric_columns"].append(col)
        else:
            schema["categorical_columns"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            schema["datetime_columns"].append(col)
            continue

        unique_vals = set(series.dropna().astype(str).str.lower().unique())
        if unique_vals.issubset({"0", "1", "true", "false", "yes", "no"}):
            schema["boolean_like_columns"].append(col)
            continue

        if pd.api.types.is_bool_dtype(series):
            schema["boolean_like_columns"].append(col)
            continue

    return schema
