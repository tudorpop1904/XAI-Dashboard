"""
cleaner.py — Dataset cleaning utilities.

Drops empty, ID-like, and unique-integer columns;
fills missing values with medians (numeric) or modes (categorical).
"""

import pandas as pd


def clean_dataset(df, schema):
    """
    Clean the raw DataFrame using the inferred schema.

    Returns
    -------
    df : pd.DataFrame           Cleaned copy.
    dropped_columns : list      Names of columns that were removed.
    """
    df = df.copy()

    # Drop entirely-null columns
    df = df.dropna(axis=1, how="all")

    cols_to_drop = set(schema["empty_columns"] + schema["id_like_columns"])

    # Drop integer columns where every value is unique (row-ID proxies)
    for col in df.columns:
        if df[col].dtype == "int64" and df[col].nunique() == len(df):
            cols_to_drop.add(col)

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Impute missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

    return df, list(cols_to_drop)
