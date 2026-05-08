"""
preprocessing.py — Data preprocessing and encoding utilities.

Handles label encoding for categorical features and target columns,
task-type detection, and input encoding at prediction time.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def detect_task(y):
    """
    Heuristic to decide classification vs regression.
    If <= 20 unique values or the column is object-typed → classification.
    """
    if (
        len(np.unique(y)) <= 20
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        return "classification"
    return "regression"


def preprocess_data(df, target_column):
    """
    Encode categorical features and the target column.

    Returns
    -------
    X : pd.DataFrame       Encoded feature matrix.
    y : np.ndarray          Encoded target vector.
    encoders : dict         {column_name: LabelEncoder} for every encoded column.
    task : str              "classification" or "regression".
    """
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    encoders = {}

    for col in X.columns:
        # Encode every non-numeric feature (object, string, category, etc.)
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        encoders[target_column] = le

    task = detect_task(df[target_column])

    return X, y, encoders, task


def encode_input(input_df, encoders):
    """
    Apply the same label encoders used during training to a new input row.
    """
    encoded = input_df.copy()

    for col, le in encoders.items():
        if col in encoded.columns:
            encoded[col] = le.transform(encoded[col].astype(str))

    return encoded
