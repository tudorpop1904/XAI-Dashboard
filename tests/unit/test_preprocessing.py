"""Tests for core.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from core.preprocessing import preprocess_data, encode_input, detect_task


class TestDetectTask:
    def test_few_unique_values_is_classification(self):
        y = pd.Series(["A", "B", "C", "A", "B"])
        assert detect_task(y) == "classification"

    def test_many_unique_numeric_is_regression(self):
        y = pd.Series(np.random.rand(100))
        assert detect_task(y) == "regression"

    def test_integer_few_unique_is_classification(self):
        y = pd.Series([1, 2, 3, 1, 2, 3])
        assert detect_task(y) == "classification"


class TestPreprocessData:
    def test_encodes_categorical_features(self, sample_career_df):
        X, y, encoders, task = preprocess_data(sample_career_df, "career")
        assert "coding_skill" in encoders
        assert "career" in encoders
        assert X["coding_skill"].dtype in (np.int32, np.int64)

    def test_numeric_features_unchanged(self, sample_career_df):
        X, y, encoders, task = preprocess_data(sample_career_df, "career")
        assert "math_score" not in encoders
        assert X["math_score"].dtype in (np.int32, np.int64)

    def test_returns_correct_task_type(self, sample_career_df):
        _, _, _, task = preprocess_data(sample_career_df, "career")
        assert task == "classification"

    def test_shapes_match(self, sample_career_df):
        X, y, _, _ = preprocess_data(sample_career_df, "career")
        assert len(X) == len(y) == len(sample_career_df)
        assert len(X.columns) == len(sample_career_df.columns) - 1


class TestEncodeInput:
    def test_encode_single_row(self, sample_career_df, sample_input_row):
        _, _, encoders, _ = preprocess_data(sample_career_df, "career")
        encoded = encode_input(sample_input_row, encoders)
        assert encoded["coding_skill"].dtype in (np.int32, np.int64)
        assert len(encoded) == 1

    def test_unknown_column_passthrough(self, sample_career_df, sample_input_row):
        _, _, encoders, _ = preprocess_data(sample_career_df, "career")
        row = sample_input_row.copy()
        row["extra_col"] = "hello"
        encoded = encode_input(row, encoders)
        assert "extra_col" in encoded.columns
