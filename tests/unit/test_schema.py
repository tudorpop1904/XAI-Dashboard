"""Tests for ui.schema module — column classification logic."""

import numpy as np
import pandas as pd

from ui.schema import infer_schema


class TestInferSchema:
    """Verify that infer_schema correctly classifies every column type."""

    def test_numeric_columns(self):
        df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000.0, 60000.0, 70000.0]})
        schema = infer_schema(df)
        assert "age" in schema["numeric_columns"]
        assert "salary" in schema["numeric_columns"]

    def test_categorical_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"], "size": ["S", "M", "L"]})
        schema = infer_schema(df)
        assert "color" in schema["categorical_columns"]
        assert "size" in schema["categorical_columns"]

    def test_id_like_columns(self):
        df = pd.DataFrame({"user_id": [1, 2, 3], "index": [10, 20, 30], "score": [90, 80, 70]})
        schema = infer_schema(df)
        assert "user_id" in schema["id_like_columns"]
        assert "index" in schema["id_like_columns"]
        assert "score" not in schema["id_like_columns"]

    def test_empty_columns(self):
        df = pd.DataFrame({"filled": [1, 2, 3], "empty": [np.nan, np.nan, np.nan]})
        schema = infer_schema(df)
        assert "empty" in schema["empty_columns"]
        assert "filled" not in schema["empty_columns"]

    def test_boolean_like_columns(self):
        df = pd.DataFrame({"flag": [0, 1, 1, 0], "other": [10, 20, 30, 40]})
        schema = infer_schema(df)
        assert "flag" in schema["boolean_like_columns"]

    def test_shape_and_size(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        schema = infer_schema(df)
        assert schema["shape"] == (2, 2)
        assert schema["size"] == 4

    def test_missing_values_counted(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        schema = infer_schema(df)
        assert schema["missing_values"]["a"] == 1
        assert schema["missing_values"]["b"] == 0

    def test_unique_values_counted(self):
        df = pd.DataFrame({"x": [1, 1, 2]})
        schema = infer_schema(df)
        assert schema["unique_values"]["x"] == 2

    def test_mixed_dataframe(self):
        """A realistic DF with numeric, categorical, id-like, and empty columns."""
        df = pd.DataFrame(
            {
                "student_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "math_score": [95, 85, 75],
                "notes": [np.nan, np.nan, np.nan],
            }
        )
        schema = infer_schema(df)
        assert "student_id" in schema["id_like_columns"]
        assert "name" in schema["categorical_columns"]
        assert "math_score" in schema["numeric_columns"]
        assert "notes" in schema["empty_columns"]

    def test_all_columns_key_complete(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [True]})
        schema = infer_schema(df)
        assert set(schema["all_columns"]) == {"a", "b", "c"}
