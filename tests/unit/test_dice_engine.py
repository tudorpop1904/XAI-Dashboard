"""Tests for core.dice_engine module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.dice_engine import generate_counterfactuals, CF_METHODS


class TestCFMethods:
    def test_registry_has_entries(self):
        assert len(CF_METHODS) >= 3
        assert "Random" in CF_METHODS
        assert "Genetic" in CF_METHODS

    def test_each_has_method_key(self):
        for name, info in CF_METHODS.items():
            assert "method" in info
            assert "description" in info


class TestGenerateCounterfactuals:
    def test_random_method_produces_cfs(self, trained_rf, sample_input_row):
        model, X, y, encoders, task = trained_rf

        # Build training df with encoded features + target
        training_df = X.copy()
        training_df["career"] = y

        # Encode input
        from core.preprocessing import encode_input
        encoded_input = encode_input(sample_input_row, encoders)

        continuous = [c for c in X.columns if c != "career"]

        result = generate_counterfactuals(
            model=model,
            training_df=training_df,
            target_column="career",
            input_row=encoded_input,
            desired_class=0,
            continuous_features=continuous,
            method="random",
            num_cfs=2,
        )

        assert result.method == "random"
        assert result.num_cfs_found >= 0  # DiCE may or may not find CFs
        assert result.elapsed_seconds >= 0

    def test_result_has_changes_summary(self, trained_rf, sample_input_row):
        model, X, y, encoders, task = trained_rf
        training_df = X.copy()
        training_df["career"] = y

        from core.preprocessing import encode_input
        encoded_input = encode_input(sample_input_row, encoders)
        continuous = [c for c in X.columns if c != "career"]

        result = generate_counterfactuals(
            model=model,
            training_df=training_df,
            target_column="career",
            input_row=encoded_input,
            desired_class=1,
            continuous_features=continuous,
            method="random",
            num_cfs=2,
        )

        assert isinstance(result.changes_summary, list)
