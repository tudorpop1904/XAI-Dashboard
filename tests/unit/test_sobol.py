"""Tests for core.sobol — Variance Reduction (Sobol-inspired) feature attribution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core.sobol import variance_attribution


@pytest.fixture
def sobol_clf_setup():
    """Train a small RF classifier for Sobol tests."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(
        {
            "signal": np.random.randint(0, 5, n),
            "noise_1": np.random.randint(0, 5, n),
            "noise_2": np.random.randint(0, 5, n),
        }
    )
    y = (X["signal"] > 2).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    instance = pd.DataFrame([{"signal": 4, "noise_1": 1, "noise_2": 2}])
    return model, X, instance


@pytest.fixture
def sobol_reg_setup():
    """Train a small RF regressor for Sobol tests."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(
        {
            "signal": np.random.rand(n) * 10,
            "noise": np.random.rand(n) * 10,
        }
    )
    y = X["signal"] * 2 + np.random.randn(n) * 0.1
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    instance = pd.DataFrame([{"signal": 8.0, "noise": 3.0}])
    return model, X, instance


class TestVarianceAttributionClassification:
    def test_returns_list_of_tuples(self, sobol_clf_setup):
        model, X_bg, instance = sobol_clf_setup
        result = variance_attribution(model, X_bg, instance, task_type="classification")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_all_features_present(self, sobol_clf_setup):
        model, X_bg, instance = sobol_clf_setup
        result = variance_attribution(model, X_bg, instance, task_type="classification")
        names = [name for name, _ in result]
        assert set(names) == {"signal", "noise_1", "noise_2"}

    def test_sorted_by_magnitude(self, sobol_clf_setup):
        model, X_bg, instance = sobol_clf_setup
        result = variance_attribution(model, X_bg, instance, task_type="classification")
        magnitudes = [abs(score) for _, score in result]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_dominant_feature_ranked_first(self, sobol_clf_setup):
        model, X_bg, instance = sobol_clf_setup
        result = variance_attribution(model, X_bg, instance, task_type="classification")
        assert result[0][0] == "signal"

    def test_scores_are_finite(self, sobol_clf_setup):
        model, X_bg, instance = sobol_clf_setup
        result = variance_attribution(model, X_bg, instance, task_type="classification")
        for _, score in result:
            assert np.isfinite(score)

    def test_rejects_multi_row_instance(self, sobol_clf_setup):
        model, X_bg, _ = sobol_clf_setup
        multi = X_bg.head(3)
        with pytest.raises(ValueError, match="single-row"):
            variance_attribution(model, X_bg, multi, task_type="classification")


class TestVarianceAttributionRegression:
    def test_regression_path(self, sobol_reg_setup):
        model, X_bg, instance = sobol_reg_setup
        result = variance_attribution(model, X_bg, instance, task_type="regression")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_signal_dominates_noise(self, sobol_reg_setup):
        model, X_bg, instance = sobol_reg_setup
        result = variance_attribution(model, X_bg, instance, task_type="regression")
        assert result[0][0] == "signal"

    def test_scores_are_finite_regression(self, sobol_reg_setup):
        model, X_bg, instance = sobol_reg_setup
        result = variance_attribution(model, X_bg, instance, task_type="regression")
        for _, score in result:
            assert np.isfinite(score)


class TestVarianceAttributionEdgeCases:
    def test_zero_variance_returns_all_zeros(self):
        """When the model predicts the exact same value for every background row."""
        np.random.seed(42)
        X = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        y = np.array([0, 0, 0])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        instance = pd.DataFrame([{"a": 1, "b": 2}])
        result = variance_attribution(model, X, instance, task_type="classification")
        for _, score in result:
            assert score == 0.0
