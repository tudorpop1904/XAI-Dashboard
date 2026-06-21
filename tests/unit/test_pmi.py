"""Tests for core.pmi — Pointwise Mutual Information feature attribution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core.pmi import pmi_attribution


@pytest.fixture
def pmi_setup():
    """Train a small RF classifier for PMI tests."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(
        {
            "feature_a": np.random.randint(0, 5, n),
            "feature_b": np.random.randint(0, 5, n),
            "feature_c": np.random.randint(0, 5, n),
        }
    )
    y = (X["feature_a"] > 2).astype(int)  # feature_a is the dominant signal
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    instance = pd.DataFrame([{"feature_a": 4, "feature_b": 1, "feature_c": 2}])
    return model, X, instance


class TestPMIAttribution:
    def test_returns_list_of_tuples(self, pmi_setup):
        model, X_bg, instance = pmi_setup
        result = pmi_attribution(model, X_bg, instance)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_all_features_present(self, pmi_setup):
        model, X_bg, instance = pmi_setup
        result = pmi_attribution(model, X_bg, instance)
        feature_names = [name for name, _ in result]
        assert set(feature_names) == {"feature_a", "feature_b", "feature_c"}

    def test_sorted_by_absolute_magnitude(self, pmi_setup):
        model, X_bg, instance = pmi_setup
        result = pmi_attribution(model, X_bg, instance)
        magnitudes = [abs(score) for _, score in result]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_dominant_feature_ranked_first(self, pmi_setup):
        """feature_a is the signal — it should rank highest."""
        model, X_bg, instance = pmi_setup
        result = pmi_attribution(model, X_bg, instance)
        assert result[0][0] == "feature_a"

    def test_scores_are_finite(self, pmi_setup):
        model, X_bg, instance = pmi_setup
        result = pmi_attribution(model, X_bg, instance)
        for _, score in result:
            assert np.isfinite(score)

    def test_rejects_regressor(self, pmi_setup):
        _, X_bg, instance = pmi_setup
        regressor = RandomForestRegressor(n_estimators=5, random_state=42)
        regressor.fit(X_bg, np.random.rand(len(X_bg)))
        with pytest.raises(ValueError, match="predict_proba"):
            pmi_attribution(regressor, X_bg, instance)

    def test_rejects_multi_row_instance(self, pmi_setup):
        model, X_bg, _ = pmi_setup
        multi = X_bg.head(3)
        with pytest.raises(ValueError, match="single-row"):
            pmi_attribution(model, X_bg, multi)
