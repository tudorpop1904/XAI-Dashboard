"""Tests for core.models module."""

import pytest
from core.models import (
    MODEL_REGISTRY,
    get_model_names,
    get_available_models,
    train_model,
)


class TestModelRegistry:
    def test_registry_has_entries(self):
        assert len(MODEL_REGISTRY) >= 3

    def test_all_have_classification(self):
        for name, entry in MODEL_REGISTRY.items():
            assert "classification" in entry, f"{name} missing classification"

    def test_all_have_shap_explainer(self):
        for name, entry in MODEL_REGISTRY.items():
            assert entry["shap_explainer"] in ("tree", "linear", "kernel")


class TestGetModelNames:
    def test_returns_list(self):
        names = get_model_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_contains_random_forest(self):
        assert "Random Forest" in get_model_names()


class TestGetAvailableModels:
    def test_classification_models(self):
        models = get_available_models("classification")
        assert "Random Forest" in models
        assert "SVM (RBF Kernel)" in models

    def test_regression_excludes_svm(self):
        models = get_available_models("regression")
        assert "SVM (RBF Kernel)" not in models


class TestTrainModel:
    def test_train_random_forest(self, sample_career_df):
        from core.preprocessing import preprocess_data
        X, y, _, task = preprocess_data(sample_career_df, "career")
        model, score = train_model(X, y, task, "Random Forest")
        assert model is not None
        assert 0.0 <= score <= 1.0

    def test_train_decision_tree(self, sample_career_df):
        from core.preprocessing import preprocess_data
        X, y, _, task = preprocess_data(sample_career_df, "career")
        model, score = train_model(X, y, task, "Decision Tree")
        assert model is not None
        assert 0.0 <= score <= 1.0

    def test_unknown_model_raises(self, sample_career_df):
        from core.preprocessing import preprocess_data
        X, y, _, task = preprocess_data(sample_career_df, "career")
        with pytest.raises(ValueError):
            train_model(X, y, task, "NonExistentModel")
