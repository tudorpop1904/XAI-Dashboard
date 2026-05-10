"""
Shared fixtures for the XAI Dashboard test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_career_df():
    """Small synthetic career dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "math_score": np.random.randint(40, 100, n),
        "coding_skill": np.random.choice(["Beginner", "Intermediate", "Advanced"], n),
        "creativity": np.random.randint(1, 10, n),
        "communication": np.random.randint(1, 10, n),
        "career": np.random.choice(["Engineer", "Designer", "Manager"], n),
    })


@pytest.fixture
def trained_rf(sample_career_df):
    """A fitted RandomForest on the sample dataset (label-encoded)."""
    from core.preprocessing import preprocess_data
    df = sample_career_df
    X, y, encoders, task = preprocess_data(df, "career")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y, encoders, task


@pytest.fixture
def sample_input_row(sample_career_df):
    """A single input row (raw, before encoding)."""
    return pd.DataFrame([{
        "math_score": 75,
        "coding_skill": "Advanced",
        "creativity": 7,
        "communication": 6,
    }])
