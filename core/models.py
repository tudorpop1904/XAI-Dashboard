"""
models.py — Multi-model registry and training logic.

Provides a registry of sklearn-compatible predictors,
a training function that clones from the registry, and
metadata about which SHAP explainer each model family requires.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ---- Model Imports ----
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------------------
# MODEL REGISTRY
# ---------------------

MODEL_REGISTRY = {
    "Random Forest": {
        "classification": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "regression": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "shap_explainer": "tree",
    },
    "Gradient Boosting": {
        "classification": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "regression": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "shap_explainer": "tree",
    },
    "Decision Tree": {
        "classification": DecisionTreeClassifier(random_state=42),
        "regression": DecisionTreeRegressor(random_state=42),
        "shap_explainer": "tree",
    },
    "Logistic Regression": {
        "classification": LogisticRegression(max_iter=1000, random_state=42),
        "regression": None,
        "shap_explainer": "linear",
    },
    "SVM (RBF Kernel)": {
        "classification": SVC(kernel="rbf", probability=True, random_state=42),
        "regression": None,
        "shap_explainer": "kernel",
    },
}


def get_model_names():
    """Return list of all registered model names."""
    return list(MODEL_REGISTRY.keys())


def get_available_models(task_type):
    """
    Return model names that support the given task type.
    Filters out models whose entry for the task type is None.
    """
    return [
        name
        for name, entry in MODEL_REGISTRY.items()
        if entry.get(task_type) is not None
    ]


def get_shap_explainer_type(model_name):
    """
    Return the SHAP explainer family for a model:
    'tree', 'linear', or 'kernel'.
    """
    return MODEL_REGISTRY[model_name]["shap_explainer"]


def train_model(X, y, task_type, model_name="Random Forest"):
    """
    Train the selected model on the data.

    Parameters
    ----------
    X : pd.DataFrame        Feature matrix (encoded).
    y : array-like          Target vector.
    task_type : str         'classification' or 'regression'.
    model_name : str        Key from MODEL_REGISTRY.

    Returns
    -------
    model : fitted estimator
    score : float           Accuracy (classification) or R² (regression).
    """
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(f"Unknown model: {model_name}")

    template = entry.get(task_type)
    if template is None:
        raise ValueError(
            f"Model '{model_name}' does not support task type '{task_type}'."
        )

    # Clone so the registry prototype stays clean
    model = clone(template)

    # Cap very large datasets to keep training time reasonable
    if len(X) > 50_000:
        sampled_idx = X.sample(n=50_000, random_state=42).index
        X = X.loc[sampled_idx]
        y = (
            y[sampled_idx]
            if isinstance(y, np.ndarray)
            else y.loc[sampled_idx]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task_type == "classification":
        score = accuracy_score(y_test, preds)
    else:
        score = r2_score(y_test, preds)

    return model, score
