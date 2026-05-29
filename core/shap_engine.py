"""
shap_engine.py — SHAP explainer with automatic routing.

Selects TreeExplainer, LinearExplainer, or KernelExplainer
based on the model family, then normalises outputs into a DataFrame.
"""

import numpy as np
import pandas as pd
import shap

# ---------------------
# EXPLAINER BUILDERS
# ---------------------


def build_shap_explainer(model, X_background, explainer_type="tree"):
    """
    Build the appropriate SHAP explainer.

    Parameters
    ----------
    model           Fitted sklearn estimator.
    X_background    Training feature matrix (used as background for
                    KernelExplainer / LinearExplainer).
    explainer_type  One of 'tree', 'linear', 'kernel'.
    """
    if explainer_type == "tree":
        return shap.TreeExplainer(model)

    if explainer_type == "linear":
        return shap.LinearExplainer(model, X_background)

    if explainer_type == "kernel":
        # KernelExplainer is model-agnostic but slow.
        # Use a small sample of the background to keep it tractable.
        bg = shap.sample(X_background, min(100, len(X_background)))

        # Use predict_proba if available, else predict
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        return shap.KernelExplainer(predict_fn, bg)

    raise ValueError(f"Unsupported SHAP explainer type: {explainer_type}")


# ---------------------
# SHAP VALUE COMPUTATION
# ---------------------


def compute_shap_values(explainer, input_df):
    """
    Compute SHAP values for a single-row DataFrame.
    Returns the raw SHAP output (varies by explainer/version).
    """
    return explainer.shap_values(input_df)


# ---------------------
# UTILITIES
# ---------------------


def get_predicted_class_index(model, encoded_input_df):
    """For classifiers, return the index of the predicted class."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(encoded_input_df)[0]
        return int(np.argmax(probs))
    return 0


def shap_values_to_dataframe(shap_values, input_df, predicted_class_idx=1):
    """
    Normalise heterogeneous SHAP outputs into a tidy DataFrame.

    Supports:
    - regression:  ndarray (1, n_features)
    - classification (legacy):  list of arrays (one per class)
    - classification (modern):  ndarray (1, n_features, n_classes)
    """
    feature_names = list(input_df.columns)

    # Legacy SHAP: list of arrays, one per class
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            class_idx = min(predicted_class_idx, len(shap_values) - 1)
            values = shap_values[class_idx][0]
        else:
            values = shap_values[0][0]

    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            values = shap_values[0]
        elif shap_values.ndim == 3:
            class_idx = min(predicted_class_idx, shap_values.shape[2] - 1)
            values = shap_values[0, :, class_idx]
        else:
            raise ValueError(f"Unsupported SHAP ndarray shape: {shap_values.shape}")
    else:
        raise ValueError(f"Unsupported SHAP output type: {type(shap_values)}")

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values,
        }
    )
    shap_df["abs_shap_value"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values(by="abs_shap_value", ascending=False).reset_index(drop=True)
    return shap_df


def get_base_value(explainer, predicted_class_idx=1):
    """
    Extract a usable scalar base (expected) value from SHAP.
    """
    base_value = explainer.expected_value

    if isinstance(base_value, list):
        idx = min(predicted_class_idx, len(base_value) - 1)
        return base_value[idx]

    if isinstance(base_value, np.ndarray):
        if base_value.ndim == 0:
            return float(base_value)
        idx = min(predicted_class_idx, len(base_value) - 1)
        return float(base_value[idx])

    return float(base_value)
