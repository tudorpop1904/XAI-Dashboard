"""
Pointwise Mutual Information (PMI) Feature Attribution.

This module implements an information-theoretic approach to feature attribution.
Instead of asking "What is the marginal contribution of this feature?" (SHAP),
PMI asks: "How much did knowing this feature reduce uncertainty about the prediction?"

For a predicted class y_hat and a specific feature value x_j:
PMI(x_j -> y_hat) = log( P(y_hat | x_j) / P(y_hat) )

P(y_hat) is the baseline probability of predicting y_hat (averaged over the background).
P(y_hat | x_j) is the expected probability of predicting y_hat when feature j is fixed
to x_j, marginalizing over all other features using the background distribution.
"""

import numpy as np
import pandas as pd


def pmi_attribution(
    model, X_background: pd.DataFrame, instance: pd.DataFrame, epsilon: float = 1e-8
) -> list[tuple[str, float]]:
    """
    Compute PMI-based feature attributions for a single instance.

    Parameters
    ----------
    model :
        A fitted classifier that implements predict_proba().
    X_background : pd.DataFrame
        The background dataset (e.g., training data) used to marginalize features.
    instance : pd.DataFrame
        A single-row DataFrame representing the instance to explain.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    List[Tuple[str, float]]
        A list of (feature_name, pmi_score) tuples, sorted by absolute importance.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("PMI attribution requires a model with predict_proba() (classification).")

    if len(instance) != 1:
        raise ValueError("Instance must be a single-row DataFrame.")

    # 1. Determine the predicted class for the instance
    probas = model.predict_proba(instance)[0]
    y_hat = np.argmax(probas)

    # 2. Calculate P(y_hat) — the prior probability of this class over the background
    bg_probas = model.predict_proba(X_background)
    p_y_hat_prior = np.mean(bg_probas[:, y_hat])

    # 3. Calculate P(y_hat | x_j) for each feature
    features = instance.columns
    pmi_scores = []

    for col in features:
        # Create a modified background where feature 'col' is fixed to the instance's value
        x_j_val = instance[col].iloc[0]
        X_modified = X_background.copy()
        X_modified[col] = x_j_val

        # Predict on the modified background
        mod_probas = model.predict_proba(X_modified)
        p_y_hat_given_xj = np.mean(mod_probas[:, y_hat])

        # Compute PMI
        # If the feature makes the prediction *more* likely than the prior, PMI > 0.
        # If it makes it *less* likely, PMI < 0.
        ratio = (p_y_hat_given_xj + epsilon) / (p_y_hat_prior + epsilon)
        pmi = np.log2(ratio)

        pmi_scores.append((col, pmi))

    # Sort by absolute magnitude to highlight the most impactful features (positive or negative)
    pmi_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return pmi_scores
