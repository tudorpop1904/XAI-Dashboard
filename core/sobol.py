"""
Variance Reduction Feature Attribution (Sobol-inspired).

This module implements a sensitivity analysis approach to feature attribution,
rooted in engineering and uncertainty quantification (Sobol indices).

While SHAP computes additive marginal contributions, this method asks:
"How much does fixing this feature to the user's specific value reduce the
overall variance (uncertainty) of the model's output?"

Method:
1. Compute the total variance of the model's predictions over the background data.
2. For each feature j, fix its value to x_j (the instance's value), and sample
   all other features from the background data.
3. Compute the variance of these new predictions.
4. The attribution is the reduction in variance: Var(Total) - Var(Fixed j).

A large reduction means the feature's specific value strongly constrains the model's
possible outputs, making it a highly influential feature for this specific instance.
"""


import numpy as np
import pandas as pd


def variance_attribution(
    model,
    X_background: pd.DataFrame,
    instance: pd.DataFrame,
    task_type: str = "classification"
) -> list[tuple[str, float]]:
    """
    Compute Variance Reduction feature attributions for a single instance.

    Parameters
    ----------
    model :
        A fitted model (classifier or regressor).
    X_background : pd.DataFrame
        The background dataset (e.g., training data).
    instance : pd.DataFrame
        A single-row DataFrame representing the instance to explain.
    task_type : str
        "classification" or "regression".

    Returns
    -------
    List[Tuple[str, float]]
        A list of (feature_name, variance_reduction) tuples, sorted by magnitude.
    """
    if len(instance) != 1:
        raise ValueError("Instance must be a single-row DataFrame.")

    # 1. Compute total variance over the background
    if task_type == "classification":
        if not hasattr(model, "predict_proba"):
            raise ValueError("Classification requires predict_proba().")
        # For classification, we look at the variance of the predicted probability
        # for the class that the model actually predicts for this instance.
        probas = model.predict_proba(instance)[0]
        y_hat = np.argmax(probas)
        bg_preds = model.predict_proba(X_background)[:, y_hat]
    else:
        bg_preds = model.predict(X_background)

    var_total = np.var(bg_preds)

    if var_total == 0:
        # If the model always predicts the exact same thing, no feature has importance
        return [(col, 0.0) for col in instance.columns]

    features = instance.columns
    var_reductions = []

    for col in features:
        # Create a modified background where feature 'col' is fixed to the instance's value
        x_j_val = instance[col].iloc[0]
        X_modified = X_background.copy()
        X_modified[col] = x_j_val

        # Predict on the modified background
        if task_type == "classification":
            mod_preds = model.predict_proba(X_modified)[:, y_hat]
        else:
            mod_preds = model.predict(X_modified)

        var_restricted = np.var(mod_preds)

        # How much did the variance drop?
        # (Normalize by var_total so it acts like a percentage / Sobol index)
        reduction = (var_total - var_restricted) / var_total

        var_reductions.append((col, reduction))

    # Sort by reduction magnitude (largest reduction first)
    var_reductions.sort(key=lambda x: abs(x[1]), reverse=True)

    return var_reductions
