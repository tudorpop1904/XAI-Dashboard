"""
lime_engine.py — LIME tabular explainer.

Builds a LimeTabularExplainer and generates local explanations
for a single encoded input row.
"""

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def _get_categorical_feature_indices(X):
    """
    Return indices of low-cardinality integer columns as categorical.
    Since categorical features are label-encoded to int, this heuristic
    lets LIME treat them correctly.
    """
    indices = []
    for i, col in enumerate(X.columns):
        series = X[col]
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= 20:
            indices.append(i)
    return indices


def build_lime_explainer(X, y, task_type="classification"):
    """
    Build a LIME tabular explainer using the encoded training data.
    """
    categorical_features = _get_categorical_feature_indices(X)

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=list(X.columns),
        class_names=([str(c) for c in sorted(pd.Series(y).unique())] if task_type == "classification" else None),
        categorical_features=categorical_features,
        mode=task_type,
        discretize_continuous=True,
        random_state=None,  # keep stochastic across runs
    )
    return explainer


def explain_with_lime(explainer, model, encoded_input_df, task_type="classification", num_features=10):
    """
    Generate one LIME explanation for one encoded input row.
    Returns a DataFrame with feature descriptions and weights.
    """
    instance = encoded_input_df.iloc[0].values

    if task_type == "classification":
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=min(num_features, encoded_input_df.shape[1]),
        )
    else:
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict,
            num_features=min(num_features, encoded_input_df.shape[1]),
        )

    lime_list = exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["feature", "lime_weight"])
    lime_df["abs_lime_weight"] = lime_df["lime_weight"].abs()
    lime_df = lime_df.sort_values(by="abs_lime_weight", ascending=False).reset_index(drop=True)
    return lime_df
