"""
dice_engine.py — DiCE counterfactual explanation engine.

Wraps the dice_ml library to generate diverse counterfactual
explanations for sklearn classifiers.  Supports:
  - Multiple CF generation methods (random, genetic, kdtree)
  - Feature immutability constraints (e.g. "age" can't change)
  - Feature range constraints (permitted directions of change)
  - Multi-target comparison (CFs for several desired outcomes)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import dice_ml
import pandas as pd

# ─────────────────────────────────────────────
# CF Generation Methods
# ─────────────────────────────────────────────

CF_METHODS = {
    "Random": {
        "method": "random",
        "description": "Samples random perturbations from the training data distribution.",
    },
    "Genetic": {
        "method": "genetic",
        "description": "Evolves candidate CFs using a genetic algorithm for diversity.",
    },
    "KD-Tree": {
        "method": "kdtree",
        "description": "Finds nearest neighbours in feature space using a KD-Tree.",
    },
}


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────


@dataclass
class CounterfactualResult:
    """Result of a single CF generation run."""

    method: str
    desired_class: str
    cf_df: pd.DataFrame  # The counterfactual examples
    original_input: pd.DataFrame  # The user's original input
    changes_summary: list[dict]  # Per-CF list of {feature: (old, new)}
    num_cfs_found: int
    elapsed_seconds: float = 0.0


# ─────────────────────────────────────────────
# Core Engine
# ─────────────────────────────────────────────


def generate_counterfactuals(
    model,
    training_df: pd.DataFrame,
    target_column: str,
    input_row: pd.DataFrame,
    desired_class: str | int,
    continuous_features: list[str],
    method: str = "random",
    num_cfs: int = 4,
    features_to_vary: list[str] | None = None,
    permitted_range: dict[str, list] | None = None,
) -> CounterfactualResult:
    """
    Generate diverse counterfactual explanations using DiCE.

    Parameters
    ----------
    model : fitted sklearn classifier
    training_df : pd.DataFrame with target column included
    target_column : str
    input_row : pd.DataFrame (single row, same columns as training_df minus target)
    desired_class : the class the user WANTS to achieve
    continuous_features : list of feature names that are continuous
    method : "random", "genetic", or "kdtree"
    num_cfs : how many CFs to generate
    features_to_vary : if set, only these features are allowed to change
    permitted_range : dict mapping feature -> [min, max] allowed values

    Returns
    -------
    CounterfactualResult
    """
    # Build DiCE data object
    d = dice_ml.Data(
        dataframe=training_df,
        continuous_features=continuous_features,
        outcome_name=target_column,
    )

    # Build DiCE model object
    m = dice_ml.Model(model=model, backend="sklearn")

    # Create the explainer
    exp = dice_ml.Dice(d, m, method=method)

    # Build query instance (must include all feature columns, no target)
    feature_cols = [c for c in training_df.columns if c != target_column]
    query = input_row[feature_cols].copy()

    t0 = time.time()

    cf_result = exp.generate_counterfactuals(
        query_instances=query,
        total_CFs=num_cfs,
        desired_class=desired_class,
        features_to_vary=features_to_vary if features_to_vary else "all",
        permitted_range=permitted_range if permitted_range else {},
    )

    elapsed = time.time() - t0

    # Extract CF dataframe
    cf_df = cf_result.cf_examples_list[0].final_cfs_df
    if cf_df is None or cf_df.empty:
        return CounterfactualResult(
            method=method,
            desired_class=str(desired_class),
            cf_df=pd.DataFrame(),
            original_input=query,
            changes_summary=[],
            num_cfs_found=0,
            elapsed_seconds=round(elapsed, 2),
        )

    # Build changes summary: for each CF, what changed?
    changes_list = []
    for _, cf_row in cf_df.iterrows():
        changes = {}
        for col in feature_cols:
            orig_val = query.iloc[0][col]
            cf_val = cf_row[col]
            if str(orig_val) != str(cf_val):
                changes[col] = (orig_val, cf_val)
        changes_list.append(changes)

    return CounterfactualResult(
        method=method,
        desired_class=str(desired_class),
        cf_df=cf_df,
        original_input=query,
        changes_summary=changes_list,
        num_cfs_found=len(cf_df),
        elapsed_seconds=round(elapsed, 2),
    )
