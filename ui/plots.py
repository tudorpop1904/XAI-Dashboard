"""
plots.py — Matplotlib chart builders for SHAP and LIME bar plots.
"""

import matplotlib.pyplot as plt


def plot_shap_bar(shap_df, top_n=10):
    """
    Horizontal bar chart of top-N SHAP feature attributions.
    Returns a matplotlib Figure.
    """
    df = shap_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * top_n)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df["shap_value"]]
    ax.barh(df["feature"], df["shap_value"], color=colors)
    ax.set_title("SHAP Feature Attribution", fontweight="bold")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    return fig


def plot_lime_bar(lime_df, top_n=10):
    """
    Horizontal bar chart of top-N LIME weights.
    Returns a matplotlib Figure.
    """
    df = lime_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * top_n)))
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in df["lime_weight"]]
    ax.barh(df["feature"], df["lime_weight"], color=colors)
    ax.set_title("LIME Local Explanation", fontweight="bold")
    ax.set_xlabel("LIME Weight")
    ax.set_ylabel("Feature / Rule")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    return fig
