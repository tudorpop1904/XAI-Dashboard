"""
loader.py — Kaggle dataset download and CSV loading.
"""

import os
import kagglehub
import pandas as pd


CACHE_DIR = "data_cache"


def download_dataset(dataset_slug):
    """
    Download a dataset from Kaggle via KaggleHub.

    Accepts either a single slug (str) or a list/tuple of fallback slugs.
    Returns the local directory path for the first successful download.
    """
    slugs = dataset_slug if isinstance(dataset_slug, (list, tuple)) else [dataset_slug]
    errors = []

    for slug in slugs:
        try:
            return kagglehub.dataset_download(slug)
        except Exception as exc:
            errors.append(f"{slug}: {exc.__class__.__name__}: {exc}")

    error_details = "\n".join(errors)
    raise RuntimeError(
        "All dataset download attempts failed.\n"
        f"Tried {len(slugs)} slug(s):\n{error_details}"
    )


def load_csv_from_dataset(dataset_path):
    """
    Load the first CSV file found inside the dataset directory.
    """
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            csv_path = os.path.join(dataset_path, file)
            df = pd.read_csv(csv_path)
            return df

    raise FileNotFoundError("No CSV file found in dataset directory.")
