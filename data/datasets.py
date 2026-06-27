"""
datasets.py — Dataset registry.

Maps human-readable dataset names to one or more Kaggle slugs.
If a primary slug fails (403 / removed / policy-restricted), the loader
can automatically try the next slug in the list.
"""

DATASETS = {
    "CIFAKE": [
        "birdy654/cifake-real-and-ai-generated-synthetic-images/data",
    ],
    "Fallback Dataset": [
        "tristanzhang32/ai-generated-images-vs-real-images",
    ],
}
