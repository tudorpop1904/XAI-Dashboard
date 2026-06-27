"""Unit tests for Kaggle datasets module."""

from core.datasets import list_datasets


def test_list_datasets():
    datasets = list_datasets()
    assert isinstance(datasets, dict)
    assert len(datasets) > 0
    assert "CIFAKE (CIFAR-10 scale)" in datasets
    assert "birdy654/cifake-real-and-ai-generated-synthetic-images" in datasets.values()


# Note: testing download_dataset and load_image_folder would require
# mocking kagglehub and file system. For simplicity in CI, we just test the registry.
