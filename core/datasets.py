"""
datasets.py — Kaggle dataset download and loading for deepfake detection.

Integrates with `kagglehub` to download public datasets containing
real and AI-generated images.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import kagglehub
import numpy as np
import torch
from PIL import Image

from core.fake_data import FakeImageBundle, LABEL_REAL, LABEL_AI

DATASETS = {
    "CIFAKE (CIFAR-10 scale)": "birdy654/cifake-real-and-ai-generated-synthetic-images",
    "AI-ArtBench": "ravidussilva/real-ai-art",
    "140k Real vs Fake Faces": "xhlulu/140k-real-and-fake-faces",
}


def list_datasets() -> dict[str, str]:
    """Return a registry of supported Kaggle datasets."""
    return DATASETS


def download_dataset(slug: str, force: bool = False) -> str:
    """
    Download a dataset using kagglehub.
    Returns the absolute local path to the dataset.
    """
    return kagglehub.dataset_download(slug, force_download=force)


def _find_images(base_dir: Path, labels_map: dict[str, int]) -> list[tuple[Path, int]]:
    """Recursively find images and map them to 0/1 labels based on path."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in base_dir.rglob(ext):
            path_str = str(p).lower()
            # Determine label based on keywords in path
            label = -1
            if "real" in path_str or "true" in path_str or "original" in path_str:
                label = LABEL_REAL
            elif "fake" in path_str or "ai" in path_str or "generated" in path_str:
                label = LABEL_AI

            if label != -1:
                images.append((p, label))
    return images


def load_image_folder(
    path: str,
    img_size: int = 128,
    max_samples: int = 200,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> FakeImageBundle:
    """
    Load images from a local path and convert them to a FakeImageBundle.
    """
    rng = random.Random(seed)
    base_dir = Path(path)

    all_images = _find_images(base_dir, {"real": LABEL_REAL, "ai": LABEL_AI, "fake": LABEL_AI})
    
    # Separate into classes
    reals = [p for p in all_images if p[1] == LABEL_REAL]
    fakes = [p for p in all_images if p[1] == LABEL_AI]

    rng.shuffle(reals)
    rng.shuffle(fakes)

    # Limit to max_samples per class
    reals = reals[:max_samples]
    fakes = fakes[:max_samples]

    selected = reals + fakes
    rng.shuffle(selected)

    def _load_tensor(p: Path) -> np.ndarray:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((img_size, img_size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr
        except Exception:
            return np.zeros((img_size, img_size, 3), dtype=np.float32)

    images = []
    labels = []
    for p, label in selected:
        images.append(_load_tensor(p))
        labels.append(label)

    n = len(images)
    n_val = max(2, int(round(n * val_fraction)))
    split = max(2, n - n_val)

    def to_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(imgs, axis=0).transpose(0, 3, 1, 2)
        return torch.from_numpy(stacked.astype(np.float32))

    train_x = to_tensor(images[:split])
    val_x = to_tensor(images[split:])
    train_y = torch.tensor(labels[:split], dtype=torch.long)
    val_y = torch.tensor(labels[split:], dtype=torch.long)

    return FakeImageBundle(
        img_size=img_size,
        train_images=train_x,
        train_labels=train_y,
        val_images=val_x,
        val_labels=val_y,
    )
