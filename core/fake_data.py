"""
fake_data.py — Synthetic Real vs AI-generated image pairs for detector training.

Generates procedurally distinct classes so the app runs fully offline (no Kaggle).
Production deployments would fine-tune on CiFAKE or similar forensic datasets.
"""

from __future__ import annotations

from core.image_features import fft_channel, lbp_channel

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


CLASS_NAMES = ("Real", "AI-Generated")
LABEL_REAL = 0
LABEL_AI = 1


@dataclass
class FakeImageBundle:
    img_size: int

    train_images: torch.Tensor
    train_labels: torch.Tensor

    val_images: torch.Tensor
    val_labels: torch.Tensor

    train_fft: torch.Tensor | None = None
    train_lbp: torch.Tensor | None = None

    val_fft: torch.Tensor | None = None
    val_lbp: torch.Tensor | None = None


def _noise_layer(rng: random.Random, size: int) -> np.ndarray:
    arr = np.array([rng.random() for _ in range(size * size)], dtype=np.float32).reshape(size, size)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.5)))
    return np.asarray(img, dtype=np.float32) / 255.0


def _render_real_like(rng: random.Random, size: int) -> np.ndarray:
    """Photo-like: mixed channels, sensor noise, irregular edges."""
    base = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        layer = _noise_layer(rng, size)
        layer = layer * rng.uniform(0.3, 1.0) + rng.uniform(0.0, 0.2)
        base[:, :, c] = np.clip(layer, 0, 1)

    img = Image.fromarray((base * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 8)):
        x0, y0 = rng.randint(0, size), rng.randint(0, size)
        x1, y1 = rng.randint(0, size), rng.randint(0, size)
        color = tuple(rng.randint(40, 220) for _ in range(3))
        draw.line([(x0, y0), (x1, y1)], fill=color, width=rng.randint(1, 3))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = rng.gauss(0, 0.03)
    return np.clip(arr + noise, 0.0, 1.0)


def _render_ai_like(rng: random.Random, size: int) -> np.ndarray:
    """GAN-like: smooth gradients, symmetry, high-frequency grid artifacts."""
    cx, cy = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(size, 1)
    hue = rng.uniform(0, 1)
    r = np.clip(0.5 + 0.4 * np.sin(dist * rng.uniform(4, 10) + hue * 6.28), 0, 1)
    g = np.clip(0.5 + 0.4 * np.cos(dist * rng.uniform(3, 8) + hue * 3.14), 0, 1)
    b = np.clip(1.0 - 0.5 * dist + rng.uniform(-0.1, 0.1), 0, 1)
    arr = np.stack([r, g, b], axis=-1).astype(np.float32)

    if rng.random() > 0.3:
        arr = arr + arr[:, ::-1, :]
        arr = arr / 2.0

    grid = rng.randint(4, 12)
    for i in range(0, size, grid):
        arr[i, :, :] *= 0.92
        arr[:, i, :] *= 0.92

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.5)))
    return np.asarray(img, dtype=np.float32) / 255.0


def generate_synthetic_bundle(
    samples_per_class: int = 200,
    img_size: int = 128,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> FakeImageBundle:
    rng = random.Random(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label, renderer in ((LABEL_REAL, _render_real_like), (LABEL_AI, _render_ai_like)):
        for _ in range(samples_per_class):
            images.append(renderer(rng, img_size))
            labels.append(label)

    idx = list(range(len(images)))
    rng.shuffle(idx)
    images = [images[i] for i in idx]
    labels = [labels[i] for i in idx]

    n = len(images)
    n_val = max(2, int(round(n * val_fraction)))
    split = max(2, n - n_val)

    def to_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(imgs, axis=0).transpose(0, 3, 1, 2)
        return torch.from_numpy(stacked.astype(np.float32))

    def compute_features(
        imgs: torch.Tensor,
        use_fft: bool = True,
        use_lbp: bool = True,
    ):
        """
        Precompute forensic feature channels.

        Returns:
            fft tensor [N,1,H,W] or None
            lbp tensor [N,1,H,W] or None
        """

        fft_features = []
        lbp_features = []

        for img in imgs:

            if use_fft:
                fft_features.append(
                    fft_channel(img)
                )

            if use_lbp:
                lbp_features.append(
                    lbp_channel(img)
                )

        fft_tensor = (
            torch.stack(fft_features)
            if use_fft
            else None
        )

        lbp_tensor = (
            torch.stack(lbp_features)
            if use_lbp
            else None
        )

        return fft_tensor, lbp_tensor

    train_x = to_tensor(images[:split])
    val_x = to_tensor(images[split:])
    train_y = torch.tensor(labels[:split], dtype=torch.long)
    val_y = torch.tensor(labels[split:], dtype=torch.long)

    train_fft, train_lbp = compute_features(
        train_x,
        use_fft=True,
        use_lbp=True
    )

    val_fft, val_lbp = compute_features(
        val_x,
        use_fft=True,
        use_lbp=True
    )

    return FakeImageBundle(
        img_size=img_size,

        train_images=train_x,
        train_labels=train_y,

        val_images=val_x,
        val_labels=val_y,

        train_fft=train_fft,
        train_lbp=train_lbp,

        val_fft=val_fft,
        val_lbp=val_lbp,
    )