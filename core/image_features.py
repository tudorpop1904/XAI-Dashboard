"""
Image frequency/texture feature extraction for Faster-Than-Lies style detector.

Provides:
- FFT magnitude channel
- Local Binary Pattern channel
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern


def _tensor_to_gray_np(img: torch.Tensor) -> np.ndarray:
    """
    Convert CHW tensor [3,H,W] to grayscale numpy image [H,W].
    Assumes tensor values are in [0,1].
    """
    img = img.detach().cpu().numpy()

    # CHW -> HWC
    img = np.transpose(img, (1, 2, 0))

    # float -> uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray


def fft_channel(img: torch.Tensor) -> torch.Tensor:
    """
    FFT magnitude feature channel.

    Returns:
        [1,H,W]
    """

    gray = _tensor_to_gray_np(img)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    # log compression
    magnitude = np.log1p(magnitude)

    # normalize
    magnitude -= magnitude.min()
    magnitude /= magnitude.max() + 1e-8

    return torch.tensor(magnitude, dtype=img.dtype, device=img.device).unsqueeze(0)


def magnitude_channel(img: torch.Tensor) -> torch.Tensor:
    """
    Image gradient magnitude feature channel (Sobel).

    Returns:
        [1,H,W]
    """
    gray = _tensor_to_gray_np(img)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    # normalize
    magnitude -= magnitude.min()
    magnitude /= magnitude.max() + 1e-8

    return torch.tensor(magnitude, dtype=img.dtype, device=img.device).unsqueeze(0)


def lbp_channel(img: torch.Tensor, radius: int = 1, points: int = 8) -> torch.Tensor:
    """
    Local Binary Pattern texture channel.

    Returns:
        [1,H,W]
    """

    gray = _tensor_to_gray_np(img)

    lbp = local_binary_pattern(gray, points, radius, method="uniform")

    lbp -= lbp.min()
    lbp /= lbp.max() + 1e-8

    return torch.tensor(lbp, dtype=img.dtype, device=img.device).unsqueeze(0)
