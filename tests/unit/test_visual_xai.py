"""
test_visual_xai.py — Unit tests for CNN visual XAI (PMI + Sobol).
"""

import numpy as np
import torch

from core.accessible_cnn import AccessibleCNNLight
from core.visual_xai import visual_pmi_attribution, visual_sobol_attribution


def test_visual_pmi_attribution():
    model = AccessibleCNNLight(num_classes=62)
    # EMNIST shape (1, 1, 28, 28)
    x = torch.rand(1, 1, 28, 28)
    heatmap = visual_pmi_attribution(model, x, target_class=5, grid_rows=3, grid_cols=3)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (28, 28)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_visual_sobol_attribution():
    model = AccessibleCNNLight(num_classes=62)
    x = torch.rand(1, 1, 28, 28)
    heatmap = visual_sobol_attribution(model, x, target_class=5, grid_rows=3, grid_cols=3, n_samples=15)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (28, 28)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
