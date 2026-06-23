"""Unit tests for xai_metrics (no PyTorch required)."""

import numpy as np

from core.xai_metrics import complexity_score, fidelity_drop_score, heatmap_stability


def test_heatmap_stability_identical():
    h = np.ones((4, 4), dtype=np.float32) * 0.5
    assert heatmap_stability([h, h, h]) == 1.0


def test_heatmap_stability_different():
    a = np.zeros((3, 3))
    b = np.ones((3, 3))
    score = heatmap_stability([a, b])
    assert score < 0.5


def test_fidelity_drop_score():
    score = fidelity_drop_score(0.9, [0.8, 0.5, 0.85])
    assert 0.0 < score <= 0.9


def test_complexity_score():
    c = complexity_score(16, grid_cells=16)
    assert c["forward_passes"] == 16
    assert "O(16)" in c["asymptotic"]
