"""Unit tests for visual XAI methods."""

import numpy as np
import pytest


def test_xai_methods_import():
    pytest.importorskip("torch")
    from core.image_xai import BLACK_BOX_METHODS, WHITE_BOX_METHODS

    assert "Occlusion" in BLACK_BOX_METHODS
    assert "Visual PMI" in BLACK_BOX_METHODS
    assert "Visual Sobol" in BLACK_BOX_METHODS
    assert "Grad-CAM" in WHITE_BOX_METHODS
    assert "Saliency" in WHITE_BOX_METHODS


def test_normalize_grid():
    pytest.importorskip("torch")
    from core.image_xai import _normalize_grid

    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm = _normalize_grid(grid)

    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert np.allclose(norm, np.array([[0.0, 1 / 3], [2 / 3, 1.0]]))
