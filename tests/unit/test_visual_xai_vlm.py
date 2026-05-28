"""
test_visual_xai_vlm.py — Unit tests for VLM visual XAI (PMI + Sobol) using mocked VLM calls.
"""

from unittest.mock import patch

import numpy as np
from PIL import Image

from core.visual_xai_vlm import vlm_pmi_sensitivity, vlm_sobol_sensitivity


def mock_transcribe_for_xai(images, model):
    """Hermetic mock transcription for VLM XAI testing."""
    img0 = images[0]
    img_arr = np.asarray(img0)

    # If the image is fully filled with gray (200), return empty
    if np.all(img_arr == 200):
        return ""

    # If some regions are occluded/grayed out, return degraded text
    if np.any(img_arr == 200):
        return "x + = 5"

    # Otherwise return clean transcription
    return "x + y = 5"


@patch("core.visual_xai_vlm._transcribe_for_xai", side_effect=mock_transcribe_for_xai)
def test_vlm_pmi_sensitivity(mock_trans):
    img = Image.new("RGB", (120, 120), color=255)
    res = vlm_pmi_sensitivity([img], "x + y = 5", grid_rows=3, grid_cols=3, use_surrogate=True)

    assert res.heatmap.shape == (120, 120)
    assert len(res.cell_similarities) == 9
    assert res.surrogate_model == "minicpm-v"
    assert res.grid_rows == 3
    assert res.grid_cols == 3


@patch("core.visual_xai_vlm._transcribe_for_xai", side_effect=mock_transcribe_for_xai)
def test_vlm_sobol_sensitivity(mock_trans):
    img = Image.new("RGB", (120, 120), color=255)
    res = vlm_sobol_sensitivity([img], "x + y = 5", grid_rows=3, grid_cols=3, n_samples=6, use_surrogate=True)

    assert res.heatmap.shape == (120, 120)
    assert len(res.cell_similarities) == 6
    assert res.surrogate_model == "minicpm-v"
    assert res.grid_rows == 3
    assert res.grid_cols == 3
