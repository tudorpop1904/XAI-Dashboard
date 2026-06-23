"""PyTorch tests for fake detector and image XAI."""

import numpy as np
import pytest
import torch

from core.fake_data import generate_synthetic_bundle
from core.fake_detector import FakeDetectorCNN, grad_cam, pil_to_tensor, predict, train_detector
from core.image_xai import run_all_methods
from PIL import Image


@pytest.fixture
def tiny_model():
    bundle = generate_synthetic_bundle(samples_per_class=24, img_size=64)
    model, acc, _ = train_detector(bundle, epochs=2, batch_size=16, device="cpu")
    assert acc >= 0.0
    return model, bundle.img_size


def test_predict_shape(tiny_model):
    model, img_size = tiny_model
    arr = np.random.rand(img_size, img_size, 3).astype(np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    x = pil_to_tensor(img, img_size)
    result = predict(model, x, "cpu")
    assert result.label in ("Real", "AI-Generated")
    assert 0.0 <= result.confidence <= 1.0


def test_grad_cam_output(tiny_model):
    model, img_size = tiny_model
    x = torch.rand(1, 3, img_size, img_size)
    cam = grad_cam(model, x, target_class=0, img_hw=(img_size, img_size), device="cpu")
    assert cam.shape == (img_size, img_size)
    assert cam.min() >= 0.0


def test_run_all_xai_methods(tiny_model):
    model, img_size = tiny_model
    x = torch.rand(1, 3, img_size, img_size)
    results = run_all_methods(
        model,
        x,
        target_class=0,
        grid_rows=3,
        grid_cols=3,
        n_sobol_samples=12,
        device="cpu",
    )
    assert len(results) == 5
    for name, res in results.items():
        assert res.heatmap.shape == (img_size, img_size)
        assert res.metrics.forward_passes >= 1
