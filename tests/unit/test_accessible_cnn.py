"""Tests for core.accessible_cnn module."""

import numpy as np
import torch
from PIL import Image

from core.accessible_cnn import (
    AccessibleCNN,
    build_emnist_label_map,
    canvas_to_tensor,
    grad_cam_accessible,
    load_model_from_bytes,
    model_state_bytes,
    pil_to_emnist_tensor,
    predict_character,
    saliency_accessible,
)


class TestLabelMap:
    def test_has_62_classes(self):
        m = build_emnist_label_map()
        assert len(m) == 62

    def test_contains_digits(self):
        m = build_emnist_label_map()
        for i in range(10):
            assert m[i] == str(i)

    def test_contains_uppercase(self):
        m = build_emnist_label_map()
        assert m[10] == "A"
        assert m[35] == "Z"

    def test_contains_lowercase(self):
        m = build_emnist_label_map()
        assert m[36] == "a"
        assert m[61] == "z"


class TestAccessibleCNN:
    def test_forward_shape(self):
        model = AccessibleCNN(num_classes=62)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 62)

    def test_forward_with_conv(self):
        model = AccessibleCNN(num_classes=62)
        x = torch.randn(1, 1, 28, 28)
        logits, conv = model.forward_with_conv(x)
        assert logits.shape == (1, 62)
        assert conv.ndim == 4


class TestCanvasToTensor:
    def test_rgba_input(self):
        # Simulate canvas RGBA output
        canvas = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas[30:70, 30:70, 3] = 255  # Draw a square in alpha
        t = canvas_to_tensor(canvas)
        assert t.shape == (1, 1, 28, 28)
        assert t.dtype == torch.float32

    def test_none_input(self):
        t = canvas_to_tensor(None)
        assert t.shape == (1, 1, 28, 28)
        assert t.sum().item() == 0.0


class TestPilToEmnist:
    def test_converts_pil(self):
        img = Image.new("L", (50, 50), color=128)
        t = pil_to_emnist_tensor(img)
        assert t.shape == (1, 1, 28, 28)


class TestPredictCharacter:
    def test_returns_top_k(self):
        model = AccessibleCNN(num_classes=62)
        x = torch.randn(1, 1, 28, 28)
        label_map = build_emnist_label_map()
        results = predict_character(model, x, label_map, top_k=3)
        assert len(results) == 3
        for char, conf in results:
            assert isinstance(char, str)
            assert 0.0 <= conf <= 1.0


class TestXAI:
    def test_grad_cam_shape(self):
        model = AccessibleCNN(num_classes=62)
        x = torch.randn(1, 1, 28, 28)
        cam = grad_cam_accessible(model, x, target_class=0)
        assert cam.shape == (28, 28)

    def test_saliency_shape(self):
        model = AccessibleCNN(num_classes=62)
        x = torch.randn(1, 1, 28, 28)
        sal = saliency_accessible(model, x, pred_class=0)
        assert sal.shape == (28, 28)


class TestSerialization:
    def test_roundtrip(self):
        model = AccessibleCNN(num_classes=62)
        model.eval()
        state = model_state_bytes(model)
        loaded = load_model_from_bytes(state, num_classes=62)
        loaded.eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            orig = model(x)
            reloaded = loaded(x)
        assert torch.allclose(orig, reloaded)
