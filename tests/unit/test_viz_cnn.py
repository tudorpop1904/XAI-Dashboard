"""Tests for core.viz_math_cnn module (CNN architecture, training, XAI)."""

import torch

from core.viz_math_cnn import (
    generate_synthetic_bundle,
    HandwritingCNN,
    train_model,
    grad_cam_for_image,
    input_saliency,
    model_state_bytes,
    load_model_from_bytes,
)


class TestSyntheticBundle:
    def test_generates_correct_shape(self):
        bundle = generate_synthetic_bundle(num_classes=4, samples_per_class=10, img_size=32)
        assert bundle.train_images.ndim == 4
        assert bundle.train_images.shape[1] == 1  # single channel
        assert bundle.train_images.shape[2] == 32
        assert len(bundle.class_map) == 4

    def test_class_map_has_expressions_and_answers(self):
        bundle = generate_synthetic_bundle(num_classes=4, samples_per_class=10)
        for expr, ans in bundle.class_map:
            assert isinstance(expr, str)
            assert isinstance(ans, str)


class TestHandwritingCNN:
    def test_forward_shape(self):
        model = HandwritingCNN(num_classes=10)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_with_conv_returns_both(self):
        model = HandwritingCNN(num_classes=10)
        x = torch.randn(1, 1, 64, 64)
        logits, conv = model.forward_with_conv(x)
        assert logits.shape == (1, 10)
        assert conv.ndim == 4


class TestTraining:
    def test_train_small_bundle(self):
        bundle = generate_synthetic_bundle(num_classes=4, samples_per_class=20, img_size=32)
        model, acc, history = train_model(bundle, epochs=2, batch_size=16)
        assert model is not None
        assert 0.0 <= acc <= 1.0
        assert len(history["loss"]) == 2


class TestXAI:
    def test_grad_cam_shape(self):
        model = HandwritingCNN(num_classes=4)
        x = torch.randn(1, 1, 64, 64)
        cam = grad_cam_for_image(model, x, target_class=0, img_hw=(64, 64))
        assert cam.shape == (64, 64)
        assert cam.min() >= 0.0

    def test_saliency_shape(self):
        model = HandwritingCNN(num_classes=4)
        x = torch.randn(1, 1, 64, 64)
        sal = input_saliency(model, x, pred_class=0)
        assert sal.shape == (64, 64)
        assert sal.min() >= 0.0


class TestSerialization:
    def test_roundtrip(self):
        model = HandwritingCNN(num_classes=4)
        model.eval()
        state = model_state_bytes(model)
        loaded = load_model_from_bytes(state, num_classes=4)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            orig = model(x)
            reloaded = loaded(x)
        assert torch.allclose(orig, reloaded)
