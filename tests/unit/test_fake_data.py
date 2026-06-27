"""Unit tests for synthetic fake-image data generation."""

from core.fake_data import CLASS_NAMES, LABEL_AI, LABEL_REAL, generate_synthetic_bundle


def test_class_names():
    assert len(CLASS_NAMES) == 2
    assert CLASS_NAMES[LABEL_REAL] == "Real"
    assert CLASS_NAMES[LABEL_AI] == "AI-Generated"


def test_bundle_shapes():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    bundle = generate_synthetic_bundle(samples_per_class=20, img_size=64)
    assert bundle.train_images.shape[1:] == (3, 64, 64)
    assert bundle.val_images.shape[0] >= 2
    assert set(bundle.train_labels.tolist()) <= {0, 1}
