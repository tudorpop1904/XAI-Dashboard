import torch

from core.image_features import fft_channel, lbp_channel, magnitude_channel


def test_features():

    x = torch.rand(3, 224, 224)

    fft = fft_channel(x)
    lbp = lbp_channel(x)
    mag = magnitude_channel(x)

    assert fft.shape == (1, 224, 224)
    assert lbp.shape == (1, 224, 224)
    assert mag.shape == (1, 224, 224)

    assert fft.min() >= 0
    assert fft.max() <= 1

    assert lbp.min() >= 0
    assert lbp.max() <= 1

    assert mag.min() >= 0
    assert mag.max() <= 1
