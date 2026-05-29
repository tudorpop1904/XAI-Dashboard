"""
cnn_registry.py — Registry of PyTorch CNN architectures for EMNIST character recognition.
"""

from __future__ import annotations

from core.accessible_cnn import AccessibleCNN, AccessibleCNNDeep, AccessibleCNNLight

CNN_REGISTRY = {
    "AccessibleCNN (Default)": {
        "class": AccessibleCNN,
        "description": "Standard 3-layer CNN. Great balance between training speed and predictive performance.",
    },
    "AccessibleCNN-Deep": {
        "class": AccessibleCNNDeep,
        "description": "4-layer deep CNN with residual skip connections. Highly accurate but slower to train.",
    },
    "AccessibleCNN-Light": {
        "class": AccessibleCNNLight,
        "description": "Sleek 2-layer CNN. Trains incredibly fast, making it ideal for quick demos.",
    },
}
