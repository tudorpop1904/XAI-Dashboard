"""
fake_detector.py — Binary CNN classifier: Real vs AI-generated images.

Provides training, inference, Grad-CAM and input saliency (white-box baselines
for comparison with black-box perturbation methods).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from core.fake_data import CLASS_NAMES, FakeImageBundle


@dataclass
class DetectionResult:
    label: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class FakeDetectorCNN(nn.Module):
    """RGB CNN with spatial conv output for Grad-CAM."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_conv(x)
        return logits

    def forward_with_conv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.block(x)
        pooled = self.gap(conv_out)
        logits = self.fc(self.dropout(pooled.flatten(1)))
        return logits, conv_out


def train_detector(
    bundle: FakeImageBundle,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> tuple[FakeDetectorCNN, float, dict]:
    torch.manual_seed(seed)
    device = torch.device(device)
    model = FakeDetectorCNN(len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(bundle.train_images, bundle.train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    history: dict = {"loss": []}
    model.train()
    for _ in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(xb)
        history["loss"].append(ep_loss / len(bundle.train_labels))

    model.eval()
    with torch.no_grad():
        vx = bundle.val_images.to(device)
        vy = bundle.val_labels.to(device)
        acc = float((model(vx).argmax(dim=1) == vy).float().mean().item())
    return model, acc, history


def pil_to_tensor(img: Image.Image, img_size: int) -> torch.Tensor:
    """(1, 3, H, W) float tensor in [0, 1]."""
    rgb = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def predict(
    model: FakeDetectorCNN,
    x: torch.Tensor,
    device: str | torch.device = "cpu",
) -> DetectionResult:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
    prob_dict = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
    return DetectionResult(
        label=CLASS_NAMES[idx],
        class_index=idx,
        confidence=conf,
        probabilities=prob_dict,
    )


def target_probability(
    model: FakeDetectorCNN,
    x: torch.Tensor,
    target_class: int,
    device: str | torch.device = "cpu",
) -> float:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        return float(torch.softmax(logits, dim=1)[0, target_class].item())


def grad_cam(
    model: FakeDetectorCNN,
    x: torch.Tensor,
    target_class: int,
    img_hw: tuple[int, int],
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.eval()
    x = x.to(device).detach().clone().requires_grad_(True)
    logits, conv = model.forward_with_conv(x)
    conv.retain_grad()
    model.zero_grad(set_to_none=True)
    logits[0, target_class].backward(retain_graph=False)
    g = conv.grad
    if g is None:
        return np.zeros(img_hw, dtype=np.float32)

    conv_out = conv[0].detach()
    conv_grad = g[0].detach()
    weights = conv_grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * conv_out).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    if float(cam.max()) > 1e-8:
        cam = cam / cam.max()
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=img_hw,
        mode="bilinear",
        align_corners=False,
    )
    return cam_up.squeeze().detach().cpu().numpy()


def saliency(
    model: FakeDetectorCNN,
    x: torch.Tensor,
    target_class: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.eval()
    x = x.to(device).detach().clone().requires_grad_(True)
    logits, _ = model.forward_with_conv(x)
    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    score.backward()
    sal = x.grad.abs().mean(dim=1).squeeze(0)
    sal = sal - sal.min()
    if float(sal.max()) > 1e-8:
        sal = sal / sal.max()
    return sal.detach().cpu().numpy()


def model_state_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def load_model_from_bytes(
    state_bytes: bytes,
    map_location: str | torch.device = "cpu",
) -> FakeDetectorCNN:
    model = FakeDetectorCNN(len(CLASS_NAMES))
    buf = io.BytesIO(state_bytes)
    model.load_state_dict(torch.load(buf, map_location=map_location, weights_only=True))
    model.eval()
    return model
