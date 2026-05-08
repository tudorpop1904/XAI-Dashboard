"""
viz_math_cnn.py — Synthetic handwritten “equation” images, small CNN,
and visualization XAI (Grad-CAM + input saliency).

Scope: finite set of simple integer expressions (e.g. 3+5). The CNN
classifies which expression the handwriting depicts; the numeric answer
is looked up from metadata (Photomath-style read + solve for a closed set).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


@dataclass
class VizMathBundle:
    """Everything needed to train, infer, and explain."""

    class_map: List[Tuple[str, str]]  # (expression_str, answer_str)
    img_size: int
    train_images: torch.Tensor  # (N, 1, H, W) float 0–1
    train_labels: torch.Tensor  # (N,) long
    val_images: torch.Tensor
    val_labels: torch.Tensor


def _default_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_class_map(num_classes: int, seed: int) -> List[Tuple[str, str]]:
    """Fixed vocabulary of simple expressions and their answers."""
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add(expr: str, ans: str) -> None:
        if expr in seen or len(pairs) >= num_classes:
            return
        seen.add(expr)
        pairs.append((expr, ans))

    for a in range(0, 10):
        for b in range(0, 10):
            add(f"{a}+{b}", str(a + b))
            add(f"{a}-{b}", str(a - b))
            if a <= 9 and b <= 9:
                add(f"{a}×{b}", str(a * b))
            if len(pairs) >= num_classes:
                break
        if len(pairs) >= num_classes:
            break
    k = 0
    while len(pairs) < num_classes:
        add(f"e{k}+1", str(k + 1))
        k += 1
    rng.shuffle(pairs)
    return pairs[:num_classes]


def render_expression_image(
    expr: str,
    img_size: int,
    rng: random.Random,
) -> np.ndarray:
    """Render a noisy grayscale image; ink ≈ higher values."""
    img = Image.new("L", (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    font_size = rng.randint(max(12, img_size // 5), max(14, img_size // 3))
    font = _default_font(font_size)
    bbox = draw.textbbox((0, 0), expr, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, (img_size - tw) // 2 + rng.randint(-3, 3))
    y = max(0, (img_size - th) // 2 + rng.randint(-3, 3))
    draw.text((x, y), expr, fill=0, font=font)
    for _ in range(rng.randint(0, 6)):
        x0, y0 = rng.randint(0, img_size - 1), rng.randint(0, img_size - 1)
        x1, y1 = rng.randint(0, img_size - 1), rng.randint(0, img_size - 1)
        draw.line([(x0, y0), (x1, y1)], fill=rng.randint(180, 255), width=1)
    img = img.rotate(rng.uniform(-18.0, 18.0), fillcolor=255, resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    noise = rng.gauss(0, 0.04)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return arr


def generate_synthetic_bundle(
    num_classes: int = 28,
    samples_per_class: int = 120,
    img_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> VizMathBundle:
    class_map = _build_class_map(num_classes, seed)
    rng = random.Random(seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for cls_idx, (expr, _) in enumerate(class_map):
        for _ in range(samples_per_class):
            images.append(render_expression_image(expr, img_size, rng))
            labels.append(cls_idx)
    idx = list(range(len(images)))
    rng.shuffle(idx)
    images = [images[i] for i in idx]
    labels = [labels[i] for i in idx]
    n = len(images)
    n_val = max(1, int(round(n * val_fraction)))
    split = max(1, n - n_val)
    train_x = np.stack(images[:split], axis=0)[:, None, :, :]
    val_x = np.stack(images[split:], axis=0)[:, None, :, :]
    train_y = np.array(labels[:split], dtype=np.int64)
    val_y = np.array(labels[split:], dtype=np.int64)
    return VizMathBundle(
        class_map=class_map,
        img_size=img_size,
        train_images=torch.from_numpy(train_x),
        train_labels=torch.from_numpy(train_y),
        val_images=torch.from_numpy(val_x),
        val_labels=torch.from_numpy(val_y),
    )


class HandwritingCNN(nn.Module):
    """Small CNN with one spatial conv block output for Grad-CAM."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_conv(x)
        return logits

    def forward_with_conv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.block(x)
        pooled = self.gap(conv_out)
        logits = self.fc(pooled.flatten(1))
        return logits, conv_out


def train_model(
    bundle: VizMathBundle,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> Tuple[HandwritingCNN, float, dict]:
    torch.manual_seed(seed)
    device = torch.device(device)
    n_cls = len(bundle.class_map)
    model = HandwritingCNN(n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(bundle.train_images, bundle.train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    history: dict = {"loss": []}
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(xb)
        history["loss"].append(ep_loss / len(bundle.train_labels))
    model.eval()
    with torch.no_grad():
        vx = bundle.val_images.to(device)
        vy = bundle.val_labels.to(device)
        pred = model(vx).argmax(dim=1)
        acc = float((pred == vy).float().mean().item())
    return model, acc, history


def pil_to_tensor_gray(img: Image.Image, img_size: int) -> torch.Tensor:
    """Single image (1,1,H,W) float in [0,1], same convention as training."""
    g = img.convert("L").resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t


def predict_class(
    model: HandwritingCNN,
    x: torch.Tensor,
    device: str | torch.device = "cpu",
) -> int:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        return int(logits.argmax(dim=1).item())


def grad_cam_on_conv(
    conv_out: torch.Tensor,
    conv_grad: torch.Tensor,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """
    conv_out, conv_grad: (C, H, W)
    Returns upsampled heatmap in [0,1], shape target_hw.
    """
    c, h, w = conv_out.shape
    weights = conv_grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * conv_out).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    if float(cam.max()) > 1e-8:
        cam = cam / cam.max()
    cam_up = F.interpolate(
        cam.view(1, 1, h, w),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return cam_up.view(target_hw[0], target_hw[1]).detach().cpu().numpy()


def input_saliency(
    model: HandwritingCNN,
    x: torch.Tensor,
    pred_class: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Vanilla gradients |d logit / d input| on the image plane."""
    device = torch.device(device)
    model.eval()
    x = x.to(device).detach().clone().requires_grad_(True)
    logits, _ = model.forward_with_conv(x)
    score = logits[0, pred_class]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    score.backward()
    sal = x.grad.abs().squeeze(0).squeeze(0)
    sal = sal - sal.min()
    if float(sal.max()) > 1e-8:
        sal = sal / sal.max()
    return sal.detach().cpu().numpy()


def grad_cam_for_image(
    model: HandwritingCNN,
    x: torch.Tensor,
    target_class: int,
    img_hw: Tuple[int, int],
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
    return grad_cam_on_conv(conv[0].detach(), g[0].detach(), img_hw)


def class_to_solution(class_map: Sequence[Tuple[str, str]], cls: int) -> Tuple[str, str]:
    expr, ans = class_map[cls]
    return expr, ans


def model_state_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def load_model_from_bytes(
    state_bytes: bytes,
    num_classes: int,
    map_location: str | torch.device = "cpu",
) -> HandwritingCNN:
    model = HandwritingCNN(num_classes)
    buf = io.BytesIO(state_bytes)
    model.load_state_dict(torch.load(buf, map_location=map_location))
    model.eval()
    return model
