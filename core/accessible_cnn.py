"""
accessible_cnn.py — EMNIST-based handwriting recognition for the Accessible Writing Instructor.

Loads the EMNIST ByClass dataset (62 classes: A-Z, a-z, 0-9),
provides canvas preprocessing, and extends the existing HandwritingCNN
architecture for single-character recognition.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─────────────────────────────────────────────
# EMNIST class mapping
# ─────────────────────────────────────────────

# EMNIST ByClass has 62 classes: 0-9 (indices 0–9), A-Z (10–35), a-z (36–61)
# The label mapping follows ASCII order after transformation.


def build_emnist_label_map() -> dict[int, str]:
    """Build mapping from EMNIST ByClass label index → character string."""
    mapping = {}
    for i in range(10):
        mapping[i] = str(i)
    for i in range(26):
        mapping[10 + i] = chr(ord("A") + i)
    for i in range(26):
        mapping[36 + i] = chr(ord("a") + i)
    return mapping


EMNIST_LABEL_MAP = build_emnist_label_map()
NUM_CLASSES = len(EMNIST_LABEL_MAP)  # 62


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────


@dataclass
class EMNISTBundle:
    """Training and validation data for the accessible CNN."""

    train_images: torch.Tensor  # (N, 1, 28, 28) float 0–1
    train_labels: torch.Tensor  # (N,) long
    val_images: torch.Tensor
    val_labels: torch.Tensor
    label_map: dict[int, str]
    num_classes: int


def load_emnist_bundle(
    max_per_class: int = 500,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> EMNISTBundle:
    """
    Load EMNIST ByClass and return a balanced, capped bundle.

    Uses bulk tensor operations for speed (avoids per-sample iteration
    over 800K+ images).
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download EMNIST ByClass (auto-cached in ./data_cache/)
    train_dataset = datasets.EMNIST(
        root="data_cache",
        split="byclass",
        train=True,
        download=True,
        transform=transform,
    )
    val_dataset = datasets.EMNIST(
        root="data_cache",
        split="byclass",
        train=False,
        download=True,
        transform=transform,
    )

    def cap_and_stack(dataset, max_n: int):
        """Balance classes by capping at max_n per class (fast bulk path)."""
        rng = random.Random(seed)

        # Use the raw underlying tensors for speed instead of per-item __getitem__
        if hasattr(dataset, "data") and hasattr(dataset, "targets"):
            raw_data = dataset.data  # (N, 28, 28) uint8
            raw_labels = dataset.targets  # (N,) int
            if not isinstance(raw_labels, torch.Tensor):
                raw_labels = torch.tensor(raw_labels, dtype=torch.long)

            selected_indices = []
            for cls in range(NUM_CLASSES):
                cls_mask = (raw_labels == cls).nonzero(as_tuple=True)[0].tolist()
                rng.shuffle(cls_mask)
                selected_indices.extend(cls_mask[:max_n])

            rng.shuffle(selected_indices)
            idx_tensor = torch.tensor(selected_indices, dtype=torch.long)

            imgs = raw_data[idx_tensor].float() / 255.0  # (N, 28, 28)
            # EMNIST images are stored transposed (X and Y swapped)
            imgs = imgs.transpose(1, 2)
            imgs = imgs.unsqueeze(1)  # (N, 1, 28, 28)
            lbls = raw_labels[idx_tensor]

            return imgs, lbls

        # Fallback: per-item access (slow but always works)
        buckets: dict[int, list[int]] = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label = int(label)
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(idx)

        selected_indices = []
        for label in sorted(buckets.keys()):
            indices = buckets[label]
            rng.shuffle(indices)
            selected_indices.extend(indices[:max_n])
        rng.shuffle(selected_indices)

        images = []
        labels = []
        for idx in selected_indices:
            img, lbl = dataset[idx]
            img = img.transpose(1, 2)
            images.append(img)
            labels.append(int(lbl))

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    train_x, train_y = cap_and_stack(train_dataset, max_per_class)
    val_x, val_y = cap_and_stack(val_dataset, max(50, max_per_class // 3))

    return EMNISTBundle(
        train_images=train_x,
        train_labels=train_y,
        val_images=val_x,
        val_labels=val_y,
        label_map=EMNIST_LABEL_MAP,
        num_classes=NUM_CLASSES,
    )


# ─────────────────────────────────────────────
# CNN Architecture
# ─────────────────────────────────────────────


class AccessibleCNN(nn.Module):
    """
    Small CNN for single-character recognition (62 classes).
    Same Grad-CAM-friendly architecture as HandwritingCNN but tuned for 28×28 EMNIST.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28→14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14→7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_conv(x)
        return logits

    def forward_with_conv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.block(x)
        pooled = self.gap(conv_out)
        flat = self.dropout(pooled.flatten(1))
        logits = self.fc(flat)
        return logits, conv_out


class AccessibleCNNLight(nn.Module):
    """
    Sleek 2-layer CNN for single-character recognition (62 classes).
    Trains incredibly fast, making it ideal for quick demos.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28→14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14→7
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_conv(x)
        return logits

    def forward_with_conv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.block(x)
        pooled = self.gap(conv_out)
        flat = self.dropout(pooled.flatten(1))
        logits = self.fc(flat)
        return logits, conv_out


class AccessibleCNNDeep(nn.Module):
    """
    4-layer deep CNN with residual-like skip connections for single-character recognition (62 classes).
    Highly accurate but slower to train.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Residual-like layers
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_conv(x)
        return logits

    def forward_with_conv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Layer 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)  # 28→14

        # Layer 2
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)  # 14→7

        # Layer 3 & 4 (with residual/skip logic)
        res = out
        out = F.relu(self.bn3(self.conv3(out)))
        out = out + res  # Skip connection

        conv_out = F.relu(self.bn4(self.conv4(out)))

        pooled = self.gap(conv_out)
        flat = self.dropout(pooled.flatten(1))
        logits = self.fc(flat)
        return logits, conv_out


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────


def train_accessible_model(
    bundle: EMNISTBundle,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    progress_callback=None,
) -> tuple[AccessibleCNN, float, dict]:
    """
    Train the AccessibleCNN on EMNIST.

    Returns (model, val_accuracy, history_dict).
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model = AccessibleCNN(bundle.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(bundle.train_images, bundle.train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    history: dict = {"loss": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(xb)

        avg_loss = ep_loss / len(bundle.train_labels)
        history["loss"].append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            vx = bundle.val_images.to(device)
            vy = bundle.val_labels.to(device)
            pred = model(vx).argmax(dim=1)
            acc = float((pred == vy).float().mean().item())
        history["val_acc"].append(acc)

        if progress_callback:
            progress_callback(ep + 1, epochs, avg_loss, acc)

    return model, acc, history


# ─────────────────────────────────────────────
# Canvas preprocessing
# ─────────────────────────────────────────────


def _crop_to_content(gray: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Crop a grayscale image to the bounding box of non-zero content,
    then re-center in a square with padding.

    Without this, a small character drawn on a 300×300 canvas would be
    squished into a ~3px blob after resize to 28×28.
    """
    # Find bounding box of ink
    rows = np.any(gray > 0.05, axis=1)
    cols = np.any(gray > 0.05, axis=0)

    if not rows.any() or not cols.any():
        return gray  # Nothing drawn

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop
    cropped = gray[rmin : rmax + 1, cmin : cmax + 1]

    # Make square (pad shorter dimension)
    h, w = cropped.shape
    size = max(h, w) + 2 * padding
    square = np.zeros((size, size), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = cropped

    return square


def canvas_to_tensor(canvas_data: np.ndarray) -> torch.Tensor:
    """
    Convert streamlit-drawable-canvas output to a model-ready tensor.

    The canvas gives an RGBA numpy array (H, W, 4).
    We draw WHITE strokes on a BLACK background, so the drawing lives
    in the RGB channels (not alpha — both bg and strokes are opaque).

    Steps:
    1. Extract grayscale from RGB (ignore alpha)
    2. Crop to the bounding box of the ink
    3. Re-center in a padded square
    4. Resize to 28×28
    5. Return as (1, 1, 28, 28) float tensor
    """
    if canvas_data is None:
        return torch.zeros(1, 1, 28, 28)

    # Extract grayscale from RGB channels (not alpha!)
    if canvas_data.ndim == 3 and canvas_data.shape[2] >= 3:
        # Average the RGB channels to get grayscale intensity
        gray = np.mean(canvas_data[:, :, :3], axis=2).astype(np.float32) / 255.0
    else:
        gray = canvas_data.astype(np.float32) / 255.0

    # Crop to content bounding box + re-center
    gray = _crop_to_content(gray, padding=20)

    # Resize to 28×28 (EMNIST native size)
    pil = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    pil = pil.resize((28, 28), Image.BICUBIC)

    arr = np.asarray(pil, dtype=np.float32) / 255.0

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def pil_to_emnist_tensor(img: Image.Image) -> torch.Tensor:
    """Convert an uploaded PIL image to EMNIST-format tensor (1,1,28,28)."""
    g = img.convert("L").resize((28, 28), Image.BICUBIC)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # Invert: EMNIST convention is white-on-black
    # Crop to content if mostly empty
    arr = _crop_to_content(arr, padding=4)
    # Re-resize after crop
    pil2 = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil2 = pil2.resize((28, 28), Image.BICUBIC)
    arr = np.asarray(pil2, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict_character(
    model: AccessibleCNN,
    x: torch.Tensor,
    label_map: dict[int, str],
    device: str | torch.device = "cpu",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Predict character from tensor input.

    Returns list of (character, confidence) tuples, sorted by confidence.
    """
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        char = label_map.get(idx, "?")
        results.append((char, round(prob, 4)))
    return results


# ─────────────────────────────────────────────
# Grad-CAM + Saliency (reuse patterns from viz_math_cnn)
# ─────────────────────────────────────────────


def grad_cam_accessible(
    model: AccessibleCNN,
    x: torch.Tensor,
    target_class: int,
    img_hw: tuple[int, int] = (28, 28),
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Grad-CAM for AccessibleCNN. Returns heatmap in [0,1]."""
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


def saliency_accessible(
    model: AccessibleCNN,
    x: torch.Tensor,
    pred_class: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Vanilla gradient saliency for AccessibleCNN."""
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


# ─────────────────────────────────────────────
# Model serialization
# ─────────────────────────────────────────────


def model_state_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def load_model_from_bytes(
    state_bytes: bytes,
    num_classes: int = NUM_CLASSES,
    map_location: str | torch.device = "cpu",
) -> AccessibleCNN:
    model = AccessibleCNN(num_classes)
    buf = io.BytesIO(state_bytes)
    model.load_state_dict(torch.load(buf, map_location=map_location))
    model.eval()
    return model
