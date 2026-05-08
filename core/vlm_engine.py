"""
vlm_engine.py — Vision-Language Model engine for handwritten math review.

Uses a multimodal model served through Ollama (e.g. Qwen2.5-VL) to:
  1. Read handwritten math from one or more uploaded images.
  2. Transcribe into LaTeX.
  3. Evaluate the mathematical reasoning (correctness check).
  4. Provide step-by-step feedback and practice problems.

The architecture is designed around a generic "compute backend" interface
so that the VLM inference can later be offloaded to external hardware
(e.g. a Raspberry Pi cluster or any HTTP-reachable compute node) without
changing any page-level code.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, List, Optional, Tuple

import ollama
from PIL import Image


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# The VLM model used for vision tasks (reading images).
# Qwen2.5-VL is an open-source multimodal model with strong OCR performance.
VLM_MODEL: str = os.environ.get("VLM_MODEL", "qwen2.5vl:7b")

# A lightweight VLM used as a *surrogate* for XAI probing (occlusion
# sensitivity).  For XAI, we don't need accuracy — just consistency.
# A fast model that reliably changes its output when regions are masked
# produces equally valid heatmaps.  Falls back to VLM_MODEL if unset.
XAI_SURROGATE_MODEL: str = os.environ.get("XAI_SURROGATE_MODEL", "minicpm-v")

# Ollama endpoint — same as the text LLM by default.
VLM_HOST: str = os.environ.get("VLM_HOST", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

# The text-only LLM used for reasoning / evaluation after transcription.
EVAL_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Remote compute backend (future: Raspberry Pi, etc.).
# When set, VLM inference requests are forwarded to this host instead of
# the local Ollama instance.  Format: "http://<ip>:<port>"
REMOTE_VLM_HOST: str | None = os.environ.get("REMOTE_VLM_HOST", None)


def _get_vlm_client() -> ollama.Client:
    """Return an Ollama client pointed at the appropriate VLM backend."""
    host = REMOTE_VLM_HOST if REMOTE_VLM_HOST else VLM_HOST
    return ollama.Client(host=host)


def _get_eval_client() -> ollama.Client:
    """Return an Ollama client for the text-only evaluation LLM."""
    return ollama.Client(host=VLM_HOST)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

class Verdict(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    UNCLEAR = "unclear"


@dataclass
class TranscriptionResult:
    """Result of reading one or more pages of handwritten math."""
    latex: str                         # Full LaTeX transcription
    raw_text: str                      # Raw model output before cleanup
    page_count: int = 1
    elapsed_seconds: float = 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating the mathematical reasoning."""
    verdict: Verdict
    summary: str                       # One-paragraph plain-English summary
    step_by_step: str                  # Detailed step-by-step analysis
    errors: List[str] = field(default_factory=list)  # Specific errors found
    corrected_latex: str = ""          # Corrected version (if errors exist)
    practice_problems: str = ""        # Suggested practice problems
    elapsed_seconds: float = 0.0


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to raw bytes for the Ollama API."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    return base64.b64encode(pil_to_bytes(img, fmt)).decode("utf-8")


def _prepare_images(
    images: List[Image.Image],
    max_dim: int = 1536,
) -> List[bytes]:
    """Prepare a list of PIL images as byte blobs for Ollama."""
    prepared: List[bytes] = []
    for img in images:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # Convert to RGB if necessary (e.g. RGBA PNGs)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        prepared.append(pil_to_bytes(img, "PNG"))
    return prepared


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_TRANSCRIPTION_SYSTEM = (
    "You are a specialist mathematical OCR assistant. Your job is to read "
    "handwritten mathematics from photographs of notebook pages and produce "
    "a faithful, structured transcription.\n\n"
    "CRITICAL FORMATTING RULES — follow these exactly:\n\n"
    "1. Output ONLY the transcription — no commentary, no greetings, "
    "   no explanations of what you are doing.\n"
    "2. ALWAYS use dollar-sign delimiters for math mode:\n"
    "   - Inline math: $x^2 + 1 = 0$\n"
    "   - Display math: $$\\int_0^1 f(x)\\,dx$$\n"
    "   NEVER use \\( ... \\) or \\[ ... \\] or bare parentheses for math.\n"
    "3. Preserve the logical order of the argument. Number the steps.\n"
    "4. Transcribe verbal / textual reasoning as plain English.\n"
    "5. If something is illegible, write [illegible] in that spot.\n"
    "6. If multiple images are given, they are consecutive pages of the "
    "   same solution — combine them into one continuous transcription.\n"
    "7. INTERMEDIATE CALCULATIONS (e.g. polynomial long division, "
    "   synthetic division, scratch-work multiplication, trial-and-error "
    "   substitutions): Do NOT attempt to reproduce their visual layout "
    "   in LaTeX.  Instead, write a brief annotation like:\n"
    "   % [Intermediate calculation: polynomial long division of f(z) by (z-1)]\n"
    "   and then state the RESULT of that calculation as a normal equation. "
    "   For example:\n"
    "   % [Intermediate calculation: polynomial long division of "
    "   z^3 - 3z^2 + 7z - 5 by (z - 1)]\n"
    "   Performing polynomial long division yields $f(z) = (z - 1)(z^2 - 2z + 5)$.\n"
    "8. Use standard LaTeX commands: \\frac{}{}, \\sqrt{}, \\in, \\mathbb{C}, etc.\n"
)

_TRANSCRIPTION_USER = (
    "Transcribe the handwritten mathematics in the attached image(s). "
    "Use $...$ for inline math and $$...$$ for display math — NEVER use "
    "\\(\\) or \\[\\] or bare parentheses. "
    "For intermediate calculations (long division, scratch work), "
    "describe them with a comment and state the result. "
    "Preserve the logical flow."
)


def _build_evaluation_prompt(latex: str) -> str:
    """Build the prompt for the evaluation LLM to check the math."""
    return (
        "You are a rigorous but encouraging mathematics tutor and grader. "
        "A student has submitted the following handwritten solution (transcribed "
        "into LaTeX). Your tasks:\n\n"
        "1. **Verify** every mathematical step for correctness.\n"
        "2. **Identify** the first error (if any) and explain clearly what went "
        "   wrong and why.\n"
        "3. **Provide a corrected solution** if errors exist.\n"
        "4. **Classify** the overall verdict as one of: CORRECT, INCORRECT, "
        "   PARTIALLY_CORRECT, or UNCLEAR.\n"
        "5. If the solution is correct, congratulate the student briefly and "
        "   suggest 2–3 similar practice problems.\n"
        "6. If the solution is incorrect or partially correct, after pointing "
        "   out errors, also suggest 2–3 practice problems targeting the "
        "   weak spots.\n\n"
        "Format your response EXACTLY as follows (keep the headers):\n\n"
        "## Verdict\n<one of: CORRECT / INCORRECT / PARTIALLY_CORRECT / UNCLEAR>\n\n"
        "## Summary\n<one-paragraph plain-English summary>\n\n"
        "## Step-by-Step Analysis\n<detailed analysis of each step>\n\n"
        "## Errors Found\n<bullet list of errors, or 'None' if correct>\n\n"
        "## Corrected Solution\n<corrected LaTeX, or 'N/A' if already correct>\n\n"
        "## Practice Problems\n<2–3 similar problems for the student>\n\n"
        "---\n\n"
        "**Student's solution (LaTeX transcription):**\n\n"
        f"{latex}\n"
    )


# ─────────────────────────────────────────────
# VLM health checks
# ─────────────────────────────────────────────

def check_vlm_available() -> Tuple[bool, str]:
    """
    Check whether the VLM model is available on Ollama.

    Returns
    -------
    ok : bool
    message : str
    """
    try:
        client = _get_vlm_client()
        models = client.list()
        model_names = [m.model for m in models.models]
        prefix = VLM_MODEL.split(":")[0]
        matches = [n for n in model_names if n.startswith(prefix)]
        if matches:
            return True, f"VLM ready: `{matches[0]}` on `{REMOTE_VLM_HOST or VLM_HOST}`"
        else:
            return False, (
                f"VLM model `{VLM_MODEL}` not found. "
                f"Pull it with: `ollama pull {VLM_MODEL}`"
            )
    except Exception as e:
        return False, f"Cannot reach VLM backend: {e}"


def check_eval_available() -> Tuple[bool, str]:
    """Check whether the text evaluation LLM is available."""
    try:
        client = _get_eval_client()
        models = client.list()
        model_names = [m.model for m in models.models]
        prefix = EVAL_MODEL.split(":")[0]
        matches = [n for n in model_names if n.startswith(prefix)]
        if matches:
            return True, f"Eval LLM ready: `{matches[0]}`"
        else:
            return False, (
                f"Eval model `{EVAL_MODEL}` not found. "
                f"Pull it with: `ollama pull {EVAL_MODEL}`"
            )
    except Exception as e:
        return False, f"Cannot reach eval LLM: {e}"


# ─────────────────────────────────────────────
# Core pipeline functions
# ─────────────────────────────────────────────

def transcribe_images(images: List[Image.Image]) -> TranscriptionResult:
    """
    Send one or more images to the VLM and get a LaTeX transcription.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Ordered list of notebook page photos (page 1 first).

    Returns
    -------
    TranscriptionResult
    """
    client = _get_vlm_client()
    image_bytes = _prepare_images(images)

    t0 = time.time()
    response = client.chat(
        model=VLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": _TRANSCRIPTION_SYSTEM,
            },
            {
                "role": "user",
                "content": _TRANSCRIPTION_USER,
                "images": image_bytes,
            },
        ],
    )
    elapsed = time.time() - t0

    raw = response["message"]["content"]

    return TranscriptionResult(
        latex=raw.strip(),
        raw_text=raw,
        page_count=len(images),
        elapsed_seconds=round(elapsed, 2),
    )


def transcribe_images_stream(
    images: List[Image.Image],
) -> Generator[str, None, TranscriptionResult]:
    """
    Streaming variant — yields token chunks as they arrive.
    The final return value is the full TranscriptionResult.
    """
    client = _get_vlm_client()
    image_bytes = _prepare_images(images)

    t0 = time.time()
    stream = client.chat(
        model=VLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": _TRANSCRIPTION_SYSTEM,
            },
            {
                "role": "user",
                "content": _TRANSCRIPTION_USER,
                "images": image_bytes,
            },
        ],
        stream=True,
    )

    chunks: list[str] = []
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            chunks.append(token)
            yield token

    elapsed = time.time() - t0
    full_text = "".join(chunks)

    return TranscriptionResult(
        latex=full_text.strip(),
        raw_text=full_text,
        page_count=len(images),
        elapsed_seconds=round(elapsed, 2),
    )


def evaluate_solution(latex: str) -> EvaluationResult:
    """
    Send the transcribed LaTeX to the evaluation LLM for grading.

    Parameters
    ----------
    latex : str
        LaTeX transcription of the student's work.

    Returns
    -------
    EvaluationResult
    """
    client = _get_eval_client()
    prompt = _build_evaluation_prompt(latex)

    t0 = time.time()
    response = client.chat(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0

    raw = response["message"]["content"]
    return _parse_evaluation(raw, elapsed)


def evaluate_solution_stream(
    latex: str,
) -> Generator[str, None, EvaluationResult]:
    """Streaming variant of evaluate_solution."""
    client = _get_eval_client()
    prompt = _build_evaluation_prompt(latex)

    t0 = time.time()
    stream = client.chat(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    chunks: list[str] = []
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            chunks.append(token)
            yield token

    elapsed = time.time() - t0
    full_text = "".join(chunks)

    return _parse_evaluation(full_text, elapsed)


def _parse_evaluation(raw: str, elapsed: float) -> EvaluationResult:
    """Parse structured LLM output into an EvaluationResult."""
    # Best-effort section parsing
    sections = {
        "verdict": "",
        "summary": "",
        "step_by_step": "",
        "errors": "",
        "corrected": "",
        "practice": "",
    }

    # Map header keywords to section keys
    header_map = {
        "verdict": "verdict",
        "summary": "summary",
        "step-by-step": "step_by_step",
        "step by step": "step_by_step",
        "errors found": "errors",
        "errors": "errors",
        "corrected solution": "corrected",
        "corrected": "corrected",
        "practice problems": "practice",
        "practice": "practice",
    }

    current_key: Optional[str] = None
    for line in raw.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        lower = stripped.lower()
        matched = False
        for header, key in header_map.items():
            if lower.startswith(header):
                current_key = key
                # Capture any inline content after the header
                rest = stripped[len(header):].strip().lstrip(":").strip()
                if rest:
                    sections[current_key] = rest + "\n"
                matched = True
                break
        if not matched and current_key is not None:
            sections[current_key] += line + "\n"

    # Parse verdict
    v_str = sections["verdict"].strip().upper()
    verdict = Verdict.UNCLEAR
    for v in Verdict:
        if v.value.upper() in v_str:
            verdict = v
            break

    # Parse errors list
    error_lines: List[str] = []
    for line in sections["errors"].strip().split("\n"):
        line = line.strip().lstrip("-•*").strip()
        if line and line.lower() != "none" and line.lower() != "n/a":
            error_lines.append(line)

    return EvaluationResult(
        verdict=verdict,
        summary=sections["summary"].strip(),
        step_by_step=sections["step_by_step"].strip(),
        errors=error_lines,
        corrected_latex=sections["corrected"].strip(),
        practice_problems=sections["practice"].strip(),
        elapsed_seconds=round(elapsed, 2),
    )


# ─────────────────────────────────────────────
# Occlusion Sensitivity XAI
# ─────────────────────────────────────────────

import numpy as np


@dataclass
class OcclusionResult:
    """Result of occlusion sensitivity analysis."""
    heatmap: np.ndarray             # (H, W) float in [0, 1]
    grid_rows: int
    grid_cols: int
    baseline_text: str              # Transcription of the unoccluded image
    cell_texts: List[str]           # Transcription per occluded cell
    cell_similarities: List[float]  # Similarity score per cell (1 = identical)
    surrogate_model: str            # Which model was used for probing
    elapsed_seconds: float = 0.0


def _normalised_levenshtein(a: str, b: str) -> float:
    """
    Normalised Levenshtein similarity in [0, 1].
    1.0 = identical strings, 0.0 = completely different.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Standard DP Levenshtein
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _occlude_region(
    img: Image.Image,
    row: int,
    col: int,
    grid_rows: int,
    grid_cols: int,
    fill: int = 200,
) -> Image.Image:
    """Return a copy of img with the (row, col) grid cell filled with gray."""
    w, h = img.size
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    x0 = col * cell_w
    y0 = row * cell_h
    x1 = x0 + cell_w if col < grid_cols - 1 else w
    y1 = y0 + cell_h if row < grid_rows - 1 else h
    occluded = img.copy()
    from PIL import ImageDraw as _IDraw
    draw = _IDraw.Draw(occluded)
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    return occluded


def _transcribe_for_xai(
    images: List[Image.Image],
    model: str,
) -> str:
    """
    Lightweight transcription call used only for XAI probing.

    Uses a potentially different (faster) model and lower resolution
    to minimize latency per call.  Accuracy doesn't matter here —
    only *consistency* matters (same input should give similar output,
    different input should give measurably different output).
    """
    client = _get_vlm_client()
    # Lower resolution for speed — 768px is enough for change detection
    image_bytes = _prepare_images(images, max_dim=768)

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _TRANSCRIPTION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": _TRANSCRIPTION_USER,
                    "images": image_bytes,
                },
            ],
        )
        return response["message"]["content"].strip()
    except Exception:
        return ""


def check_surrogate_available() -> Tuple[bool, str]:
    """Check whether the XAI surrogate model is available."""
    try:
        client = _get_vlm_client()
        models = client.list()
        model_names = [m.model for m in models.models]
        prefix = XAI_SURROGATE_MODEL.split(":")[0]
        matches = [n for n in model_names if n.startswith(prefix)]
        if matches:
            return True, f"XAI surrogate ready: `{matches[0]}`"
        else:
            return False, (
                f"XAI surrogate `{XAI_SURROGATE_MODEL}` not found. "
                f"Pull it with: `ollama pull {XAI_SURROGATE_MODEL}` "
                f"— or set XAI_SURROGATE_MODEL to a model you have."
            )
    except Exception as e:
        return False, f"Cannot check surrogate: {e}"


def occlusion_sensitivity(
    images: List[Image.Image],
    baseline_text: str,
    grid_rows: int = 3,
    grid_cols: int = 3,
    use_surrogate: bool = True,
    progress_callback=None,
) -> OcclusionResult:
    """
    Occlusion Sensitivity for VLM transcription (model-agnostic Visual XAI).

    For each cell in a (grid_rows x grid_cols) grid overlaid on the FIRST
    image, we gray-out that cell, re-run a VLM, and measure how much the
    transcription changes compared to baseline_text.

    When use_surrogate=True (default), the fast surrogate model is used
    for probing calls.  This is valid because occlusion sensitivity only
    measures *relative change* — it doesn't need accurate transcription,
    just consistent behaviour.

    Parameters
    ----------
    images : list of PIL images (only the first is perturbed)
    baseline_text : str — the original transcription (avoids re-running)
    grid_rows, grid_cols : grid resolution
    use_surrogate : bool — if True, use XAI_SURROGATE_MODEL for probing
    progress_callback : callable(step, total) — for progress bars

    Returns
    -------
    OcclusionResult with a heatmap sized to the first image.
    """
    total_cells = grid_rows * grid_cols
    img0 = images[0]
    # Ensure RGB for drawing
    if img0.mode not in ("RGB", "L"):
        img0 = img0.convert("RGB")

    # Decide which model to use for probing
    probe_model = XAI_SURROGATE_MODEL if use_surrogate else VLM_MODEL

    # If using surrogate, we need a surrogate baseline (the surrogate
    # "reads" the original image with its own vocabulary/style)
    if use_surrogate:
        surrogate_baseline = _transcribe_for_xai(images, probe_model)
    else:
        surrogate_baseline = baseline_text

    cell_texts: List[str] = []
    cell_sims: List[float] = []

    t0 = time.time()

    for idx in range(total_cells):
        r = idx // grid_cols
        c = idx % grid_cols
        occluded = _occlude_region(img0, r, c, grid_rows, grid_cols)

        # Build the image list: occluded first image + remaining pages
        query_images = [occluded] + images[1:]
        cell_text = _transcribe_for_xai(query_images, probe_model)

        sim = _normalised_levenshtein(surrogate_baseline, cell_text)
        cell_texts.append(cell_text)
        cell_sims.append(sim)

        if progress_callback:
            progress_callback(idx + 1, total_cells)

    elapsed = time.time() - t0

    # Build heatmap: importance = 1 - similarity (more change = more important)
    importance = np.array([1.0 - s for s in cell_sims]).reshape(grid_rows, grid_cols)

    # Normalise to [0, 1]
    imin, imax = float(importance.min()), float(importance.max())
    if imax - imin > 1e-8:
        importance = (importance - imin) / (imax - imin)
    else:
        importance = np.zeros_like(importance)

    # Upscale to image dimensions for overlay
    w, h = img0.size
    from PIL import Image as _PILImage
    heatmap_pil = _PILImage.fromarray((importance * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), _PILImage.BILINEAR)
    heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    return OcclusionResult(
        heatmap=heatmap,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        baseline_text=surrogate_baseline,
        cell_texts=cell_texts,
        cell_similarities=cell_sims,
        surrogate_model=probe_model,
        elapsed_seconds=round(elapsed, 2),
    )
