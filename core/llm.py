"""
llm.py — Ollama LLM integration for forensic XAI narrative reports.
"""

import os
import ollama


OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_client = ollama.Client(host=OLLAMA_HOST)


def check_ollama_available() -> tuple[bool, str]:
    try:
        models = _client.list()
        names = [m.model for m in models.models]
        prefix = OLLAMA_MODEL.split(":")[0]
        matches = [n for n in names if n.startswith(prefix)]
        if matches:
            return True, f"LLM ready: `{matches[0]}`"
        return False, f"Model `{OLLAMA_MODEL}` not found. Run: ollama pull {OLLAMA_MODEL}"
    except Exception as e:
        return False, f"Cannot reach Ollama at {OLLAMA_HOST}: {e}"

def build_forensic_prompt(
    detection_label: str,
    confidence: float,
    probabilities: dict[str, float],
    xai_results: list[dict],
    deployment_env: str = "local",
) -> str:
    """Build prompt comparing XAI methods for AI-generated image detection."""
    parts = [
        "You are a digital forensics analyst explaining AI-generated image detection "
        "to a researcher comparing Explainable AI methods.\n",
        f"Detection: **{detection_label}** (confidence {confidence:.1%})\n",
        f"Class probabilities: {probabilities}\n",
        f"Deployment environment: {deployment_env}\n",
        "\nThe following visual XAI methods were applied to the same image:\n",
    ]

    for row in xai_results:
        parts.append(
            f"- **{row['method']}** ({row['category']}): "
            f"runtime {row['elapsed_s']}s, peak RAM {row['peak_memory_mb']} MB, "
            f"forward passes {row['forward_passes']}, "
            f"stability {row.get('stability', 'N/A')}.\n"

        )
    parts.append(
        "\nWrite a concise report (3–5 paragraphs) covering:\n"
        "1. What the detection result means for this image.\n"
        "2. How the black-box methods (Occlusion, PMI, Sobol) compare in terms of "
        "which regions they highlight and their computational cost.\n"
        "3. How white-box methods (Grad-CAM, Saliency) differ from black-box approaches.\n"
        "4. Practical recommendation: which method balances fidelity, stability, and cost "
        f"for {deployment_env} deployment.\n"
        "Use clear academic Romanian-friendly English (bilingual terms OK). "
        "Do not invent numeric heatmap values — refer only to the metrics provided.\n"
    )
    return "".join(parts)

def stream_explanation(prompt: str):
    """Stream tokens from Ollama for st.write_stream."""

    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
