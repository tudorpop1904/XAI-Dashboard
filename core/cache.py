"""
cache.py — Response caching for LLM/VLM inference calls.

Caches Ollama responses keyed by a SHA-256 hash of the request
(model + messages). Dramatically speeds up repeated queries
(same image re-uploaded, same prompt re-run).

Cache is file-based under data_cache/llm_responses/ so it
persists across Streamlit reruns and Docker restarts (if volume-mounted).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import pickle
from typing import Any, Dict, Optional

CACHE_DIR = pathlib.Path("data_cache/llm_responses")
MODEL_DIR = pathlib.Path("data_cache/models")

# Create a hash of the request (model + messages + extra)
def _make_key(model: str, messages: list, **extra) -> str:
    """Deterministic SHA-256 hash of a request."""
    payload = json.dumps(
        {"model": model, "messages": messages, **extra},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()

# Get cached response
def get_cached(model: str, messages: list, **extra) -> Optional[Dict[str, Any]]:
    """Return cached response dict, or None if not cached."""
    key = _make_key(model, messages, **extra)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None

# Store a response in the cache
def put_cached(model: str, messages: list, response: Dict[str, Any], **extra) -> None:
    """Store a response in the cache."""
    key = _make_key(model, messages, **extra)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps(response, default=str), encoding="utf-8")
    except OSError:
        pass  # Non-critical — silently skip if write fails

# Wrapper around client.chat() with transparent file-based caching
def cached_chat(client, model: str, messages: list, **kwargs) -> Dict[str, Any]:
    """
    Wrapper around client.chat() with transparent file-based caching.

    Use this for non-streaming calls (transcription, evaluation).
    Streaming calls (counselor) should NOT be cached since st.write_stream
    consumes the generator.
    """
    cached = get_cached(model, messages)
    if cached is not None:
        return cached

    response = client.chat(model=model, messages=messages, **kwargs)
    put_cached(model, messages, response)
    return response

# Delete all cached responses
def clear_cache() -> int:
    """Delete all cached responses. Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)

# Cache Trained Model
def save_trained_model(model, name: str) -> None:
    """Serialize and save the trained model to a pickle file."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

# Load Cached Model
def load_trained_model(name: str):
    """Load a trained model from a pickle file."""
    path = MODEL_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)