# Raspberry Pi 4 — Ollama Compute Offloading

## Overview

The application already supports offloading VLM inference to a remote
Ollama instance via the `REMOTE_VLM_HOST` environment variable. This
document explains how to set up a Raspberry Pi 4 as an inference node.

## Architecture

```
┌──────────────┐        HTTP         ┌──────────────────┐
│  xai-app     │ ──────────────────▶ │  Pi 4 (Ollama)   │
│  (Streamlit) │  /api/chat          │  minicpm-v (XAI) │
│  Docker host │ ◀────────────────── │  Port 11434      │
└──────────────┘     JSON stream     └──────────────────┘
```

The main Docker host runs Streamlit + the text LLM. The Pi handles only
the **surrogate VLM** calls (occlusion sensitivity XAI), which are the
most numerous but don't require high accuracy.

## Pi 4 Setup

### 1. Install Ollama on the Pi

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the surrogate model

```bash
ollama pull minicpm-v
```

The minicpm-v model (3B params, ~2GB) runs on the Pi 4's 8GB RAM.
Inference is slow (~30–60s per call) but acceptable for batch XAI probing.

### 3. Start Ollama on the Pi

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 4. Configure the dashboard

In `.env` or `docker-compose.yml`:

```yaml
environment:
  - REMOTE_VLM_HOST=http://<pi-ip-address>:11434
```

The `_get_vlm_client()` in `vlm_engine.py` will automatically route
VLM calls to the Pi instead of the local Ollama instance.

## What runs where

| Task              | Model          | Runs on        |
|-------------------|----------------|----------------|
| LLM counseling    | llama3.1:8b    | Docker host    |
| VLM transcription | qwen2.5vl:7b   | Docker host    |
| XAI probing       | minicpm-v      | **Pi 4**       |
| CNN (Viz/Writing) | HandwritingCNN | Docker host    |
| DiCE              | sklearn        | Docker host    |

## Performance expectations

- **Pi 4 (8GB)**: ~30–60s per minicpm-v call. A 3×3 occlusion grid = ~5–9 min.
- **Pi 5 (8GB)**: ~15–30s per call. Same grid = ~2–5 min.
- **Jetson Nano**: With GPU acceleration, ~5–10s per call.

## Future: Pi cluster

For heavier models, consider a Pi 5 cluster with load balancing:

```
Pi1 → ollama serve (port 11434)
Pi2 → ollama serve (port 11434)
HAProxy → round-robin across Pi1, Pi2
REMOTE_VLM_HOST=http://haproxy:11434
```
