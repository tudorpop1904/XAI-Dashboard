# Docker Containerization & Memory Optimization Guide

This guide describes how to manage memory and resource allocation for the XAI Playground application, especially when running Ollama and Streamlit together inside Docker.

## 📦 Memory Usage Breakdown

| Container | Service | Typical RAM |
|-----------|---------|-------------|
| `xai-app` | Streamlit + sklearn + SHAP + LIME | ~500 MB – 1.5 GB |
| `xai-ollama` | Ollama LLM server | Depends on model ↓ |

### Ollama Model Tiers

| Model | Size | RAM Required | Quality |
|-------|------|-------------|---------|
| `tinyllama` | ~637 MB | ~1.5 GB | Basic — fast, but lower-quality explanations |
| `phi3:mini` | ~2.3 GB | ~3.5 GB | **Default** — good balance of quality and speed |
| `phi3` | ~3.8 GB | ~5 GB | Better quality, higher RAM |
| `llama3.1:8b` | ~4.7 GB | ~6.5 GB | High quality, needs 8+ GB free RAM |

## 🐳 Docker Compose Memory Limits

The `docker-compose.yml` now includes explicit resource limits:

```yaml
# xai-app: hard limit 2 GB
# ollama:  hard limit 6 GB, reservation 3 GB
```

Adjust these based on your system's total RAM.

## 🖥️ WSL2 Configuration (Windows)

Docker Desktop on Windows runs inside a WSL2 virtual machine. By default, WSL2 can consume up to 50% of your total RAM. To tune this:

1. **Create/edit** `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=8GB      # For phi3:mini. Use 12GB+ for llama3.1:8b.
   swap=2GB
   processors=4
   ```

2. **Apply** by restarting WSL:
   ```powershell
   wsl --shutdown
   ```
   Then restart Docker Desktop.

## 🛠️ Switching Models

To change the LLM model:

### Option A: Environment Variable (recommended)
In `docker-compose.yml`, change:
```yaml
- OLLAMA_MODEL=phi3:mini
```
to:
```yaml
- OLLAMA_MODEL=tinyllama
```

### Option B: Fallback in Code
In `core/llm.py`, the default is:
```python
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
```

## 🎮 GPU Acceleration

If you have an **NVIDIA GPU** + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html):
- The `deploy.resources.reservations.devices` block in `docker-compose.yml` enables GPU pass-through automatically.
- Ollama will offload model layers to VRAM, which is significantly faster and frees system RAM.

If you **don't** have an NVIDIA GPU:
- Remove or comment out the `devices` block under `reservations` in the `ollama` service. Ollama will fall back to CPU-only mode (slower but stable).

## 📊 Monitoring

Check real-time container resource usage:
```bash
docker stats
```

On Windows, watch the `vmmem` process in Task Manager — that's the WSL2 VM.

If RAM stays high after stopping containers:
```bash
wsl --shutdown
```

## 🚀 Cloud Deployment

To deploy without local LLM overhead:
- Replace the Ollama client in `core/llm.py` with an API-based LLM (e.g., **Google Gemini**, **OpenAI**).
- Remove the `ollama` service from `docker-compose.yml`.
- This reduces the Docker footprint by ~3–5 GB.
