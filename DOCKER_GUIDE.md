# Docker Containerization & Memory Optimization Guide

This guide describes how to manage memory and resource allocation for the Image Forensics & Deepfake Detection Dashboard when running Ollama and Streamlit together inside Docker.

## 📦 Memory Usage Breakdown

| Container | Service | Typical RAM |
|-----------|---------|-------------|
| `xai-app` | Streamlit + PyTorch + OpenCV + Scikit-Image | ~1.5 GB – 3 GB (during CNN model training) |
| `xai-ollama`| Ollama LLM Server | Depends on model ↓ |

### LLM Memory Requirements
For generating narrative explainability reports, we use the following LLM model:

- **Model**: `llama3.1:8b-instruct-q4_K_M` (4-bit quantized)
- **Model Size**: ~4.7 GB
- **Required System RAM**: 8 GB+ (12 GB+ recommended for stability when training and generating reports simultaneously)

---

## 🐳 Docker Compose Memory Limits

The `docker-compose.yml` includes resource limits to ensure system stability:
- **`xai-app`**: Limited to **8 GB** to allow dataset loading and parallel feature calculations.
- **`xai-ollama`**: Limited to **12 GB** (with a 12 GB reservation) for fast inference.

Adjust these values in `docker/docker-compose.yml` if your system has limited physical memory.

---

## 🖥️ WSL2 Configuration (Windows)

On Windows, Docker Desktop runs inside a WSL2 virtual machine, which by default can consume up to 50% of your total RAM. To optimize this:

1. **Create or edit** `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=16GB      # Set to at least 12GB-16GB if you train models and run the LLM locally
   swap=4GB
   processors=4
   ```
2. **Apply changes** by shutting down WSL from PowerShell:
   ```powershell
   wsl --shutdown
   ```
3. Restart Docker Desktop.

---

## 🎮 GPU Acceleration

If you have an **NVIDIA GPU** and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed:
- The GPU reservation block in `docker-compose.yml` is enabled by default for the `ollama` service.
- Ollama will automatically run inference on your GPU, which drastically speeds up report generation and leaves system RAM free.

If you **do not** have an NVIDIA GPU:
- Remove or comment out the `reservations.devices` section under the `ollama` service in `docker-compose.yml`. Ollama will fall back to CPU-only execution.

---

## 📊 Monitoring Resources

You can check real-time container CPU and RAM usage by running:
```bash
docker stats
```

On Windows, you can monitor the `vmmem` (or `vmmemWSL`) process in the Task Manager to see exactly how much RAM the WSL2 Linux VM is using.
