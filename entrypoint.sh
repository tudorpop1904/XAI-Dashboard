#!/bin/bash
set -e

# ─────────────────────────────────────────────
# Ollama container entrypoint
#   1. Start ollama serve in the background
#   2. Wait until the server is responsive
#   3. Pull the required models
#   4. Keep the server running in the foreground
# ─────────────────────────────────────────────

# Start the Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for the server to become responsive
# We use `ollama list` instead of curl because curl isn't
# installed in the ollama/ollama image by default.
echo "Waiting for Ollama server to start..."
MAX_WAIT=60
WAITED=0
until ollama list > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Ollama server did not start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
echo "Ollama server is ready! (took ${WAITED}s)"

# Pull models (skips download if already present)
echo ""
echo "Pulling LLM model (llama3.1:8b-instruct-q4_K_M)..."
ollama pull llama3.1:8b-instruct-q4_K_M
echo "✓ llama3.1:8b-instruct-q4_K_M ready"

echo ""
echo "Pulling VLM model (qwen2.5vl:7b-q4_K_M)..."
ollama pull qwen2.5vl:7b-q4_K_M
echo "✓ qwen2.5vl:7b-q4_K_M ready"

echo ""
echo "Pulling XAI surrogate model (minicpm-v)..."
ollama pull minicpm-v
echo "✓ minicpm-v ready"


# Ready banner
echo ""
echo "=========================================="
echo "  All models pulled. Ollama is ready!     "
echo "=========================================="
echo ""

# Keep the server running (wait on the background process)
wait $OLLAMA_PID
