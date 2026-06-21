#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# deploy.sh — Deploy latest code to the Microsoft Azure VM.
#
# Called by Jenkins (or manually):
#   ./deploy/deploy.sh <vm-ip> <ssh-key-path>
#
# What it does:
#   1. SSH into the VM
#   2. git pull latest changes
#   3. Rebuild and restart containers
#   4. Run a smoke test
# ──────────────────────────────────────────────────────────────────────

VM_IP="${1:?Usage: deploy.sh <vm-ip> <ssh-key-path>}"
SSH_KEY="${2:?Usage: deploy.sh <vm-ip> <ssh-key-path>}"
SSH_USER="${3:-azureuser}"
REPO_DIR="~/xai-app"
COMPOSE_FILE="docker/docker-compose.cloud.yml"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${SSH_KEY}"

echo "=== Deploying to Microsoft Azure VM: ${VM_IP} ==="

# 1. Pull latest code
echo "[1/3] Pulling latest code..."
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" \
    "cd ${REPO_DIR} && git fetch --all && git reset --hard origin/main"

# 2. Rebuild and restart
echo "[2/3] Building and restarting containers..."
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" \
    "cd ${REPO_DIR} && docker compose -f ${COMPOSE_FILE} up -d --build --remove-orphans"

# 3. Smoke test (wait for Streamlit to be ready)
echo "[3/3] Running smoke test..."
sleep 15
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" \
    "curl -sf http://localhost:8501/ > /dev/null && echo '✓ Streamlit is up' || echo '✗ Streamlit check failed'"
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" \
    "curl -sf http://localhost:11434/api/tags > /dev/null && echo '✓ Ollama is up' || echo '✗ Ollama check failed'"

echo ""
echo "=== Deployment complete ==="
echo "App available at: http://${VM_IP}:8501"
