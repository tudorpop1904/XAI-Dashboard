#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# setup-cloud.sh — One-time provisioning for a Microsoft Azure VM.
#
# Run this ONCE after creating your VM:
#   ssh -i <key> ubuntu@<vm-ip> 'bash -s' < deploy/setup-cloud.sh
# ──────────────────────────────────────────────────────────────────────

echo "=== XAI Dashboard — Microsoft Azure VM Setup ==="

# 1. Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

# 2. Install Docker
echo "[2/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "  ✓ Docker installed. You may need to re-login for group changes."
else
    echo "  ✓ Docker already installed."
fi

# 3. Install Docker Compose plugin
echo "[3/5] Ensuring Docker Compose plugin..."
if ! docker compose version &> /dev/null; then
    sudo apt-get install -y -qq docker-compose-plugin
    echo "  ✓ Docker Compose plugin installed."
else
    echo "  ✓ Docker Compose already available."
fi

# 4. Open firewall ports (Streamlit 8501, Ollama 11434)
echo "[4/5] Configuring iptables for ports 8501 and 11434..."
sudo iptables -I INPUT -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo iptables -I INPUT -m state --state NEW -p tcp --dport 11434 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || true
echo "  ✓ Ports 8501 and 11434 opened."

# 5. Clone the repository (if not present)
echo "[5/5] Cloning repository..."
REPO_DIR="$HOME/xai-app"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/tudorpop1904/XAI-Dashboard.git "$REPO_DIR"
    echo "  ✓ Repository cloned to $REPO_DIR"
else
    echo "  ✓ Repository already exists at $REPO_DIR"
fi
git checkout v0.2/logistics-and-metrics
git pull


# Create .env if not present
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "# Kaggle credentials (optional, only needed for dataset downloads)" > "$REPO_DIR/.env"
    echo "KAGGLE_USERNAME=" >> "$REPO_DIR/.env"
    echo "KAGGLE_API_TOKEN=" >> "$REPO_DIR/.env"
    echo "  ✓ Created .env template — fill in Kaggle creds if needed."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Re-login (or run: newgrp docker)"
echo "  2. cd $REPO_DIR"
echo "  3. docker compose -f docker/docker-compose.cloud.yml up -d --build"
echo "  4. Access the app at http://<your-vm-ip>:8501"
echo ""
echo "NOTE: First build will take ~10-15 min (downloading PyTorch,"
echo "pulling Ollama models). Subsequent builds use Docker cache."
