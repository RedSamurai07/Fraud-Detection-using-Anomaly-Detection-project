#!/bin/bash
# deploy_ec2.sh - Full automated deployment for Fraud Detection API

set -e

# Configuration
TARGET_DIR="$HOME/fraud-deploy"
COMPOSE_FILE="$TARGET_DIR/docker/docker-compose.yml"

echo "[*] Starting Fresh Deployment..."

# 1. System Updates & Docker Installation
if ! [ -x "$(command -v docker)" ]; then
    echo "[!] Docker not found. Installing..."
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    echo "[+] Docker installed."
else
    echo "[+] Docker already present."
fi

# 2. Setup directory
mkdir -p $TARGET_DIR
cd $TARGET_DIR

# 3. Pulled files check
echo "[*] Files in $TARGET_DIR:"
ls -F $TARGET_DIR

# 4. Launch with Docker Compose
echo "[*] Restarting services..."
cd $TARGET_DIR
sudo docker-compose -f docker/docker-compose.yml down || true
sudo docker-compose -f docker/docker-compose.yml up -d --build

# 5. Verification
echo "[*] Verifying services..."
sleep 15
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    echo "[✅] API is LIVE and HEALTHY!"
else
    echo "[⚠️] API started but health check failed. Check logs with 'docker logs fraud-api'"
fi

echo "[🏁] Deployment Complete!"
