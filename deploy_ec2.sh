#!/bin/bash
# AWS EC2 Free Tier Deployment Script for Fraud Detection API
# This script sets up the application on an EC2 t2.micro instance

set -e  # Exit on error

echo "========================================"
echo "Fraud Detection API - AWS EC2 Setup"
echo "========================================"

# Update system packages
echo ""
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo ""
echo "Step 2: Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Install Docker
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo \
      "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    echo "✓ Docker installed successfully"
else
    echo "✓ Docker already installed"
fi

# Install Docker Compose
echo ""
echo "Step 3: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✓ Docker Compose installed successfully"
else
    echo "✓ Docker Compose already installed"
fi

# Install Python and pip (for local testing)
echo ""
echo "Step 4: Installing Python..."
sudo apt-get install -y python3 python3-pip python3-venv
echo "✓ Python installed"

# Install Git
echo ""
echo "Step 5: Installing Git..."
sudo apt-get install -y git
echo "✓ Git installed"

# Create application directory
echo ""
echo "Step 6: Setting up application directory..."
APP_DIR="/home/ubuntu/fraud-detection-api"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository (if using Git) or copy files
echo ""
echo "Step 7: Deploying application files..."
echo "Note: Copy your application files to $APP_DIR"
echo "      Including: api/, docker/, models/, requirements.txt"

# Build Docker image
echo ""
echo "Step 8: Building Docker image..."
# This assumes docker files are in place
if [ -f "docker/Dockerfile" ]; then
    sudo docker build -f docker/Dockerfile -t fraud-detection-api .
    echo "✓ Docker image built"
else
    echo "⚠ Dockerfile not found. Please copy application files first."
fi

# Configure firewall
echo ""
echo "Step 9: Configuring firewall..."
sudo ufw allow 22    # SSH
sudo ufw allow 5000  # API
sudo ufw allow 5001  # MLflow
sudo ufw --force enable
echo "✓ Firewall configured"

# Create systemd service for auto-start
echo ""
echo "Step 10: Creating systemd service..."
sudo tee /etc/systemd/system/fraud-api.service > /dev/null <<EOF
[Unit]
Description=Fraud Detection API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose -f docker/docker-compose.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker/docker-compose.yml down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fraud-api.service
echo "✓ Systemd service created"

# Start the service
echo ""
echo "Step 11: Starting application..."
if [ -f "docker/docker-compose.yml" ]; then
    cd $APP_DIR
    sudo docker-compose -f docker/docker-compose.yml up -d
    echo "✓ Application started"
else
    echo "⚠ docker-compose.yml not found. Please copy application files first."
fi

# Display status
echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "API URL: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
echo "MLflow URL: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5001"
echo ""
echo "To check status:"
echo "  sudo docker ps"
echo "  sudo systemctl status fraud-api"
echo ""
echo "To view logs:"
echo "  sudo docker-compose -f docker/docker-compose.yml logs -f"
echo ""
echo "To stop:"
echo "  sudo systemctl stop fraud-api"
echo ""
echo "========================================"