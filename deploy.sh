#!/bin/bash
# AWS EC2 Deployment Script for Fraud Detection API

# 1. Update and install Docker
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 2. Build the Docker image
echo "Building Docker image..."
sudo docker build -t fraud-detection-api:latest .

# 3. Run the Docker container
echo "Running Docker container..."
# Stop existing container if it exists
sudo docker stop fraud-api || true
sudo docker rm fraud-api || true
sudo docker run -d -p 80:8000 --name fraud-api fraud-detection-api:latest

echo "Deployment completed. API is running on port 80."
