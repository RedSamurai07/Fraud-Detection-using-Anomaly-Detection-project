#!/bin/bash
# Quick Start Script for Fraud Detection API
# Automates the complete deployment process

set -e

echo "========================================"
echo "Fraud Detection API - Quick Start"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
echo ""
echo "Step 1: Checking prerequisites..."

if command -v python3 &> /dev/null; then
    print_success "Python $(python3 --version) installed"
else
    print_error "Python not found. Please install Python 3.9+"
    exit 1
fi

if command -v docker &> /dev/null; then
    print_success "Docker $(docker --version) installed"
else
    print_error "Docker not found. Please install Docker"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose $(docker-compose --version) installed"
else
    print_error "Docker Compose not found. Please install Docker Compose"
    exit 1
fi

# Ask user for deployment type
echo ""
echo "Select deployment type:"
echo "1) Local development (Docker Compose)"
echo "2) Train model only"
echo "3) Run tests only"
echo "4) Full deployment (Train + Docker + Test)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        print_info "Starting local development environment..."
        
        # Check if models exist
        if [ ! -d "models" ] || [ ! -f "models/fraud_model.pkl" ]; then
            print_error "Models not found. Please train model first (option 2)"
            exit 1
        fi
        
        # Start Docker Compose
        cd docker
        docker-compose up -d
        
        print_success "Services started!"
        echo ""
        echo "API: http://localhost:5000"
        echo "MLflow: http://localhost:5001"
        echo ""
        echo "Test with: curl http://localhost:5000/health"
        ;;
        
    2)
        echo ""
        print_info "Training fraud detection model..."
        
        # Check if dataset exists
        if [ ! -f "creditcard.csv" ]; then
            print_error "creditcard.csv not found in current directory"
            echo "Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            exit 1
        fi
        
        # Install dependencies
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        
        # Train model
        print_info "Training model (this may take a few minutes)..."
        python mlflow/train_model.py creditcard.csv
        
        print_success "Model training complete!"
        print_info "Model artifacts saved to ./models/"
        print_info "MLflow logs saved to ./mlruns/"
        ;;
        
    3)
        echo ""
        print_info "Running test suite..."
        
        # Install dependencies
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        
        # Run tests
        pytest tests/ -v --cov=api --cov-report=html --cov-report=json
        
        # Generate badges
        print_info "Generating test badges..."
        python tests/generate_badges.py
        
        print_success "Tests complete!"
        print_info "Coverage report: htmlcov/index.html"
        print_info "Badges: badges.md"
        ;;
        
    4)
        echo ""
        print_info "Full deployment pipeline starting..."
        
        # Check dataset
        if [ ! -f "creditcard.csv" ]; then
            print_error "creditcard.csv not found"
            exit 1
        fi
        
        # Install dependencies
        echo ""
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        print_success "Dependencies installed"
        
        # Train model
        echo ""
        print_info "Training model..."
        python mlflow/train_model.py creditcard.csv
        print_success "Model trained"
        
        # Run tests
        echo ""
        print_info "Running tests..."
        pytest tests/ -v --cov=api --cov-report=json
        python tests/generate_badges.py
        print_success "Tests passed"
        
        # Build Docker
        echo ""
        print_info "Building Docker image..."
        docker build -f docker/Dockerfile -t fraud-detection-api .
        print_success "Docker image built"
        
        # Start services
        echo ""
        print_info "Starting services..."
        cd docker
        docker-compose up -d
        print_success "Services started"
        
        # Wait for services to be ready
        echo ""
        print_info "Waiting for services to be ready..."
        sleep 10
        
        # Test deployment
        echo ""
        print_info "Testing deployment..."
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)
        if [ "$response" = "200" ]; then
            print_success "Deployment successful!"
        else
            print_error "Deployment test failed (HTTP $response)"
        fi
        
        # Summary
        echo ""
        echo "========================================"
        print_success "DEPLOYMENT COMPLETE!"
        echo "========================================"
        echo ""
        echo "Services:"
        echo "  API:    http://localhost:5000"
        echo "  MLflow: http://localhost:5001"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:5000/health"
        echo ""
        echo "View logs:"
        echo "  docker-compose -f docker/docker-compose.yml logs -f"
        echo ""
        echo "Stop services:"
        echo "  docker-compose -f docker/docker-compose.yml down"
        echo "========================================"
        ;;
        
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Done!"