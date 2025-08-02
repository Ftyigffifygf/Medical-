#!/bin/bash

# MedExpert Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

echo "🏥 MedExpert Deployment Script v2.0.0"
echo "======================================"

# Configuration
APP_NAME="medexpert"
DOCKER_IMAGE="medexpert:latest"
CONTAINER_NAME="medexpert-app"
PORT=8501

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_warning "Created .env file from .env.example. Please update with your values."
        else
            log_error ".env.example file not found"
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p logs data monitoring
    
    log_success "Environment setup complete"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE . || {
        log_error "Failed to build Docker image"
        exit 1
    }
    
    log_success "Docker image built successfully"
}

# Deploy application
deploy_app() {
    log_info "Deploying application..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_info "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Start new container
    docker-compose up -d || {
        log_error "Failed to start application"
        exit 1
    }
    
    log_success "Application deployed successfully"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for application to start
    sleep 10
    
    # Check if container is running
    if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_error "Container is not running"
        exit 1
    fi
    
    # Check application health
    for i in {1..30}; do
        if curl -f http://localhost:$PORT/_stcore/health &> /dev/null; then
            log_success "Application is healthy"
            return 0
        fi
        log_info "Waiting for application to be ready... ($i/30)"
        sleep 2
    done
    
    log_error "Application health check failed"
    exit 1
}

# Show deployment info
show_info() {
    log_success "🎉 MedExpert deployed successfully!"
    echo ""
    echo "📊 Deployment Information:"
    echo "========================="
    echo "🌐 Application URL: http://localhost:$PORT"
    echo "🐳 Container Name: $CONTAINER_NAME"
    echo "📋 Docker Image: $DOCKER_IMAGE"
    echo "📁 Logs Directory: ./logs"
    echo "💾 Data Directory: ./data"
    echo ""
    echo "🔧 Management Commands:"
    echo "======================"
    echo "📊 View logs: docker-compose logs -f"
    echo "🔄 Restart: docker-compose restart"
    echo "⏹️  Stop: docker-compose down"
    echo "🗑️  Remove: docker-compose down -v"
    echo ""
    echo "⚠️  Important Notes:"
    echo "==================="
    echo "• Update .env file with your configuration"
    echo "• Ensure proper medical compliance settings"
    echo "• Monitor logs for any issues"
    echo "• Set up SSL/TLS for production use"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up old images..."
    docker image prune -f
    log_success "Cleanup complete"
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    check_prerequisites
    setup_environment
    build_image
    deploy_app
    health_check
    cleanup
    show_info
    
    log_success "🏥 MedExpert deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping MedExpert..."
        docker-compose down
        log_success "MedExpert stopped"
        ;;
    "restart")
        log_info "Restarting MedExpert..."
        docker-compose restart
        log_success "MedExpert restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        log_info "Cleaning up MedExpert..."
        docker-compose down -v
        docker rmi $DOCKER_IMAGE 2>/dev/null || true
        log_success "Cleanup complete"
        ;;
    "help")
        echo "MedExpert Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  stop     - Stop the application"
        echo "  restart  - Restart the application"
        echo "  logs     - View application logs"
        echo "  status   - Show application status"
        echo "  clean    - Remove application and data"
        echo "  help     - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac