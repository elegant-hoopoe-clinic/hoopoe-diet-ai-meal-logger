# Food Recognition API Setup and Management Script
# PowerShell script for easy setup and management

param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "dev", "test", "docker-build", "docker-run", "docker-compose", "clean", "help")]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Food Recognition API Management Script" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\manage.ps1 <command>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  setup         - Set up virtual environment and install dependencies"
    Write-Host "  dev           - Run development server"
    Write-Host "  test          - Run tests"
    Write-Host "  docker-build  - Build Docker image"
    Write-Host "  docker-run    - Run Docker container"
    Write-Host "  docker-compose - Start with Docker Compose"
    Write-Host "  clean         - Clean temporary files and cache"
    Write-Host "  help          - Show this help message"
}

function Setup-Environment {
    Write-Host "Setting up Food Recognition API environment..." -ForegroundColor Green
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Create necessary directories
    Write-Host "Creating directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "temp_uploads", "model_storage", "models" | Out-Null
    
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host "Run '.\manage.ps1 dev' to start the development server" -ForegroundColor Cyan
}

function Start-DevServer {
    Write-Host "Starting development server..." -ForegroundColor Green
    
    # Activate virtual environment if it exists
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & ".\venv\Scripts\Activate.ps1"
    }
    
    # Create directories if they don't exist
    New-Item -ItemType Directory -Force -Path "temp_uploads", "model_storage", "models" | Out-Null
    
    # Start the server
    python run_dev.py
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    
    # Activate virtual environment if it exists
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & ".\venv\Scripts\Activate.ps1"
    }
    
    # Run tests
    pytest test_api.py -v
}

function Build-DockerImage {
    Write-Host "Building Docker image..." -ForegroundColor Green
    docker build -t food-recognition-api .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker image built successfully!" -ForegroundColor Green
    } else {
        Write-Host "Docker build failed!" -ForegroundColor Red
        exit 1
    }
}

function Run-DockerContainer {
    Write-Host "Running Docker container..." -ForegroundColor Green
    
    # Create directories for volume mounts
    New-Item -ItemType Directory -Force -Path "models", "model_storage" | Out-Null
    
    # Run container
    docker run -p 8000:8000 `
        -v "${PWD}\models:/app/models" `
        -v "${PWD}\model_storage:/app/model_storage" `
        --name food-recognition-api `
        --rm `
        food-recognition-api
}

function Start-DockerCompose {
    Write-Host "Starting with Docker Compose..." -ForegroundColor Green
    
    # Create directories for volume mounts
    New-Item -ItemType Directory -Force -Path "models", "model_storage" | Out-Null
    
    # Start services
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Services started successfully!" -ForegroundColor Green
        Write-Host "API available at: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "View logs with: docker-compose logs -f" -ForegroundColor Yellow
        Write-Host "Stop with: docker-compose down" -ForegroundColor Yellow
    } else {
        Write-Host "Failed to start services!" -ForegroundColor Red
        exit 1
    }
}

function Clean-Environment {
    Write-Host "Cleaning environment..." -ForegroundColor Green
    
    # Remove Python cache
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Name ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Clean temp uploads
    if (Test-Path "temp_uploads") {
        Get-ChildItem "temp_uploads\*" | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    
    # Remove Docker containers and images (optional)
    $response = Read-Host "Remove Docker containers and images? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        docker-compose down -v --remove-orphans 2>$null
        docker rmi food-recognition-api 2>$null
    }
    
    Write-Host "Cleanup complete!" -ForegroundColor Green
}

# Main execution
switch ($Command) {
    "setup" { Setup-Environment }
    "dev" { Start-DevServer }
    "test" { Run-Tests }
    "docker-build" { Build-DockerImage }
    "docker-run" { Run-DockerContainer }
    "docker-compose" { Start-DockerCompose }
    "clean" { Clean-Environment }
    "help" { Show-Help }
    default { Show-Help }
}
