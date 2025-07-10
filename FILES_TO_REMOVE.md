# Files that can be removed for a simpler setup:

## Development/Testing Files (can be removed):
- `test_api.py` - API tests
- `client_example.py` - Client example code
- `run_dev.py` - Development server runner
- `gunicorn.conf.py` - Gunicorn configuration
- `Makefile` - Build automation
- `manage.ps1` - PowerShell management script
- `QUICK_START.md` - Quick start guide
- `.env.example` - Environment example
- `.gitignore` - Git ignore file

## Essential Files (keep these):
- `app/` - Main application directory
- `models/` - Model storage directory
- `model_storage/` - Model persistence directory
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker compose configuration
- `requirements.txt` - Python dependencies
- `README.md` - Documentation

## Core Application Files:
- `app/main.py` - FastAPI application
- `app/models.py` - Food recognition model
- `app/schemas.py` - API schemas
- `app/utils.py` - Utility functions
- `app/config.py` - Configuration settings
- `app/__init__.py` - Package initialization

You can safely delete the development/testing files if you want a minimal setup.
