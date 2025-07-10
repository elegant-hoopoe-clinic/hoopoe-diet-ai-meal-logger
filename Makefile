# Food Recognition API - Makefile for Windows PowerShell
# Run these commands in PowerShell

.PHONY: help install dev test docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  clean       - Clean temporary files"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev:
	python run_dev.py

test:
	pytest test_api.py -v

docker-build:
	docker build -t food-recognition-api .

docker-run:
	docker run -p 8000:8000 -v ${PWD}/models:/app/models -v ${PWD}/model_storage:/app/model_storage food-recognition-api

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

clean:
	Remove-Item -Recurse -Force __pycache__, .pytest_cache, temp_uploads/* -ErrorAction SilentlyContinue
