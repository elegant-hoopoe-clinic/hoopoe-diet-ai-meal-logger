"""
Configuration settings for the Food Recognition API.
"""

import os
from pathlib import Path

# Model configuration
MODEL_NAME = "ViT-L/14"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
MAX_AUGMENTATIONS = 8
DEFAULT_AUGMENTATIONS = 3

# File paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/food_recognizer_model.json")
TEMP_UPLOAD_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "temp_uploads"))
MODEL_STORAGE_DIR = Path(os.getenv("MODEL_STORAGE_DIR", "model_storage"))

# API configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
API_TITLE = "Food Recognition API"
API_DESCRIPTION = """
A professional food recognition service using OpenAI's CLIP model with one-shot learning capabilities.

## Features:
- Real-time food recognition from images
- One-shot learning for new food classes
- Model persistence and management
- Confidence scoring and detailed predictions
"""
API_VERSION = "1.0.0"

# Performance settings
WORKER_TIMEOUT = 300  # 5 minutes
MAX_CONCURRENT_REQUESTS = 10

# Create directories if they don't exist
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_STORAGE_DIR.mkdir(exist_ok=True)
