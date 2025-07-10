#!/usr/bin/env python3
"""
Development server runner for the Food Recognition API.
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Set environment variables for development
    os.environ.setdefault("MODEL_PATH", "models/food_recognizer_model.json")
    os.environ.setdefault("TEMP_UPLOAD_DIR", "temp_uploads")
    os.environ.setdefault("MODEL_STORAGE_DIR", "model_storage")

    # Create directories if they don't exist
    Path("temp_uploads").mkdir(exist_ok=True)
    Path("model_storage").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # Run the development server
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
