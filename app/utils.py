"""
Utility functions for the Food Recognition API.
"""

import os
import aiofiles
import hashlib
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image
import io
import logging

from .config import ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE, TEMP_UPLOAD_DIR

logger = logging.getLogger(__name__)


def is_valid_image_extension(filename: str) -> bool:
    """Check if the file has a valid image extension."""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def generate_temp_filename(original_filename: str, content: bytes) -> str:
    """Generate a unique temporary filename based on content hash."""
    file_hash = hashlib.md5(content).hexdigest()
    extension = Path(original_filename).suffix.lower()
    return f"{file_hash}{extension}"


async def save_upload_file(content: bytes, filename: str) -> str:
    """
    Save uploaded file to temporary directory.

    Args:
        content: File content as bytes
        filename: Original filename

    Returns:
        Path to saved file

    Raises:
        ValueError: If file is invalid
    """
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )

    if not is_valid_image_extension(filename):
        raise ValueError(
            f"Invalid file type. Allowed extensions: {ALLOWED_IMAGE_EXTENSIONS}"
        )

    # Generate unique filename
    temp_filename = generate_temp_filename(filename, content)
    temp_path = TEMP_UPLOAD_DIR / temp_filename

    # Save file
    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(content)

    logger.info(f"Saved uploaded file to {temp_path}")
    return str(temp_path)


async def cleanup_temp_file(filepath: str) -> None:
    """Remove temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temporary file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")


def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load and validate image from file path.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Validate image size (basic sanity check)
        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError("Image too small (minimum 32x32 pixels)")

        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError("Image too large (maximum 4096x4096 pixels)")
        
        print('image is all ready')

        return image

    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def load_image_from_bytes(content: bytes) -> Image.Image:
    """
    Load and validate image from bytes.

    Args:
        content: Image content as bytes

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        image = Image.open(io.BytesIO(content))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Validate image size
        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError("Image too small (minimum 32x32 pixels)")

        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError("Image too large (maximum 4096x4096 pixels)")

        return image

    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def validate_class_name(class_name: str) -> str:
    """
    Validate and sanitize class name.

    Args:
        class_name: Raw class name

    Returns:
        Sanitized class name

    Raises:
        ValueError: If class name is invalid
    """
    if not class_name or not class_name.strip():
        raise ValueError("Class name cannot be empty")

    class_name = class_name.strip()

    if len(class_name) > 100:
        raise ValueError("Class name too long (maximum 100 characters)")

    if len(class_name) < 1:
        raise ValueError("Class name too short (minimum 1 character)")

    # Remove any potentially problematic characters for filesystem
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        if char in class_name:
            raise ValueError(f"Class name contains invalid character: {char}")

    return class_name


class ErrorMessages:
    """Centralized error messages."""

    MODEL_NOT_INITIALIZED = "Model not initialized. Please check server logs."
    INVALID_IMAGE = "Invalid image file. Please upload a valid image."
    FILE_TOO_LARGE = (
        f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB."
    )
    INVALID_FILE_TYPE = (
        f"Invalid file type. Allowed extensions: {ALLOWED_IMAGE_EXTENSIONS}"
    )
    EMPTY_CLASS_NAME = "Class name cannot be empty."
    PROCESSING_ERROR = "An error occurred while processing your request."
    MODEL_SAVE_ERROR = "Failed to save the model."
    MODEL_LOAD_ERROR = "Failed to load the model."
