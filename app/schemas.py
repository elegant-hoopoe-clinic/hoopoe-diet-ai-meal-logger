"""
Pydantic schemas for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    total_classes: int = Field(..., description="Number of learned food classes")


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    success: bool = Field(..., description="Whether the prediction was successful")
    is_known_food: bool = Field(False, description="Whether the food is recognized")
    best_prediction: Optional[Dict[str, Any]] = Field(
        None, description="Best prediction result"
    )
    predictions: List[Dict[str, Any]] = Field(
        default_factory=list, description="All predictions"
    )
    threshold: float = Field(..., description="Similarity threshold used")
    message: Optional[str] = Field(None, description="Error message if any")


class LearnRequest(BaseModel):
    """Learn new class request schema."""

    class_name: str = Field(
        ..., min_length=1, max_length=100, description="Name of the food class"
    )
    n_prototypes: int = Field(
        3, ge=1, le=10, description="Number of prototypes to generate"
    )


class LearnResponse(BaseModel):
    """Learn new class response schema."""

    success: bool = Field(..., description="Whether learning was successful")
    message: str = Field(..., description="Success or error message")
    class_name: str = Field(..., description="Name of the learned class")
    total_classes: Optional[int] = Field(
        None, description="Total number of classes after learning"
    )


class UpdateRequest(BaseModel):
    """Update class request schema."""

    class_name: str = Field(
        ..., min_length=1, max_length=100, description="Name of the food class"
    )


class UpdateResponse(BaseModel):
    """Update class response schema."""

    success: bool = Field(..., description="Whether update was successful")
    message: str = Field(..., description="Success or error message")
    class_name: str = Field(..., description="Name of the updated class")
    total_examples: Optional[int] = Field(
        None, description="Total examples for this class"
    )


class ClassesResponse(BaseModel):
    """List classes response schema."""

    total_classes: int = Field(..., description="Total number of learned classes")
    classes: List[str] = Field(..., description="List of class names")


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    model_name: str = Field(..., description="Name of the underlying model")
    device: str = Field(..., description="Device being used (CPU/CUDA)")
    similarity_threshold: float = Field(..., description="Current similarity threshold")
    total_classes: int = Field(..., description="Number of learned classes")
    classes: List[str] = Field(..., description="List of class names")
    has_index: bool = Field(..., description="Whether FAISS index is built")
    model_initialized: bool = Field(..., description="Whether the model is initialized")


class SaveModelRequest(BaseModel):
    """Save model request schema."""

    filename: Optional[str] = Field(
        None, description="Custom filename for the saved model"
    )


class SaveModelResponse(BaseModel):
    """Save model response schema."""

    success: bool = Field(..., description="Whether saving was successful")
    message: str = Field(..., description="Success or error message")
    filepath: Optional[str] = Field(None, description="Path where model was saved")
    total_classes: Optional[int] = Field(None, description="Number of classes saved")


class LoadModelRequest(BaseModel):
    """Load model request schema."""

    filename: str = Field(..., description="Filename of the model to load")


class LoadModelResponse(BaseModel):
    """Load model response schema."""

    success: bool = Field(..., description="Whether loading was successful")
    message: str = Field(..., description="Success or error message")
    filepath: Optional[str] = Field(
        None, description="Path from which model was loaded"
    )
    total_classes: Optional[int] = Field(None, description="Number of classes loaded")
    classes: Optional[List[str]] = Field(None, description="List of loaded class names")


class ClassExistsResponse(BaseModel):
    """Class exists check response schema."""

    class_name: str = Field(..., description="The checked class name")
    exists: bool = Field(..., description="Whether the class exists")
