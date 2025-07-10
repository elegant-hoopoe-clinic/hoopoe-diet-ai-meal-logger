"""
Main FastAPI application for Food Recognition API.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    BackgroundTasks,
    Depends,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import aiofiles
from pathlib import Path
import uvicorn

from .models import FoodRecognitionModel
from .schemas import (
    HealthResponse,
    PredictionResponse,
    LearnResponse,
    UpdateRequest,
    UpdateResponse,
    ClassesResponse,
    ModelInfoResponse,
    SaveModelRequest,
    SaveModelResponse,
    LoadModelRequest,
    LoadModelResponse,
    ClassExistsResponse,
)
from .utils import (
    save_upload_file,
    cleanup_temp_file,
    load_image_from_path,
    validate_class_name,
    ErrorMessages,
)
from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    MODEL_PATH,
    MODEL_STORAGE_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
model: FoodRecognitionModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    global model

    # Startup
    logger.info("Starting Food Recognition API...")

    try:
        # Initialize the model
        model = FoodRecognitionModel()

        # Try to load existing model if available
        try:
            if os.path.exists(MODEL_PATH):
                model.load_model(MODEL_PATH)
                logger.info("Loaded existing model successfully")
        except Exception as e:
            logger.warning(f"No existing model found or failed to load: {e}")
            logger.info("Starting with empty model")

        logger.info("Food Recognition API started successfully")

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Food Recognition API...")
    if model is not None:
        try:
            model.save_model(MODEL_PATH)
            logger.info("Model saved on shutdown")
        except Exception as e:
            logger.error(f"Failed to save model on shutdown: {e}")
    logger.info("Food Recognition API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model() -> FoodRecognitionModel:
    """Get the global model instance."""
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    return model


# Remove the old startup/shutdown handlers since we're using lifespan


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    model: FoodRecognitionModel = Depends(get_model),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the service and model information.
    """
    try:
        model_info = model.get_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=model_info["model_initialized"],
            total_classes=model_info["total_classes"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_food(
    image: UploadFile = File(..., description="Image file to analyze"),
    top_k: int = Query(
        3, ge=1, le=10, description="Number of top predictions to return"
    ),
    model: FoodRecognitionModel = Depends(get_model),
) -> PredictionResponse:
    """
    Predict the type of food in an uploaded image.

    - **image**: Image file (JPG, PNG, BMP, WebP supported)
    - **top_k**: Number of top predictions to return (default: 1)

    Returns prediction results with confidence scores.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_file_path = None

    try:
        # Read and validate file
        content = await image.read()
        temp_file_path = await save_upload_file(content, image.filename)

        # Load image
        pil_image = load_image_from_path(temp_file_path)

        # Make prediction
        result = model.predict(pil_image, top_k=top_k)

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )
    finally:
        if temp_file_path:
            await cleanup_temp_file(temp_file_path)


@app.post("/learn", response_model=LearnResponse, tags=["Learning"])
async def learn_food_class(
    image: UploadFile = File(..., description="Image file for the new food class"),
    class_name: str = Form(..., description="Name of the food class"),
    n_prototypes: int = Form(
        3, ge=1, le=10, description="Number of prototypes to generate"
    ),
    model: FoodRecognitionModel = Depends(get_model),
) -> LearnResponse:
    """
    Learn a new food class from an uploaded image.

    - **image**: Image file representing the food class
    - **class_name**: Name for the new food class
    - **n_prototypes**: Number of prototype embeddings to generate (default: 3)

    The system will generate multiple augmented versions of the image to create
    robust prototypes for better recognition.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_file_path = None

    try:
        # Validate class name
        class_name = validate_class_name(class_name)

        # Read and validate file
        content = await image.read()
        temp_file_path = await save_upload_file(content, image.filename)

        # Load image
        pil_image = load_image_from_path(temp_file_path)

        # Learn new class
        result = model.learn_new_class(pil_image, class_name, n_prototypes)

        return LearnResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Learning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )
    finally:
        if temp_file_path:
            await cleanup_temp_file(temp_file_path)


@app.post("/update", response_model=LearnResponse, tags=["Learning"])
async def update_food_class(
    image: UploadFile = File(..., description="Additional image for the food class"),
    class_name: str = Form(..., description="Name of the existing food class"),
    model: FoodRecognitionModel = Depends(get_model),
) -> LearnResponse:
    """
    Update an existing food class with a new example image.

    - **image**: Additional image for the food class
    - **class_name**: Name of the existing food class to update

    If the class doesn't exist, it will be created as a new class.
    This helps improve recognition accuracy by providing more examples.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_file_path = None

    try:
        # Validate class name
        class_name = validate_class_name(class_name)

        # Read and validate file
        content = await image.read()
        temp_file_path = await save_upload_file(content, image.filename)

        # Load image
        pil_image = load_image_from_path(temp_file_path)

        # Update class
        print(f"here before the update class {class_name} , {pil_image}")
        result = model.update_class(pil_image, class_name)

        return LearnResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )
    finally:
        if temp_file_path:
            await cleanup_temp_file(temp_file_path)


@app.get("/classes", response_model=ClassesResponse, tags=["Model Management"])
async def list_food_classes(
    model: FoodRecognitionModel = Depends(get_model),
) -> ClassesResponse:
    """
    List all learned food classes.

    Returns the total number of classes and their names.
    """
    try:
        model_info = model.get_model_info()
        return ClassesResponse(
            total_classes=model_info["total_classes"], classes=model_info["classes"]
        )
    except Exception as e:
        logger.error(f"Error listing classes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )


@app.get(
    "/classes/{class_name}/exists",
    response_model=ClassExistsResponse,
    tags=["Model Management"],
)
async def check_class_exists(
    class_name: str, model: FoodRecognitionModel = Depends(get_model)
) -> ClassExistsResponse:
    """
    Check if a specific food class exists in the model.

    - **class_name**: Name of the food class to check
    """
    try:
        exists = model.class_exists(class_name)
        return ClassExistsResponse(class_name=class_name, exists=exists)
    except Exception as e:
        logger.error(f"Error checking class existence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Management"])
async def get_model_info(
    model: FoodRecognitionModel = Depends(get_model),
) -> ModelInfoResponse:
    """
    Get detailed information about the current model.

    Returns model configuration, device information, and statistics.
    """
    try:
        model_info = model.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PROCESSING_ERROR,
        )


@app.post("/model/save", response_model=SaveModelResponse, tags=["Model Management"])
async def save_model(
    request: SaveModelRequest, model: FoodRecognitionModel = Depends(get_model)
) -> SaveModelResponse:
    """
    Save the current model to a file.

    - **filename**: Optional custom filename (defaults to timestamped filename)

    The model will be saved in the model storage directory.
    """
    try:
        if request.filename:
            filename = request.filename
            if not filename.endswith(".json"):
                filename += ".json"
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_model_{timestamp}.json"

        filepath = str(MODEL_STORAGE_DIR / filename)
        result = model.save_model(filepath)

        return SaveModelResponse(**result)

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.MODEL_SAVE_ERROR,
        )


@app.post("/model/load", response_model=LoadModelResponse, tags=["Model Management"])
async def load_model(
    request: LoadModelRequest, model: FoodRecognitionModel = Depends(get_model)
) -> LoadModelResponse:
    """
    Load a model from a file.

    - **filename**: Name of the model file to load

    The model will be loaded from the model storage directory.
    """
    try:
        filename = request.filename
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = str(MODEL_STORAGE_DIR / filename)
        result = model.load_model(filepath)

        return LoadModelResponse(**result)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.MODEL_LOAD_ERROR,
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "status_code": 500},
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
