# Food Recognition API

A professional, production-ready FastAPI-based food recognition service using OpenAI's CLIP model and one-shot learning. This API enables real-time food identification from images, supports learning new food classes from a single example, and provides robust model management features.

---

## Project Overview

The Food Recognition API is designed for efficient, extensible, and accurate food classification from images. It leverages OpenAI's CLIP model for extracting image embeddings and FAISS for fast similarity search. The API supports one-shot learning, allowing new food classes to be added with minimal data, and provides endpoints for prediction, learning, updating, and managing the model.

**Key Technologies & Tools:**

- **FastAPI**: High-performance, async Python web framework for building APIs.
- **OpenAI CLIP**: Used for extracting semantic image embeddings.
- **PyTorch**: Backend for CLIP model inference.
- **FAISS**: Efficient similarity search for prototype-based classification.
- **Pillow**: Image processing and validation.
- **Docker**: Containerization for easy deployment.
- **Uvicorn**: ASGI server for running FastAPI with async support.
- **pytest**: For automated API testing.
- **aiofiles**: Async file operations for efficient upload handling.
- **Async/Await**: Used throughout the API for non-blocking I/O, enabling high concurrency and responsiveness.

---

## Project Contents

| File/Directory         | Description                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------|
| `app/`                 | Main application package containing all core logic and API endpoints.                       |
| ├── `main.py`          | FastAPI app, all API endpoints, async logic, and application lifecycle management.          |
| ├── `models.py`        | Food recognition model: CLIP integration, FAISS index, learning, prediction, persistence.   |
| ├── `schemas.py`       | Pydantic models for request/response validation and OpenAPI docs.                           |
| ├── `utils.py`         | Utility functions: file handling, validation, error messages.                               |
| ├── `config.py`        | Centralized configuration: paths, thresholds, API metadata, env vars.                       |
| └── `__init__.py`      | Package metadata.                                                                           |
| `models/`              | Directory for storing the main model file (JSON with learned classes/prototypes).           |
| `model_storage/`       | Directory for saving/loading multiple model versions.                                       |
| `requirements.txt`     | Python dependencies for the project.                                                        |
| `Dockerfile`           | Docker build instructions for containerized deployment.                                     |
| `docker-compose.yml`   | Docker Compose config for multi-container orchestration and volume mounting.                |
| `run_dev.py`           | Script for running the development server with auto-reload and env setup.                   |
| `test_api.py`          | Automated tests for all API endpoints using pytest and FastAPI TestClient.                  |
| `client_example.py`    | Example Python client for interacting with the API.                                         |
| `manage.ps1`           | PowerShell script for setup, testing, Docker, and cleaning (Windows).                      |
| `Makefile`             | Build automation for PowerShell users.                                                      |
| `.env.example`         | Example environment variable file for configuration.                                        |
| `README.md`            | This documentation file.                                                                    |                                     |
| `FILES_TO_REMOVE.md`   | List of files that can be deleted for a minimal production setup.                           |                                                 |

---

## How to Use

### 1. Run with Docker (Recommended)

```bash
docker build -t food-recognition-api .
docker run -p 8000:8000 -v ${PWD}/models:/app/models -v ${PWD}/model_storage:/app/model_storage food-recognition-api
```

Or with Docker Compose:

```bash
docker-compose up --build
```

- API will be available at: [http://localhost:8000](http://localhost:8000)
- Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Development (Local)

```bash
pip install -r requirements.txt
python run_dev.py
```

---

## API Endpoints

| Endpoint                       | Method | Description                                                                                   |
|--------------------------------|--------|-----------------------------------------------------------------------------------------------|
| `/health`                      | GET    | Health check. Returns API and model status.                                                   |
| `/predict`                     | POST   | Predict food type from an uploaded image.                                                     |
| `/learn`                       | POST   | Learn a new food class from an image and class name.                                          |
| `/update`                      | POST   | Update an existing food class with a new example image.                                       |
| `/classes`                     | GET    | List all learned food classes.                                                                |
| `/classes/{class_name}/exists` | GET    | Check if a specific food class exists.                                                        |
| `/model/info`                  | GET    | Get current model information and statistics.                                                 |
| `/model/save`                  | POST   | Save the current model to a file (optionally specify filename).                               |
| `/model/load`                  | POST   | Load a model from a file (specify filename).                                                  |

### Example: `/predict`

- **Request:** `POST /predict` (multipart/form-data)
  - `image`: Image file (JPG, PNG, etc.)
  - `top_k`: (optional) Number of top predictions to return (default: 3)
- **Response:**
  ```json
  {
    "success": true,
    "is_known_food": true,
    "best_prediction": {"class_name": "pizza", "confidence": 0.92, "is_confident": true},
    "predictions": [
      {"class_name": "pizza", "confidence": 0.92, "is_confident": true},
      {"class_name": "burger", "confidence": 0.65, "is_confident": false}
    ],
    "threshold": 0.75,
    "message": null
  }
  ```

### Example: `/learn`

- **Request:** `POST /learn` (multipart/form-data)
  - `image`: Image file
  - `class_name`: Name for the new food class
  - `n_prototypes`: (optional) Number of prototypes to generate (default: 3)
- **Response:**
  ```json
  {
    "success": true,
    "message": "Class 'pizza' learned successfully!",
    "class_name": "pizza",
    "total_classes": 1
  }
  ```

### All endpoints return structured JSON responses with clear status and error messages.



## Project Scenarios

This section describes recommended integration scenarios and best practices for using the Food Recognition API in your application. Please note that the API returns food names in the format:  
**`"foodIdInDatabase-Food name"`**  
(e.g., `"123-Pizza"`), which allows you to map predictions directly to your own food database.

### Database Integration

To ensure robust and auditable learning, we recommend adding an **extra table** in your database to track user activity and feedback related to food recognition. This table should store:

- User actions (e.g., "add new class", "update class", "incorrect guess")
- Uploaded images and metadata
- The food class or correction suggested by the user
- Review status (pending/approved/rejected)

A reviewer should periodically check this table. Once an entry is approved, your backend can call the `/learn` or `/update` endpoint of this API to update the recognition model.

### Scenario 1: Model Returns a Correct Answer

- If the model's top prediction is correct **and the confidence is high** (`is_confident: true`), do **not** send this image for learning or updating. This prevents overfitting and keeps the model robust.
- If the prediction is correct but **confidence is low** (`is_confident: false`), add the image and its label to your feedback table for review. After approval, use the `/update` endpoint to improve the model for this class.

### Scenario 2: Model Returns an Incorrect Answer

- The API returns both the best prediction and a list of top guesses.
- Show these guesses to the user. If the correct food is among them, let the user select it.
    - Add this feedback to your table. After review, call `/update` with the correct image and class.
- If **none of the guesses are correct**:
    - Allow the user to search your food database. If the food exists, add the image and correct class to your table for review and later call `/update`.
    - If the food is **not in your database**, first add it to your food database, then after review, call `/learn` to add the new class to the model.

### Scenario 3: Adding New Foods

- For foods not recognized by the model or not present in your database, add them to your food database first.
- After review and approval, use the `/learn` endpoint to teach the model about the new food class.

### Model Management

- **Saving the Model:**  
  Use the `/model/save` endpoint to save the current state of the model. You can specify a filename or let the API generate one with a timestamp.
- **Loading a Model:**  
  Use the `/model/load` endpoint to restore a previous model version by filename.

**Best Practice:**  
Regularly save and back up your model files (from the `model_storage/` directory). If the model's performance degrades, you can quickly revert to a previous stable version by loading an older model file.

---

## Additional Notes

- **Async/Await**: All API endpoints use async for non-blocking file I/O and high concurrency.
- **Model Persistence**: Model is automatically saved on shutdown and loaded on startup.
- **Extensibility**: Add new endpoints, augmentations, or model logic as needed.
- **Testing**: Use `pytest test_api.py -v` for automated API tests.
- **Security**: Input validation and file type checks are enforced.

---
