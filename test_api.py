"""
Tests for the Food Recognition API.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from PIL import Image
import io
import json

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB image
    image = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "total_classes" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_predict_no_classes(self, client, sample_image):
        """Test prediction when no classes are learned."""
        files = {"image": ("test.jpg", sample_image, "image/jpeg")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No classes learned yet" in data["message"]

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_predict_no_file(self, client):
        """Test prediction without file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error


class TestLearningEndpoint:
    """Test learning endpoint."""

    def test_learn_new_class(self, client, sample_image):
        """Test learning a new class."""
        files = {"image": ("test.jpg", sample_image, "image/jpeg")}
        data = {"class_name": "test_food", "n_prototypes": 2}

        response = client.post("/learn", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["class_name"] == "test_food"

    def test_learn_empty_class_name(self, client, sample_image):
        """Test learning with empty class name."""
        files = {"image": ("test.jpg", sample_image, "image/jpeg")}
        data = {"class_name": "", "n_prototypes": 2}

        response = client.post("/learn", files=files, data=data)
        assert response.status_code == 400
        assert "Class name cannot be empty" in response.json()["detail"]

    def test_learn_invalid_prototypes(self, client, sample_image):
        """Test learning with invalid number of prototypes."""
        files = {"image": ("test.jpg", sample_image, "image/jpeg")}
        data = {"class_name": "test_food", "n_prototypes": 0}

        response = client.post("/learn", files=files, data=data)
        assert response.status_code == 422  # Validation error


class TestModelManagement:
    """Test model management endpoints."""

    def test_get_model_info(self, client):
        """Test getting model information."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "device" in data
        assert "similarity_threshold" in data
        assert "total_classes" in data

    def test_list_classes_empty(self, client):
        """Test listing classes when none exist."""
        response = client.get("/classes")
        assert response.status_code == 200

        data = response.json()
        assert data["total_classes"] == 0
        assert data["classes"] == []

    def test_check_class_exists_nonexistent(self, client):
        """Test checking for non-existent class."""
        response = client.get("/classes/nonexistent/exists")
        assert response.status_code == 200

        data = response.json()
        assert data["class_name"] == "nonexistent"
        assert data["exists"] is False

    def test_save_model(self, client):
        """Test saving model."""
        data = {"filename": "test_model"}
        response = client.post("/model/save", json=data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "test_model" in result["filepath"]


class TestIntegrationWorkflow:
    """Test complete workflow integration."""

    def test_learn_and_predict_workflow(self, client, sample_image):
        """Test complete learn and predict workflow."""
        # Learn a new class
        files = {"image": ("pizza.jpg", sample_image, "image/jpeg")}
        data = {"class_name": "pizza", "n_prototypes": 2}

        learn_response = client.post("/learn", files=files, data=data)
        assert learn_response.status_code == 200

        learn_result = learn_response.json()
        assert learn_result["success"] is True
        assert learn_result["class_name"] == "pizza"

        # Check classes list
        classes_response = client.get("/classes")
        assert classes_response.status_code == 200

        classes_data = classes_response.json()
        assert classes_data["total_classes"] == 1
        assert "pizza" in classes_data["classes"]

        # Check class exists
        exists_response = client.get("/classes/pizza/exists")
        assert exists_response.status_code == 200
        assert exists_response.json()["exists"] is True

        # Create a new sample image for prediction
        sample_image.seek(0)  # Reset image buffer

        # Make prediction
        files = {"image": ("test.jpg", sample_image, "image/jpeg")}
        predict_response = client.post("/predict", files=files)
        assert predict_response.status_code == 200

        predict_result = predict_response.json()
        assert predict_result["success"] is True
        assert len(predict_result["predictions"]) > 0
        assert predict_result["predictions"][0]["class_name"] == "pizza"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
