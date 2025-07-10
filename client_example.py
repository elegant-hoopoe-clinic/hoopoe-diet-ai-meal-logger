"""
Example client for testing the Food Recognition API.
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional


class FoodRecognitionClient:
    """Client for interacting with the Food Recognition API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict_food(self, image_path: str, top_k: int = 1) -> Dict[str, Any]:
        """
        Predict food type from an image.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Prediction results
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            params = {"top_k": top_k}
            response = self.session.post(
                f"{self.base_url}/predict", files=files, params=params
            )
        response.raise_for_status()
        return response.json()

    def learn_food_class(
        self, image_path: str, class_name: str, n_prototypes: int = 3
    ) -> Dict[str, Any]:
        """
        Learn a new food class.

        Args:
            image_path: Path to the image file
            class_name: Name of the food class
            n_prototypes: Number of prototypes to generate

        Returns:
            Learning results
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"class_name": class_name, "n_prototypes": n_prototypes}
            response = self.session.post(
                f"{self.base_url}/learn", files=files, data=data
            )
        response.raise_for_status()
        return response.json()

    def update_food_class(self, image_path: str, class_name: str) -> Dict[str, Any]:
        """
        Update an existing food class with a new example.

        Args:
            image_path: Path to the image file
            class_name: Name of the food class

        Returns:
            Update results
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"class_name": class_name}
            response = self.session.post(
                f"{self.base_url}/update", files=files, data=data
            )
        response.raise_for_status()
        return response.json()

    def list_classes(self) -> Dict[str, Any]:
        """List all learned food classes."""
        response = self.session.get(f"{self.base_url}/classes")
        response.raise_for_status()
        return response.json()

    def check_class_exists(self, class_name: str) -> Dict[str, Any]:
        """Check if a food class exists."""
        response = self.session.get(f"{self.base_url}/classes/{class_name}/exists")
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def save_model(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Save the current model."""
        data = {}
        if filename:
            data["filename"] = filename

        response = self.session.post(f"{self.base_url}/model/save", json=data)
        response.raise_for_status()
        return response.json()

    def load_model(self, filename: str) -> Dict[str, Any]:
        """Load a saved model."""
        data = {"filename": filename}
        response = self.session.post(f"{self.base_url}/model/load", json=data)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the client."""
    client = FoodRecognitionClient()

    try:
        # Check health
        print("=== Health Check ===")
        health = client.health_check()
        print(json.dumps(health, indent=2))

        # Get model info
        print("\n=== Model Info ===")
        model_info = client.get_model_info()
        print(json.dumps(model_info, indent=2))

        # List classes
        print("\n=== Current Classes ===")
        classes = client.list_classes()
        print(json.dumps(classes, indent=2))

        # Example: Learn a new class (uncomment and provide actual image path)
        # print("\n=== Learning New Class ===")
        # result = client.learn_food_class("path/to/pizza.jpg", "pizza")
        # print(json.dumps(result, indent=2))

        # Example: Make prediction (uncomment and provide actual image path)
        # print("\n=== Making Prediction ===")
        # prediction = client.predict_food("path/to/test_image.jpg", top_k=3)
        # print(json.dumps(prediction, indent=2))

    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to the API. Make sure it's running at http://localhost:8000"
        )
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
