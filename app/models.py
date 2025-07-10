"""
Core food recognition model implementation.
"""

import os
import torch
import clip
import faiss
import numpy as np
import json
import logging
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define augmentation strategies
DEFAULT_AUGMENTATIONS = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ]
)

STRONG_AUGMENTATIONS = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ]
)


class FoodRecognitionModel:
    """
    Professional food recognition model using CLIP embeddings and FAISS for efficient similarity search.

    This model implements one-shot learning capabilities with robust embedding generation
    through data augmentation and prototype-based classification.
    """

    def __init__(
        self, similarity_threshold: float = 0.75, device: Optional[str] = None
    ):
        """
        Initialize the food recognition model.

        Args:
            similarity_threshold: Minimum similarity score for confident predictions
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold

        # Model components
        self.model = None
        self.preprocess = None
        self.prototypes: Dict[str, np.ndarray] = {}
        self.class_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)

        # FAISS index for efficient similarity search
        self.index = None
        self.class_names: List[str] = []

        # Initialize CLIP model
        self._initialize_clip_model()

        logger.info(f"FoodRecognitionModel initialized on device: {self.device}")

    def _initialize_clip_model(self):
        """Initialize the CLIP model and preprocessing pipeline."""
        try:
            model_name = "ViT-L/14"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"CLIP model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings for cosine similarity."""
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def _get_robust_embedding(
        self, image: Image.Image, n_augment: int = 5, use_strong_aug: bool = False
    ) -> np.ndarray:
        """
        Generate robust embedding using multiple augmentations.

        Args:
            image: PIL Image object
            n_augment: Number of augmentations to apply
            use_strong_aug: Whether to use strong augmentations

        Returns:
            L2-normalized embedding vector
        """
        embeddings = []

        # Original embedding
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_input)
        embeddings.append(emb.cpu().numpy())

        # Augmented embeddings
        aug_transform = (
            STRONG_AUGMENTATIONS if use_strong_aug else DEFAULT_AUGMENTATIONS
        )
        for _ in range(n_augment):
            try:
                aug_image = aug_transform(image)
                aug_image = transforms.ToPILImage()(aug_image)
                image_input = self.preprocess(aug_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model.encode_image(image_input)
                embeddings.append(emb.cpu().numpy())
            except Exception as e:
                logger.warning(f"Failed to process augmentation: {e}")
                continue

        # Average embeddings
        embedding = np.mean(np.vstack(embeddings), axis=0)
        embedding = embedding.flatten().reshape(1, -1).astype("float32")
        return self._l2_normalize(embedding)

    def _rebuild_index(self):
        """Rebuild FAISS index with current prototypes."""
        if not self.prototypes:
            return

        # Stack all prototypes
        prototype_embeddings = []
        self.class_names = []

        for class_name, prototype in self.prototypes.items():
            prototype_embeddings.append(prototype)
            self.class_names.append(class_name)

        embeddings_np = np.vstack(prototype_embeddings)
        embedding_dim = embeddings_np.shape[1]

        # Create FAISS index for cosine similarity (Inner Product with L2 normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings_np)

        logger.info(f"FAISS index rebuilt with {len(self.class_names)} classes")

    def learn_new_class(
        self, image: Image.Image, class_name: str, n_prototypes: int = 3
    ) -> Dict[str, Any]:
        """
        Learn a new food class from a single image.

        Args:
            image: PIL Image object
            class_name: Name of the food class
            n_prototypes: Number of prototype embeddings to generate

        Returns:
            Dictionary with learning results
        """
        try:
            logger.info(f"Learning new class: {class_name}")

            # Generate multiple prototype embeddings using augmentation
            embeddings = []
            try:
                for i in range(n_prototypes):
                    logger.info(
                        f"Generating embedding {i+1}/{n_prototypes} for class '{class_name}'"
                    )
                    emb = self._get_robust_embedding(
                        image, n_augment=8, use_strong_aug=True
                    )
                    logger.info(f"Generated embedding {i+1} with shape: {emb.shape}")
                    embeddings.append(emb)
            except Exception as e:
                logger.error(
                    f"Failed to generate embeddings for class '{class_name}': {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(
                    f"Embedding generation traceback: {traceback.format_exc()}"
                )
                return {
                    "success": False,
                    "message": f"Failed to generate embeddings: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Create prototype as average of augmented embeddings
            try:
                logger.info(f"Creating prototype from {len(embeddings)} embeddings")
                prototype = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
                logger.info(f"Prototype shape before normalization: {prototype.shape}")
                prototype = self._l2_normalize(prototype)
                logger.info(f"Prototype shape after normalization: {prototype.shape}")
            except Exception as e:
                logger.error(
                    f"Failed to create prototype for class '{class_name}': {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(f"Prototype creation traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Failed to create prototype: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Store prototype and embeddings
            try:
                logger.info(
                    f"Storing prototype and embeddings for class '{class_name}'"
                )
                self.prototypes[class_name] = prototype

                # Ensure class_embeddings[class_name] is initialized as a list
                if class_name not in self.class_embeddings:
                    self.class_embeddings[class_name] = []
                elif not isinstance(self.class_embeddings[class_name], list):
                    logger.warning(
                        f"class_embeddings['{class_name}'] was not a list, converting"
                    )
                    self.class_embeddings[class_name] = []

                self.class_embeddings[class_name].extend(embeddings)
                logger.info(
                    f"Stored {len(embeddings)} embeddings for class '{class_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to store data for class '{class_name}': {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(f"Storage traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Failed to store class data: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Rebuild FAISS index
            try:
                logger.info("Rebuilding FAISS index")
                self._rebuild_index()
                logger.info("FAISS index rebuilt successfully")
            except Exception as e:
                logger.error(f"Failed to rebuild FAISS index: {type(e).__name__}: {e}")
                import traceback

                logger.error(f"FAISS rebuild traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Failed to rebuild FAISS index: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            logger.info(f"Class '{class_name}' learned successfully!")
            return {
                "success": True,
                "message": f"Class '{class_name}' learned successfully!",
                "class_name": class_name,
                "total_classes": len(self.prototypes),
            }

        except Exception as e:
            logger.error(
                f"Unexpected error in learn_new_class for '{class_name}': {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(f"Learn new class traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Unexpected error in learn_new_class: {type(e).__name__}: {str(e)}",
                "class_name": class_name,
                "total_classes": len(self.prototypes),
            }

    def predict(self, image: Image.Image, top_k: int = 1) -> Dict[str, Any]:
        """
        Predict the class of a new image.

        Args:
            image: PIL Image object
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.prototypes:
                return {
                    "success": True,
                    "is_known_food": False,
                    "best_prediction": None,
                    "predictions": [],
                    "threshold": self.similarity_threshold,
                    "message": "No classes learned yet",
                }

            # Get query embedding
            query_emb = self._get_robust_embedding(image, n_augment=3)

            # Search in FAISS index
            similarities, indices = self.index.search(
                query_emb, k=min(top_k, len(self.class_names))
            )

            predictions = []
            for i in range(len(similarities[0])):
                class_name = self.class_names[indices[0][i]]
                confidence = float(similarities[0][i])
                predictions.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "is_confident": confidence >= self.similarity_threshold,
                    }
                )

            best_prediction = predictions[0] if predictions else None
            is_known_food = best_prediction and best_prediction["is_confident"]

            return {
                "success": True,
                "is_known_food": is_known_food,
                "best_prediction": best_prediction,
                "predictions": predictions,
                "threshold": self.similarity_threshold,
                "message": None,
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "success": False,
                "is_known_food": False,
                "best_prediction": None,
                "predictions": [],
                "threshold": self.similarity_threshold,
                "message": str(e),
            }

    def update_class(self, image: Image.Image, class_name: str) -> Dict[str, Any]:
        """
        Update existing class with new example or create new class.

        Args:
            image: PIL Image object
            class_name: Name of the food class

        Returns:
            Dictionary with update results
        """
        try:
            logger.info(f"Starting update_class for: {class_name}")

            if class_name not in self.prototypes:
                logger.info(f"Class '{class_name}' not found, creating new class")
                return self.learn_new_class(image, class_name)

            logger.info(f"Generating new embedding for class: {class_name}")
            # Add new embedding to class
            try:
                new_emb = self._get_robust_embedding(image, n_augment=3)
                logger.info(f"New embedding generated with shape: {new_emb.shape}")
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return {
                    "success": False,
                    "message": f"Failed to generate embedding: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Ensure consistent shape - flatten to 1D if needed
            if new_emb.ndim > 1:
                new_emb = new_emb.flatten().reshape(1, -1).astype("float32")
                new_emb = self._l2_normalize(new_emb)
                logger.info(f"Embedding reshaped to: {new_emb.shape}")

            # Validate embedding dimensions match existing ones
            try:
                logger.info(f"Checking existing embeddings for class '{class_name}'")
                logger.info(
                    f"Class exists in prototypes: {class_name in self.prototypes}"
                )
                logger.info(
                    f"Class exists in class_embeddings: {class_name in self.class_embeddings}"
                )

                # Ensure the class exists in class_embeddings
                if class_name not in self.class_embeddings:
                    logger.info(
                        f"Creating new embeddings list for existing class '{class_name}'"
                    )
                    self.class_embeddings[class_name] = []

                logger.info(
                    f"Type of class_embeddings[class_name]: {type(self.class_embeddings[class_name])}"
                )
                logger.info(
                    f"Length of class_embeddings[class_name]: {len(self.class_embeddings[class_name])}"
                )

                if len(self.class_embeddings[class_name]) > 0:
                    existing_emb = self.class_embeddings[class_name][0]
                    logger.info(f"Type of existing_emb: {type(existing_emb)}")
                    logger.info(
                        f"Shape of existing_emb: {existing_emb.shape if hasattr(existing_emb, 'shape') else 'No shape attribute'}"
                    )
                    logger.info(f"New embedding shape: {new_emb.shape}")

                    if (
                        hasattr(existing_emb, "shape")
                        and existing_emb.shape[1] != new_emb.shape[1]
                    ):
                        logger.error(
                            f"Embedding dimension mismatch for class '{class_name}': {existing_emb.shape[1]} vs {new_emb.shape[1]}"
                        )
                        return {
                            "success": False,
                            "message": f"Embedding dimension mismatch for class '{class_name}'",
                            "class_name": class_name,
                            "total_classes": len(self.prototypes),
                        }
                else:
                    logger.info(
                        f"No existing embeddings for class '{class_name}' - this is the first update"
                    )
            except Exception as e:
                logger.error(
                    f"Error during dimension validation for '{class_name}': {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(f"Validation traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Error during dimension validation: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Append the new embedding
            try:
                logger.info(f"Appending new embedding to class '{class_name}'")
                # Double-check class_embeddings[class_name] exists and is a list
                if class_name not in self.class_embeddings:
                    logger.info(f"Initializing new list for class '{class_name}'")
                    self.class_embeddings[class_name] = []
                elif not isinstance(self.class_embeddings[class_name], list):
                    logger.error(
                        f"class_embeddings['{class_name}'] is not a list: {type(self.class_embeddings[class_name])}"
                    )
                    logger.info(f"Converting to list for class '{class_name}'")
                    self.class_embeddings[class_name] = []

                self.class_embeddings[class_name].append(new_emb)
                logger.info(
                    f"Successfully appended embedding. Total embeddings for '{class_name}': {len(self.class_embeddings[class_name])}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to append embedding for '{class_name}': {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(f"Append traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Failed to append embedding: {type(e).__name__}: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Update prototype (moving average)
            try:
                logger.info(f"Computing new prototype for class '{class_name}'")
                all_embeddings = np.vstack(self.class_embeddings[class_name])
                logger.info(f"Stacked embeddings shape: {all_embeddings.shape}")
                new_prototype = np.mean(all_embeddings, axis=0, keepdims=True)
                logger.info(f"New prototype shape: {new_prototype.shape}")
                self.prototypes[class_name] = self._l2_normalize(new_prototype)
                logger.info(
                    f"Prototype updated and normalized for class '{class_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute new prototype for class '{class_name}': {e}"
                )
                # Remove the problematic embedding
                try:
                    self.class_embeddings[class_name].pop()
                    logger.info("Removed problematic embedding")
                except:
                    pass
                return {
                    "success": False,
                    "message": f"Failed to compute new prototype for class '{class_name}': {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            # Rebuild FAISS index
            try:
                logger.info("Rebuilding FAISS index")
                self._rebuild_index()
                logger.info("FAISS index rebuilt successfully")
            except Exception as e:
                logger.error(f"Failed to rebuild FAISS index: {e}")
                return {
                    "success": False,
                    "message": f"Failed to rebuild FAISS index: {str(e)}",
                    "class_name": class_name,
                    "total_classes": len(self.prototypes),
                }

            logger.info(f"Updated class '{class_name}' with new example")
            return {
                "success": True,
                "message": f"Updated class '{class_name}' with new example",
                "class_name": class_name,
                "total_classes": len(self.prototypes),
            }

        except Exception as e:
            logger.error(
                f"Unexpected error in update_class for '{class_name}': {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Unexpected error in update_class: {type(e).__name__}: {str(e)}",
                "class_name": class_name,
                "total_classes": len(self.prototypes),
            }

    def get_class_names(self) -> List[str]:
        """Get list of all learned class names."""
        return list(self.prototypes.keys())

    def class_exists(self, class_name: str) -> bool:
        """Check if a class exists in the model."""
        return class_name in self.prototypes

    def remove_class(self, class_name: str) -> bool:
        """
        Remove a class from the model.

        Args:
            class_name: Name of the class to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            if class_name not in self.prototypes:
                return False

            del self.prototypes[class_name]
            del self.class_embeddings[class_name]

            # Rebuild FAISS index
            self._rebuild_index()

            logger.info(f"Removed class '{class_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to remove class '{class_name}': {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state."""
        return {
            "model_name": "ViT-L/14",
            "device": self.device,
            "similarity_threshold": self.similarity_threshold,
            "total_classes": len(self.prototypes),
            "classes": list(self.prototypes.keys()),
            "has_index": self.index is not None,
            "model_initialized": self.model is not None,
        }

    def save_model(self, filepath: str) -> Dict[str, Any]:
        """
        Save the learned prototypes to a JSON file.

        Args:
            filepath: Path to save the model

        Returns:
            Dictionary with save results
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
                "similarity_threshold": self.similarity_threshold,
                "device": self.device,
                "class_embeddings": {
                    k: [emb.tolist() for emb in v]
                    for k, v in self.class_embeddings.items()
                },
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Model saved to {filepath}")
            return {
                "success": True,
                "message": f"Model saved to {filepath}",
                "filepath": filepath,
                "total_classes": len(self.prototypes),
            }

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return {
                "success": False,
                "message": f"Failed to save model: {str(e)}",
                "filepath": filepath,
                "total_classes": len(self.prototypes),
            }

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load saved prototypes from a JSON file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Dictionary with load results
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file {filepath} not found")
                return {
                    "success": False,
                    "message": f"Model file {filepath} not found",
                    "filepath": filepath,
                    "total_classes": 0,
                    "classes": [],
                }

            with open(filepath, "r") as f:
                data = json.load(f)

            # Load prototypes
            self.prototypes = {k: np.array(v) for k, v in data["prototypes"].items()}
            self.similarity_threshold = data.get("similarity_threshold", 0.75)

            # Load class embeddings if available
            if "class_embeddings" in data:
                self.class_embeddings = {
                    k: [np.array(emb) for emb in v]
                    for k, v in data["class_embeddings"].items()
                }

            # Rebuild FAISS index
            self._rebuild_index()

            logger.info(f"Model loaded from {filepath}")
            logger.info(
                f"Loaded {len(self.prototypes)} classes: {list(self.prototypes.keys())}"
            )
            return {
                "success": True,
                "message": f"Model loaded from {filepath}",
                "filepath": filepath,
                "total_classes": len(self.prototypes),
                "classes": list(self.prototypes.keys()),
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {
                "success": False,
                "message": f"Failed to load model: {str(e)}",
                "filepath": filepath,
                "total_classes": len(self.prototypes),
                "classes": list(self.prototypes.keys()),
            }

    def clear_model(self) -> bool:
        """Clear all learned classes and reset the model."""
        try:
            self.prototypes.clear()
            self.class_embeddings.clear()
            self.index = None
            self.class_names.clear()

            logger.info("Model cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear model: {e}")
            return False
            return False
