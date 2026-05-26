# Taken from https://github.com/joanrod/star-vector/blob/main/starvector/metrics/compute_dino_score.py

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOScoreCalculator:
    def __init__(self, device="cuda"):
        super().__init__()
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(device)
        self.device = device

    def get_DINOv2_model(self, model_size):
        """Load and return the DINOv2 model and image processor for the specified size."""
        if model_size == "small":
            model_size = "facebook/dinov2-small"
        elif model_size == "base":
            model_size = "facebook/dinov2-base"
        elif model_size == "large":
            model_size = "facebook/dinov2-large"
        else:
            raise ValueError(
                f"model_size should be either 'small', 'base' or 'large', got {model_size}"
            )
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)

    def process_input(self, image, processor):
        """Extract DINOv2 features from an image (path, PIL Image, or tensor)."""
        # Convert file path to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            # Process PIL Image and extract feature embeddings
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            # Handle pre-computed features or tensors
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")

        return features

    def calculate_DINOv2_similarity_score(self, original_image, generated_image):
        """Compute DINOv2-based similarity score between two images (range: [0, 1])."""
        image1 = original_image
        image2 = generated_image
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        # Calculate cosine similarity between feature embeddings
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()

        # Normalize similarity score from [-1, 1] range to [0, 1] range
        sim = (sim + 1) / 2

        return sim


def DINOv2_score(original_image, generated_image, device):
    """Convenience function to compute DINOv2 similarity score between two images."""
    calculator = DINOScoreCalculator(device)
    return calculator.calculate_DINOv2_similarity_score(original_image, generated_image)
