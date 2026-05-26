import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
from transformers import CLIPModel, CLIPProcessor


def preprocess_image(img_path):
    """Load an image and convert it to a PyTorch tensor in CxHxW format."""
    img = Image.open(img_path).convert("RGB")
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to CxHxW
    return img


def CLIP_score_text_image(text, generated_image_path):
    """Compute CLIP similarity score between text and an image."""
    # Load pre-trained CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    # Compute score using torchmetrics
    clip_score = (
        _clip_score_update(text, preprocess_image(generated_image_path), model, processor)[0]
        .detach()
        .item()
    )
    return clip_score


def CLIP_score_image_image(original_image_path, generated_image_path):
    """Compute CLIP similarity score between two images."""
    # Load pre-trained CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    # Compute score using torchmetrics
    clip_score = (
        _clip_score_update(
            preprocess_image(original_image_path),
            preprocess_image(generated_image_path),
            model,
            processor,
        )[0]
        .detach()
        .item()
    )
    return clip_score
