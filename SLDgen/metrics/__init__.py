from .aesthetic_score import aesthetic_score
from .clip_score import CLIP_score_image_image, CLIP_score_text_image
from .dino_score import DINOv2_score


def get_all_metrics(
    generated_image_path, original_image_path, text, device, aesthetic_predictor_model_path
):
    """Compute all available metrics for generated images and return them in a dictionary."""
    metrics = {}
    metrics["aesthetic_score"] = aesthetic_score(
        generated_image_path, device, aesthetic_predictor_model_path
    )
    metrics["clip_score_text_image"] = CLIP_score_text_image(text, generated_image_path)
    metrics["clip_score_image_image"] = CLIP_score_image_image(
        original_image_path, generated_image_path
    )
    metrics["dino_score"] = DINOv2_score(original_image_path, generated_image_path, device)

    return metrics
