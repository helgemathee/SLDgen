import cv2
import numpy as np
import torch
from diffusers import SD3ControlNetModel
from image_gen_aux import DepthPreprocessor
from PIL import Image


def get_sd35_medium_controlnet(condition, device):
    """
    Load the SD3.5 Medium ControlNet that matches the requested condition.
    See https://huggingface.co/tensorart for details about the controlnets.
    """
    if condition == "depth":
        controlnet = SD3ControlNetModel.from_pretrained(
            "tensorart/SD3.5M-Controlnet-Depth", torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    elif condition == "canny":
        controlnet = SD3ControlNetModel.from_pretrained(
            "tensorart/SD3.5M-Controlnet-Canny", torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    return controlnet


def create_condition(image, condition):
    """Create the requested conditioning image from the input image."""
    if condition == "depth":
        condition = create_depth_condition(image)
    elif condition == "canny":
        condition = create_canny_condition(image)
    return condition


def create_canny_condition(image):
    """Build a three-channel Canny edge map from the input image."""
    # Detect edges on the raw image array.
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    # Expand the single edge channel to RGB so downstream code can treat it like an image.
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def create_depth_condition(image):
    """Generate a depth conditioning image using Depth Anything V2."""
    depth_preprocessor = DepthPreprocessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf"
    ).to("cuda")
    depth_image = depth_preprocessor(image, invert=True)[0].convert("RGB")

    return depth_image
