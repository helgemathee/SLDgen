import os

import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import dilation
from PIL import Image
from skimage.transform import resize
from torchvision import transforms
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation


def get_mask(im: Image, device):
    """Uses bria model to extract the object mask."""
    # Load the model
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    model.to(device)

    # Preprocess image
    orig_im = np.array(im)
    im_tensor = torch.tensor(orig_im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    image = torch.divide(im_tensor, 255.0)
    image_pre = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)

    # Get the mask
    with torch.no_grad():
        result = model(image_pre)[0][0]
        result = result.squeeze().cpu()

    # Postprocess image
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
    return result


def create_masked_image(image: Image, mask):
    """Apply a mask to an image, setting masked-out regions to white."""
    # Convert the image to a numpy array and normalize
    im_np = np.array(image)
    im_np = im_np / np.iinfo(im_np.dtype).max

    # Apply mask to the image
    im_np = np.expand_dims(mask, axis=-1) * im_np
    im_np[mask < mask.mean()] = 1

    # Convert back to an image
    im_final = (im_np * 255).astype(np.uint8)
    masked_im = Image.fromarray(im_final)
    return masked_im


def make_image_mask_square(image: Image, mask):
    """Pad image/mask to square dimensions with white/zero background."""
    # Convert image to numpy array and scale pixel values
    im_np = np.array(image) / 255
    mask_np = np.array(mask)

    # Get image and mask dimensions
    height, width = im_np.shape[0], im_np.shape[1]

    # Determine the dimensions for the new background (same for image and mask)
    max_len = max(height, width)
    new_background = np.ones((max_len, max_len, 3))
    new_mask = np.zeros((max_len, max_len))  # Create a new mask background filled with zeros

    # Calculate the centering offset for the image and mask
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2

    # Place the image and mask onto their respective backgrounds
    new_background[y : y + height, x : x + width] = im_np
    new_mask[y : y + height, x : x + width] = mask_np

    # Normalize and convert the new background to 0-255 scale
    new_background = (new_background * 255).astype(np.uint8)

    # Convert numpy arrays back to Image objects
    new_im = Image.fromarray(new_background)
    new_im_mask = torch.from_numpy(new_mask)

    return new_im, new_im_mask


def get_obj_bb(binary_im):
    """Extract bounding box coordinates of non-zero region in binary image."""
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1


def cut_and_resize(im, x0, x1, y0, y1, new_height, new_width, type):
    """Crop object from bounding box, resize it, and center on background."""
    # Crop object from image using bounding box coordinates
    cut_obj = im[y0:y1, x0:x1]
    # Resize cropped object to target dimensions
    resized_obj = resize(cut_obj, (new_height, new_width))
    # Initialize background: zeros for mask, ones (white) for image
    if type == "mask":
        new_mask = np.zeros(im.shape)
    else:  # type == image
        new_mask = np.ones(im.shape)
    # Calculate center positions to place resized object in middle of canvas
    center_y_new = int(new_height / 2)
    center_x_new = int(new_width / 2)
    center_targ_y = int(new_mask.shape[0] / 2)
    center_targ_x = int(new_mask.shape[1] / 2)
    # Compute top-left corner for centered placement
    startx, starty = center_targ_x - center_x_new, center_targ_y - center_y_new
    # Place resized object at center of background
    new_mask[starty : starty + resized_obj.shape[0], startx : startx + resized_obj.shape[1]] = (
        resized_obj
    )
    return new_mask


def rescale_obj(target: Image, mask, args):
    """Rescale object if it exceeds target size, maintaining aspect ratio."""
    # Convert target to numpy and binarize mask at threshold
    im_np = np.array(target)
    test_mask = mask.clone()
    test_mask[test_mask < 0.5] = 0
    test_mask[test_mask >= 0.5] = 1

    # Get object bounding box dimensions
    w, h = target.size[0], target.size[1]
    x0, x1, y0, y1 = get_obj_bb(test_mask)
    im_width, im_height = x1 - x0, y1 - y0
    max_size = max(im_width, im_height)
    target_size = int(args.render_size * args.object_size_ratio)
    # If object exceeds target size, rescale while preserving aspect ratio
    if max_size > target_size:
        if im_width > im_height:
            new_width, new_height = target_size, int((target_size / im_width) * im_height)
        else:
            new_width, new_height = int((target_size / im_height) * im_width), target_size

        # Crop object bounding box and resize both mask and image to maintain aspect ratio
        mask_np3 = np.stack([test_mask] * 3, axis=-1)
        mask = cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width, "mask")
        mask = torch.from_numpy(mask[:, :, 0])

        target_np = im_np / np.iinfo(im_np.dtype).max
        im_np = cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width, "image")
        im_np_final = (im_np * 255).astype(np.uint8)
        target = Image.fromarray(im_np_final)

        # Store bounding box and compute scaling factors for later recovery
        args.obj_bb = (x0, x1, y0, y1)
        args.original_center_y = (y0 + (y1 - y0) / 2) / h
        args.original_center_x = (x0 + (x1 - x0) / 2) / w
        args.scale_w = new_width / im_width
        args.scale_h = new_height / im_height

        args.true_scale_w = new_width / args.render_size
        args.true_scale_h = new_height / args.render_size

    return target, mask


def save_mask(mask, save_path):
    """Binarize and save mask as PNG image."""
    mask_save = mask.clone()
    mask_save[mask_save < 0.5] = 0
    mask_save[mask_save >= 0.5] = 1
    mask_save = (mask_save.cpu().numpy() * 255).astype(np.uint8)

    # Create PIL image and save to disk
    mask_image = Image.fromarray(mask_save)
    mask_image.save(os.path.join(save_path, "mask.png"))


def get_target(args):
    """Load and preprocess target image and mask for rendering."""
    args.original_target_path = args.target
    target = Image.open(args.target)

    # If image has alpha channel, composite it onto a white background to remove transparency
    if target.mode == "RGBA":
        new_image = Image.new("RGBA", target.size, "WHITE")
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    # Create mask and apply it to the image (object isolation)
    if not args.calligraphy:
        mask = get_mask(target, args.device)
    else:
        mask = torch.from_numpy(
            (np.array(target) < (np.array(target).max() - 1))[:, :, 0].astype(float)
        )
        mask = dilation(mask.unsqueeze(0).unsqueeze(0), torch.ones(7, 7))[0][0]
    target = create_masked_image(target, mask)
    target, mask = make_image_mask_square(target, mask)

    # Resize image and mask to match render size
    target = target.resize((args.render_size, args.render_size))
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]

    # Reduces the size of the object on the canvas if needed
    target, mask = rescale_obj(target, mask, args)

    # Store preprocessed inputs back into args for downstream use, and save them
    args.input_image = target
    args.mask = mask
    target.save(f"{args.output_dir}/input.png")
    save_mask(mask, args.output_dir)

    data_transform = transforms.ToTensor()
    target = data_transform(target).unsqueeze(0).to(args.device)

    return target, mask
