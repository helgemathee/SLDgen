import os
import warnings

# Disable tokenizers parallelism warning cleanly before importing HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import json

import numpy as np
import torch
import wiregrad as wg
from PIL import Image
from tqdm.auto import tqdm

from .avoidance import avoidance_loss
from .attraction import attraction_loss
from .guidance.sd3_sds_guidance_control import SD3GuidanceControl
from .metrics import get_all_metrics
from .painter.painter import SLDBSplinePainter
from .painter.painter_optimizer import PainterOptimizer
from .targets import get_target
from .utils import increase_object_size, make_video


def get_sparse_loss_weight(args, epoch):
    """Return sparse loss weight, supporting progressive schedules."""

    def is_number(s):
        try:
            float(s)
            return True
        except TypeError:
            return False
        except ValueError:
            return False

    target_weight = args.sparse_loss_weight
    if args.sparse_loss_progressive == "linear":
        return target_weight * (epoch / args.num_iter)
    else:
        return target_weight


def save_current_step(renderer, args, epoch, img):
    """Save the current sketch as SVG and PNG, and optionally the basis spline visualization."""
    renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
    init_img = img.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    init_img = Image.fromarray((init_img * 255).astype(np.uint8))
    init_img.save(f"{args.output_dir}/svg_to_png/iter_{epoch:04d}.png")
    if hasattr(renderer, "save_basis_spline"):
        renderer.save_basis_spline(f"{args.output_dir}/weights_logs/basis_spline_iter{epoch}.svg")


def run(args):
    print("Running SLDgen:", flush=True)

    # Set up input, renderer and optimizer
    inputs, mask = get_target(args)
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer = renderer.to(args.device)
    optimizer = PainterOptimizer(args, renderer)

    # Initialize renderer and optimizer
    init_img = renderer.init_image()
    optimizer.init_optimizers()

    # Setting up the SDS loss
    print(f"\tUsing {args.diffusion_model} as the diffusion model.", flush=True)
    sds_loss = SD3GuidanceControl(args=args, device=args.device)

    print("\tStarting the optimization process...", flush=True)

    # Save the initial drawing before optimization
    save_current_step(renderer, args, epoch=0, img=init_img)

    # Optimization loop
    inputs = inputs.detach()
    epoch_range = tqdm(range(args.num_iter + 1), bar_format="    {l_bar}{bar}{r_bar}")
    for epoch in epoch_range:
        optimizer.zero_grad_()

        # Semantic loss
        raster_sld = renderer.get_image().to(args.device)
        loss = sds_loss(raster_sld)
        loss.backward(retain_graph=True)

        # Regularization losses
        tqdm_update = dict()
        loss = None
        if args.repulsion_loss_weight > 0:
            loss = 0.0
            repulsion_loss = wg.repulsion_loss(renderer.sampled_curve3d, d0=25, cyclic=False)
            loss += args.repulsion_loss_weight * repulsion_loss

            tqdm_update["Repulsion Loss"] = (args.repulsion_loss_weight * repulsion_loss).item()

        # Avoidance constraint (opt-in). Only runs when --avoid loaded obstacle
        # points; otherwise this block is skipped entirely and behavior matches
        # upstream. Repels the actively-optimized control points away from the
        # fixed obstacle points, in canvas pixel coordinates (same frame as both).
        if getattr(renderer, "avoid_points", None) is not None:
            if loss is None:
                loss = 0.0
            avoid_loss = avoidance_loss(
                renderer.active_control_points,
                renderer.avoid_points,
                d0=args.avoidance_distance,
            )
            loss += args.avoidance_weight * avoid_loss

            tqdm_update["Avoidance Loss"] = (args.avoidance_weight * avoid_loss).item()

        # Attraction constraint (opt-in). Mirror of the avoidance block above and
        # composes with it: only runs when --attract loaded target points, else
        # skipped entirely. Pulls the actively-optimized control points TOWARD the
        # fixed target points, inactive within the dead-zone radius so the curve
        # stays free near the target structure.
        if getattr(renderer, "attract_points", None) is not None:
            if loss is None:
                loss = 0.0
            attract_loss = attraction_loss(
                renderer.active_control_points,
                renderer.attract_points,
                deadzone=args.attraction_distance,
            )
            loss += args.attraction_weight * attract_loss

            tqdm_update["Attraction Loss"] = (args.attraction_weight * attract_loss).item()

        if args.sparse_loss_weight > 0.0 and args.optimize_cp_weights:
            if loss is None:
                loss = 0.0
            if args.sparse_loss_type == 0.0:
                sparse_loss = get_sparse_loss_weight(args, epoch) / (
                    torch.var(renderer.weights) + 1e-5
                )
            else:
                sparse_loss = (
                    get_sparse_loss_weight(args, epoch)
                    * torch.pow(torch.abs(renderer.weights) + 1e-8, args.sparse_loss_type).mean()
                )
            loss += sparse_loss

            tqdm_update["Sparse Loss Weight"] = get_sparse_loss_weight(args, epoch)
            tqdm_update["Sparse Loss"] = sparse_loss.item()

        if args.length_shortening_loss_weight > 0.0:
            if loss is None:
                loss = 0.0
            sampled_curve = renderer.sampled_curve2d
            segments = sampled_curve[1:] - sampled_curve[:-1]
            lengths = torch.norm(segments, dim=-1)
            length_shortening_loss = torch.sum(lengths) * args.length_shortening_loss_weight
            loss += length_shortening_loss

            tqdm_update["Length Shortening Loss"] = length_shortening_loss.item()

        if loss is not None:
            loss.backward()

        if args.verbose:
            epoch_range.set_description(
                " - ".join(
                    [
                        f"{k}: {v:.2e}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in tqdm_update.items()
                    ]
                )
            )

        # Update parameters
        optimizer.step_()
        with torch.no_grad():
            renderer.post_process_params()

        # Save intermediate steps
        if epoch % args.save_interval == 0 and epoch > 0:
            save_current_step(renderer, args, epoch, raster_sld)

    # Save final SLD
    if hasattr(args, "scale_w") or hasattr(args, "scale_h"):
        # Increases the size of the object on the canvas to its original size if it has been reduced
        increase_object_size(renderer, args)
    renderer.save_svg(args.output_dir, "final_sld")

    # Rasterize final SLD at double resolution for better quality, and save it
    renderer.canvas_width *= 2
    renderer.canvas_height *= 2
    renderer.control_points = renderer.control_points * 2
    renderer.width = renderer.width * 2
    if getattr(args, "origin", None) is not None:
        # Keep the pinned origin (and its widths) consistent with the doubled
        # control points for the final double-resolution export.
        renderer.first_origin_points = renderer.first_origin_points * 2
        renderer.first_origin_widths = renderer.first_origin_widths * 2
    raster_sld = renderer.get_image().permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    raster_sld = Image.fromarray((raster_sld * 255).astype(np.uint8))
    raster_sld.save(f"{args.output_dir}/final_sld.png")

    # Compute all metrics
    print("\tComputing metrics...")
    metrics = get_all_metrics(
        f"{args.output_dir}/final_sld.png",
        args.original_target_path,
        args.caption,
        args.device,
        args.aesthetic_predictor_model_path,
    )
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save video
    print("\tSaving video...")
    make_video(args)
    print("Done!")


def set_error_logging():
    """
    Set logging to only show errors for diffusers and transformers libraries to reduce clutter in
    the output.
    """
    from diffusers.utils.logging import disable_progress_bar as diffusers_disable_pb
    from diffusers.utils.logging import set_verbosity_error as diffusers_set_verbosity_error
    from transformers.utils.logging import disable_progress_bar as transformers_disable_pb

    diffusers_set_verbosity_error()
    diffusers_disable_pb()
    transformers_disable_pb()


def save_config(args):
    """Save the configuration parameters to a JSON file in the output directory."""

    final_config = dict()
    for k, v in vars(args).items():
        if k == "mask":
            continue
        else:
            final_config[k] = str(v)
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(final_config, f, indent=4)
