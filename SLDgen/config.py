import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pydiffvg
import torch


def set_output_directories(args):
    """Set up output directories based on the target image and experiment name."""
    if args.output_dir == "":
        args.output_dir = Path.cwd() / "output"
    else:
        args.output_dir = Path(args.output_dir)
    args.output_dir = args.output_dir.absolute()

    target_name = Path(args.target).stem
    output_dir = args.output_dir / target_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_name == "":
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.experiment_name = f"{now}"

    args.output_dir = output_dir / args.experiment_name
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    png_logs_dir = args.output_dir / "svg_to_png"
    if not png_logs_dir.exists():
        png_logs_dir.mkdir(parents=True, exist_ok=True)

    svg_logs_dir = args.output_dir / "svg_logs"
    if not svg_logs_dir.exists():
        svg_logs_dir.mkdir(parents=True, exist_ok=True)

    weights_logs_dir = args.output_dir / "weights_logs"
    if not weights_logs_dir.exists():
        weights_logs_dir.mkdir(parents=True, exist_ok=True)


def set_device(args):
    """Set up the device for computation based on availability and user preference."""
    use_gpu = not args.use_cpu
    if not torch.cuda.is_available():
        use_gpu = False
        print("CUDA is not configured with GPU, running with CPU instead.")
    if use_gpu:
        args.device = torch.device(
            "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
        )
    else:
        args.device = torch.device("cpu")

    pydiffvg.set_use_gpu(torch.cuda.is_available() and use_gpu)
    pydiffvg.set_device(args.device)


def set_seed(seed):
    """Set random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def parse_arguments(custom_args=None):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--output-dir", type=str, default="./output/", help="Directory to save the output images."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help=(
            "Name of the experiment, for logging and saving purposes. If not provided, a name will "
            "be generated based on the configuration."
        ),
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Set this flag to use CPU instead of GPU, even if GPU is available.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this flag to print loss values during optimization.",
    )
    parser.add_argument("--debug", action="store_true", help="Set this flag to enable debug mode.")

    # Target image
    parser.add_argument("--target", type=str, required=True, help="Target image path.")
    parser.add_argument(
        "--object-size-ratio",
        type=float,
        default=0.75,
        help="Maximum size of the object relative to the render size, used to rescale the object.",
    )
    parser.add_argument("--render-size", type=int, default=512, help="Size of the rendered image.")
    parser.add_argument(
        "--calligraphy",
        action="store_true",
        help="Set this flag if the target is a calligraphy image (a letter).",
    )
    # Optimization params
    parser.add_argument(
        "--num-iter", type=int, default=4000, help="Number of optimization iterations to run."
    )
    parser.add_argument(
        "--lr", type=float, default=0.8, help="Base learning rate for the optimizer."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Interval (in iterations) at which to save intermediate results.",
    )

    # Curve params
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=5000,
        help="the number of points to sample from the curve for rendering",
    )
    parser.add_argument(
        "--no-optimize-cp-weights",
        action="store_false",
        dest="optimize_cp_weights",
        help=(
            "Optimize the weights of the control points for B-spline by default, set this flag to "
            "not optimize them."
        ),
    )
    parser.add_argument(
        "--keep-low-weights",
        action="store_false",
        dest="prune_low_weights",
        help="Prune low weights by default, set this flag to keep them.",
    )

    parser.add_argument(
        "--init-method",
        type=str,
        default="tsp",
        help="How to initialize the single line: trefoil, contour or tsp.",
    )

    parser.add_argument(
        "--n-control-points",
        type=int,
        default=385,
        help="Number of control points at the beginning of optimization.",
    )

    def float_or_str(value):
        """Custom argparse type to allow either a float or specific string values for width arg."""
        try:
            return float(value)
        except ValueError:
            allowed_strings = ["random", "optim", "optim_random"]
            if value in allowed_strings:
                return str(value)
            raise argparse.ArgumentTypeError(f"Invalid value: {value}")

    parser.add_argument(
        "--width",
        type=float_or_str,
        default=1.0,
        help="Stroke width or 'optim' for variable width.",
    )

    parser.add_argument(
        "--fixed-endpoints",
        action="store_true",
        help="Keep the endpoints fixed during optimization for easy drawing connection.",
    )

    parser.add_argument(
        "--origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Optional. Pin the start of the curve (control_points[0]) to this "
            "normalized location, given as two floats in [0, 1]: X is the fraction "
            "from the left, Y the fraction from the top. The pinned point is excluded "
            "from optimization so gradient descent cannot move it. If omitted, "
            "behavior is identical to upstream. Only valid with --init-method tsp, "
            "and mutually exclusive with --fixed-endpoints."
        ),
    )

    # SDS loss parameters
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument("--conditioning-scale", type=float, default=0.5)
    parser.add_argument(
        "--condition",
        type=str,
        default="depth",
        choices=["depth", "canny"],
        help="Choose between depth and canny",
    )
    parser.add_argument(
        "--lora-model",
        type=str,
        default="./SLDgen/guidance/sld-lora.safetensors",
        help="Path to the LoRA model weights, if empty, no LoRA is used",
    )
    parser.add_argument(
        "--lora-weight",
        type=float,
        default=0.1,
        help="Scale for the LoRA model, if 0, no LoRA is used",
    )

    # Avoidance constraint (opt-in). When --avoid is unset, behavior is identical
    # to upstream: no avoid points are loaded and the avoidance loss is skipped.
    parser.add_argument(
        "--avoid",
        type=str,
        nargs="+",
        default=None,
        metavar="SVG",
        help=(
            "Optional. One or more SVG files whose paths the newly-generated curve "
            "should avoid (be repelled from). Points are sampled along every path "
            "in each SVG and treated as fixed obstacles during optimization. The "
            "SVGs are assumed to be in canvas pixel coordinates at --render-size "
            "(a warning is printed if their viewBox/size differs). Enables "
            "sequential compositional workflows where each new line respects "
            "previously-generated lines. Composes with --origin and "
            "--fixed-endpoints. If omitted, behavior is identical to upstream."
        ),
    )
    parser.add_argument(
        "--avoidance-weight",
        type=float,
        default=0.004,
        help=(
            "Strength of the avoidance repulsion (mirrors --repulsion-loss-weight). "
            "Only used when --avoid is set."
        ),
    )
    parser.add_argument(
        "--avoidance-distance",
        type=float,
        default=25.0,
        help=(
            "Distance threshold in canvas pixel units below which the avoidance "
            "repulsion acts (mirrors the d0=25 of the intra-curve repulsion loss). "
            "Only used when --avoid is set."
        ),
    )

    # Attraction constraint (opt-in). Structural mirror of --avoid, but pulls the
    # curve TOWARD the sampled points instead of repelling it. When --attract is
    # unset, behavior is identical to upstream. Composes with --avoid (attract
    # your own partition, avoid the others) and with --origin.
    parser.add_argument(
        "--attract",
        type=str,
        nargs="+",
        default=None,
        metavar="SVG",
        help=(
            "Optional. One or more SVG files whose sampled points the "
            "newly-generated curve should be attracted (pulled) toward. Points "
            "are sampled along every path in each SVG (same loader/coordinate "
            "convention as --avoid) and treated as fixed targets during "
            "optimization. Enables partition-aligned workflows: split a master "
            "curve, then attract each fresh run to its own partition SVG. "
            "Composes with --avoid and --origin. If omitted, behavior is "
            "identical to upstream."
        ),
    )
    parser.add_argument(
        "--attraction-weight",
        type=float,
        default=0.004,
        help=(
            "Strength of the attraction pull (matches --avoidance-weight). "
            "Only used when --attract is set."
        ),
    )
    parser.add_argument(
        "--attraction-distance",
        type=float,
        default=25.0,
        help=(
            "Dead-zone radius in canvas pixel units. The attraction pull is "
            "INACTIVE within this distance of the target points and acts only "
            "beyond it, so the curve stays free to explore near the target "
            "structure (inverse of --avoidance-distance's active-within zone). "
            "Only used when --attract is set."
        ),
    )

    # Other losses
    parser.add_argument(
        "--repulsion-loss-weight", type=float, default=0.004, help="Weight for the repulsion loss."
    )
    parser.add_argument(
        "--sparse-loss-weight", type=float, default=2000.0, help="Weight for the sparse loss."
    )
    parser.add_argument(
        "--sparse-loss-type",
        type=float,
        default=1.0,
        help="Type of sparsity loss, define the degree of the loss. 0.0 is a std loss",
    )
    parser.add_argument(
        "--sparse-loss-progressive",
        default="linear",
        type=str,
        help="Progressive sparse loss type, set to anything else for no progressive sparse loss.",
    )
    parser.add_argument(
        "--length-shortening-loss-weight",
        type=float,
        default=0.1,
        help="Weight for the length shortening loss.",
    )

    # Metrics
    parser.add_argument(
        "--aesthetic-predictor-model-path",
        type=str,
        default="./SLDgen/metrics/aesthetic_predictor_v2_5.pth",
        help="Path to the aesthetic predictor model weights",
    )

    # Parse arguments
    args = parser.parse_args(custom_args)

    # Validate the opt-in --origin feature. Default (None) keeps behavior unchanged.
    if args.origin is not None:
        if args.fixed_endpoints:
            parser.error("--origin and --fixed-endpoints are mutually exclusive.")
        if args.init_method != "tsp":
            parser.error("--origin is only supported with --init-method tsp.")
        x, y = args.origin
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            parser.error(f"--origin values must be in [0, 1]; got {args.origin}.")

    # Validate the opt-in --avoid feature. Default (None) keeps behavior unchanged.
    if args.avoid is not None:
        for svg_path in args.avoid:
            if not Path(svg_path).exists():
                parser.error(f"--avoid SVG file does not exist: {svg_path}")

    # Validate the opt-in --attract feature. Default (None) keeps behavior unchanged.
    if args.attract is not None:
        for svg_path in args.attract:
            if not Path(svg_path).exists():
                parser.error(f"--attract SVG file does not exist: {svg_path}")

    # Set some fixed parameters
    args.diffusion_model = "stabilityai/stable-diffusion-3.5-medium"
    args.diffusion_timesteps = 1000
    args.diffusion_guidance_scale = 100
    args.vae_path = "madebyollin/taesd3"
    args.multisteps = 1
    args.negative_caption = ""

    # Check that the target image exists
    assert Path(args.target).exists(), f"{args.target} does not exist!"

    # Set up output directories
    set_output_directories(args)

    # Set up device
    set_device(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    return args
