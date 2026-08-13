"""The single translator between a job's parameters and SLDgen's command line.

Spec 2 SS4.2: this is the *only* module that knows the CLI, and it must round-trip
--- ``argv_to_params(params_to_argv(p)) == p``. Everything else in the service
treats parameters as an opaque dict, so when a flag is added to SLDgen this file
is the one place that changes.

Two design points worth stating, because both look like over-specification until
something breaks:

**Parameters are canonicalised, not sparse.** ``canonical_params`` fills in every
default, and ``params_to_argv`` emits every non-null value -- including ones that
happen to equal SLDgen's current default. The command a job records is therefore
still the command that reproduces it after an upstream default changes, and the
round-trip is exact rather than "exact up to defaults".

**Structural vs operational, with nothing in between** (Spec 2 SS4.2). A job's
parameters and its result are one thing; a job that ran under two different
structural parameter sets could not be described by either. Only the operational
settings may be edited after creation.
"""

from dataclasses import dataclass
from pathlib import Path

STRUCTURAL = "structural"
OPERATIONAL = "operational"


class ParamError(ValueError):
    """A parameter set that SLDgen would reject, caught before a segment is spawned."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    flag: str
    kind: str  # int | float | str | float_or_str | true_flag | false_flag | float_pair | path | path_list
    group: str
    default: object


#: Mirrors SLDgen/config.py, except for two operational defaults the service
#: deliberately sets differently from the CLI:
#:
#: * `checkpoint_interval` (CLI default 0): Spec 2 SS8 wants a non-zero default so
#:   a crash never costs more than 200 iterations.
#: * `save_video` (CLI default True): service jobs are served as frames -- the web
#:   UI scrubs `svg_to_png/iter_*.png` and never plays the mp4 -- so the default
#:   is off, and a daemon host needs no ffmpeg. Set it per job to get one back.
PARAM_SPECS = (
    ParamSpec("render_size", "--render-size", "int", STRUCTURAL, 512),
    ParamSpec("n_control_points", "--n-control-points", "int", STRUCTURAL, 385),
    ParamSpec("init_method", "--init-method", "str", STRUCTURAL, "tsp"),
    ParamSpec("seed", "--seed", "int", STRUCTURAL, 0),
    ParamSpec("num_iter", "--num-iter", "int", STRUCTURAL, 4000),
    ParamSpec("optimize_cp_weights", "--no-optimize-cp-weights", "false_flag", STRUCTURAL, True),
    ParamSpec("prune_low_weights", "--keep-low-weights", "false_flag", STRUCTURAL, True),
    ParamSpec("width", "--width", "float_or_str", STRUCTURAL, 1.0),
    ParamSpec("origin", "--origin", "float_pair", STRUCTURAL, None),
    ParamSpec("fixed_endpoints", "--fixed-endpoints", "true_flag", STRUCTURAL, False),
    ParamSpec("calligraphy", "--calligraphy", "true_flag", STRUCTURAL, False),
    ParamSpec("use_cpu", "--use-cpu", "true_flag", STRUCTURAL, False),
    ParamSpec("object_size_ratio", "--object-size-ratio", "float", STRUCTURAL, 0.75),
    ParamSpec("sampling_rate", "--sampling-rate", "int", STRUCTURAL, 5000),
    ParamSpec("caption", "--caption", "str", STRUCTURAL, ""),
    ParamSpec("conditioning_scale", "--conditioning-scale", "float", STRUCTURAL, 0.5),
    ParamSpec("condition", "--condition", "str", STRUCTURAL, "depth"),
    ParamSpec(
        "lora_model", "--lora-model", "str", STRUCTURAL, "./SLDgen/guidance/sld-lora.safetensors"
    ),
    ParamSpec("lora_weight", "--lora-weight", "float", STRUCTURAL, 0.1),
    ParamSpec("lr", "--lr", "float", STRUCTURAL, 0.8),
    ParamSpec("avoid", "--avoid", "path_list", STRUCTURAL, None),
    ParamSpec("avoidance_weight", "--avoidance-weight", "float", STRUCTURAL, 0.004),
    ParamSpec("avoidance_distance", "--avoidance-distance", "float", STRUCTURAL, 25.0),
    ParamSpec("attract", "--attract", "path_list", STRUCTURAL, None),
    ParamSpec("attraction_weight", "--attraction-weight", "float", STRUCTURAL, 0.004),
    ParamSpec("attraction_distance", "--attraction-distance", "float", STRUCTURAL, 25.0),
    #: Canny-derived attraction. Unlike `attract`, this needs no input file: the
    #: run generates its own SVG from the target once canvas space exists, and
    #: leaves it in the run directory as `attract_canny.svg`. So it is an
    #: ordinary parameter, not an input role.
    ParamSpec("attract_canny", "--attract-canny", "true_flag", STRUCTURAL, False),
    ParamSpec("attract_canny_low", "--attract-canny-low", "float", STRUCTURAL, 100.0),
    ParamSpec("attract_canny_high", "--attract-canny-high", "float", STRUCTURAL, 200.0),
    ParamSpec("attract_canny_blur", "--attract-canny-blur", "int", STRUCTURAL, 3),
    ParamSpec("attract_canny_simplify", "--attract-canny-simplify", "float", STRUCTURAL, 1.0),
    ParamSpec("attract_canny_min_length", "--attract-canny-min-length", "float", STRUCTURAL, 12.0),
    ParamSpec("attract_canny_max_points", "--attract-canny-max-points", "int", STRUCTURAL, 400),
    ParamSpec("init_points", "--init-points", "path", STRUCTURAL, None),
    ParamSpec("stipple_weight", "--stipple-weight", "path", STRUCTURAL, None),
    ParamSpec("stipple_weight_mode", "--stipple-weight-mode", "str", STRUCTURAL, "multiply"),
    ParamSpec("repulsion_loss_weight", "--repulsion-loss-weight", "float", STRUCTURAL, 0.004),
    ParamSpec("sparse_loss_weight", "--sparse-loss-weight", "float", STRUCTURAL, 2000.0),
    ParamSpec("sparse_loss_type", "--sparse-loss-type", "float", STRUCTURAL, 1.0),
    ParamSpec("sparse_loss_progressive", "--sparse-loss-progressive", "str", STRUCTURAL, "linear"),
    ParamSpec(
        "length_shortening_loss_weight",
        "--length-shortening-loss-weight",
        "float",
        STRUCTURAL,
        0.1,
    ),
    ParamSpec(
        "aesthetic_predictor_model_path",
        "--aesthetic-predictor-model-path",
        "str",
        STRUCTURAL,
        "./SLDgen/metrics/aesthetic_predictor_v2_5.pth",
    ),
    ParamSpec("save_interval", "--save-interval", "int", OPERATIONAL, 100),
    ParamSpec("checkpoint_interval", "--checkpoint-interval", "int", OPERATIONAL, 200),
    ParamSpec("save_video", "--no-video", "false_flag", OPERATIONAL, False),
    ParamSpec("verbose", "--verbose", "true_flag", OPERATIONAL, False),
    ParamSpec("debug", "--debug", "true_flag", OPERATIONAL, False),
)

SPEC_BY_NAME = {spec.name: spec for spec in PARAM_SPECS}
SPEC_BY_FLAG = {spec.flag: spec for spec in PARAM_SPECS}
PARAM_NAMES = tuple(spec.name for spec in PARAM_SPECS)
STRUCTURAL_NAMES = frozenset(s.name for s in PARAM_SPECS if s.group == STRUCTURAL)
OPERATIONAL_NAMES = frozenset(s.name for s in PARAM_SPECS if s.group == OPERATIONAL)

#: Flags the worker owns. A caller that tries to set one of these is rejected:
#: they are the mechanism by which the service segments a run, and letting a job
#: pin them would silently break resume.
RUNTIME_FLAGS = frozenset(
    {"--target", "--output-dir", "--experiment-name", "--stop-at", "--resume"}
)

PATH_KINDS = frozenset({"path", "path_list"})


def _coerce(spec, value):
    if value is None:
        return None
    try:
        if spec.kind == "int":
            return int(value)
        if spec.kind == "float":
            return float(value)
        if spec.kind in ("str", "path"):
            return str(value)
        if spec.kind in ("true_flag", "false_flag"):
            return bool(value)
        if spec.kind == "float_or_str":
            try:
                return float(value)
            except (TypeError, ValueError):
                return str(value)
        if spec.kind == "float_pair":
            pair = [float(v) for v in value]
            if len(pair) != 2:
                raise ParamError(f"{spec.name} takes exactly two numbers, got {value!r}")
            return pair
        if spec.kind == "path_list":
            if isinstance(value, (str, Path)):
                value = [value]
            return [str(v) for v in value]
    except ParamError:
        raise
    except (TypeError, ValueError) as exc:
        raise ParamError(f"{spec.name}: cannot interpret {value!r} as {spec.kind} ({exc})") from exc
    raise ParamError(f"unknown parameter kind {spec.kind!r}")


def canonical_params(partial=None):
    """Fill in every default and coerce every value; reject unknown names."""
    partial = dict(partial or {})
    unknown = sorted(set(partial) - set(PARAM_NAMES))
    if unknown:
        raise ParamError(
            f"unknown parameter(s): {', '.join(unknown)}. "
            f"Known parameters: {', '.join(sorted(PARAM_NAMES))}"
        )
    return {
        spec.name: _coerce(spec, partial.get(spec.name, spec.default)) for spec in PARAM_SPECS
    }


def validate_params(params):
    """Reject at submit time what SLDgen would reject at parse time.

    Failing a job before it is queued costs nothing; failing it after the worker
    has claimed it costs a scheduling slot and produces a failed job the user has
    to clean up.
    """
    params = canonical_params(params)

    if params["num_iter"] <= 0:
        raise ParamError("num_iter must be > 0")
    if params["render_size"] <= 0:
        raise ParamError("render_size must be > 0")
    if params["save_interval"] <= 0:
        raise ParamError("save_interval must be > 0")
    if params["checkpoint_interval"] < 0:
        raise ParamError("checkpoint_interval must be >= 0")
    if params["condition"] not in ("depth", "canny"):
        raise ParamError("condition must be 'depth' or 'canny'")
    if params["stipple_weight_mode"] not in ("multiply", "replace"):
        raise ParamError("stipple_weight_mode must be 'multiply' or 'replace'")
    if params["init_method"] not in ("tsp", "trefoil", "contour"):
        raise ParamError("init_method must be 'tsp', 'trefoil' or 'contour'")

    width = params["width"]
    if isinstance(width, str) and width not in ("random", "optim", "optim_random"):
        raise ParamError("width must be a number or one of 'random', 'optim', 'optim_random'")

    if params["origin"] is not None:
        if params["fixed_endpoints"]:
            raise ParamError("origin and fixed_endpoints are mutually exclusive")
        if params["init_method"] != "tsp":
            raise ParamError("origin is only supported with init_method 'tsp'")
        if not all(0.0 <= value <= 1.0 for value in params["origin"]):
            raise ParamError("origin values must be in [0, 1]")

    for name in ("init_points", "stipple_weight"):
        if params[name] is not None and params["init_method"] != "tsp":
            raise ParamError(f"{name} is only supported with init_method 'tsp'")

    # Mirrors SLDgen/config.py. Checked here too so the job is refused at
    # submission rather than by a worker that has already claimed a slot.
    if params["attract_canny"]:
        if params["attract_canny_low"] >= params["attract_canny_high"]:
            raise ParamError("attract_canny_low must be below attract_canny_high")
        if params["attract_canny_max_points"] < 2:
            raise ParamError("attract_canny_max_points must be at least 2")
        if params["attract_canny_blur"] < 0:
            raise ParamError("attract_canny_blur must be 0 (disabled) or positive")

    return params


def split_by_group(params):
    """(structural, operational) views of a parameter set."""
    params = canonical_params(params)
    return (
        {name: params[name] for name in PARAM_NAMES if name in STRUCTURAL_NAMES},
        {name: params[name] for name in PARAM_NAMES if name in OPERATIONAL_NAMES},
    )


def structural_differences(old, new):
    """Names of structural parameters that differ -- what makes PATCH a 409."""
    old_structural, _ = split_by_group(old)
    new_structural, _ = split_by_group(new)
    return sorted(
        name for name in old_structural if old_structural[name] != new_structural[name]
    )


def _format(value):
    """Render a scalar for argv. ``repr``-free so floats stay human-readable."""
    if isinstance(value, bool):
        raise AssertionError("booleans are emitted as flags, not values")
    return str(value)


def params_to_argv(params, root=None):
    """Parameter flags only -- no --target, --output-dir, --stop-at or --resume.

    ``root`` resolves stored (root-relative) input paths to absolute ones for the
    child process. Leaving it None keeps paths verbatim, which is what the
    round-trip contract is stated in terms of.
    """
    params = canonical_params(params)
    argv = []
    for spec in PARAM_SPECS:
        value = params[spec.name]
        if value is None:
            continue
        if spec.kind == "true_flag":
            if value:
                argv.append(spec.flag)
        elif spec.kind == "false_flag":
            # The CLI spells these as negations (--no-optimize-cp-weights), so
            # the flag appears exactly when the value is False.
            if not value:
                argv.append(spec.flag)
        elif spec.kind == "float_pair":
            argv += [spec.flag, _format(value[0]), _format(value[1])]
        elif spec.kind == "path_list":
            if value:
                argv += [spec.flag] + [_resolve(path, root) for path in value]
        elif spec.kind == "path":
            argv += [spec.flag, _resolve(value, root)]
        else:
            argv += [spec.flag, _format(value)]
    return argv


def _resolve(path, root):
    if root is None:
        return str(path)
    path = Path(path)
    return str(path if path.is_absolute() else Path(root) / path)


def argv_to_params(argv, root=None):
    """Inverse of :func:`params_to_argv`; runtime flags are ignored, not an error.

    Ignoring them means a full recorded argv (which includes --target and friends)
    can be fed back in to recover the parameters that produced it.
    """
    values = {}
    index = 0
    argv = list(argv)
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in RUNTIME_FLAGS:
            index += 2 if token != "--resume" or index + 1 < len(argv) else 1
            continue
        spec = SPEC_BY_FLAG.get(token)
        if spec is None:
            index += 1
            continue
        if spec.kind == "true_flag":
            values[spec.name] = True
            index += 1
        elif spec.kind == "false_flag":
            values[spec.name] = False
            index += 1
        elif spec.kind == "float_pair":
            values[spec.name] = [float(argv[index + 1]), float(argv[index + 2])]
            index += 3
        elif spec.kind == "path_list":
            collected = []
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                collected.append(_relativize(argv[index], root))
                index += 1
            values[spec.name] = collected
        elif spec.kind == "path":
            values[spec.name] = _relativize(argv[index + 1], root)
            index += 2
        else:
            values[spec.name] = argv[index + 1]
            index += 2
    return canonical_params(values)


def _relativize(path, root):
    if root is None:
        return str(path)
    try:
        return str(Path(path).relative_to(Path(root)))
    except ValueError:
        return str(path)


def runtime_argv(target, output_dir, stop_at, resume=None, experiment_name="run"):
    """The flags the worker owns: what to draw, where to put it, where to stop."""
    argv = [
        "--target",
        str(target),
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        experiment_name,
        "--stop-at",
        str(int(stop_at)),
    ]
    if resume is not None:
        argv += ["--resume", str(resume)]
    return argv


def build_argv(python, script, params, target, output_dir, stop_at, resume=None, root=None):
    """The exact command a segment runs. Recorded verbatim in ``segments.argv_json``."""
    return [str(python), str(script)] + runtime_argv(
        target, output_dir, stop_at, resume
    ) + params_to_argv(params, root=root)
