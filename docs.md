# Command-Line Parameters

This document summarizes the parameters used when running the `sldgen.py` script. The CLI is organized into sections below, with the most important parameters highlighted first.


**Notes:**
- Output folders are created automatically under `<output-dir>/<target-stem>/<experiment-name>/`.
- When `--experiment-name` is omitted, a timestamp is used.
- When CUDA is unavailable, the code automatically falls back to CPU.
- The target image must exist before running.

**Most Important Parameters**

These are the parameters you are most likely to tune when running the pipeline.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | `str` | required | Path to the image used as the target for single-line drawing generation. |
| `--caption` | `str` | `""` | Text prompt used for SDS guidance. |
| `--experiment-name` | `str` | generated automatically | Name of the experiment folder. If not provided, a timestamp is used. |
| `--num-iter` | `int` | `4000` | Number of optimization iterations. |
| `--width` | `float` or `str` | `1.0` | Stroke width, or the `optim` special mode for varying line width. |
| `--fixed-endpoints` | flag | `False` | Keep endpoints fixed during optimization to make it easier to connect drawings. |
| `--n-control-points` | `int` | `385` | Number of control points used to initialize the line. |
| `--save-interval` | `int` | `100` | Frequency for saving intermediate results. |

**Special cases**

To generate the particular configurations of single-line drawings presented in the paper, use one of the following commands:

*Single-line drawing with varying width*
```bash
python sldgen.py --target ./data/firefighter.png --width optim
```

*Single-line drawing with fixed endpoints for easier connection*
```bash
python sldgen.py --target ./data/firefighter.png --fixed-endpoints
```

## All parameters
### General Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--output-dir` | `str` | `./output/` | Directory where results are saved. |
| `--experiment-name` | `str` | generated automatically | Name of the experiment folder. If not provided, a timestamp is used. |
| `--use-cpu` | flag | `False` | Force CPU execution even if CUDA is available. |
| `--seed` | `int` | `0` | Random seed for Python, NumPy, and PyTorch. |
| `--verbose` | flag | `False` | Print loss values during optimization. |
| `--debug` | flag | `False` | Enable debug mode. Print additional debug information. |

### Target Image Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | `str` | required | Target image path. The file must exist. |
| `--object-size-ratio` | `float` | `0.75` | Maximum size of the object relative to the render size, used to rescale the target. |
| `--render-size` | `int` | `512` | Output render size in pixels. |
| `--calligraphy` | flag | `False` | Treat the target as a calligraphy image or letter. Changes how the mask is created. |

### Optimization Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--num-iter` | `int` | `4000` | Total number of optimization iterations. |
| `--lr` | `float` | `0.8` | Base learning rate for the optimizer. |
| `--save-interval` | `int` | `100` | Save intermediate outputs every N iterations. |

### Curve Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--sampling-rate` | `int` | `5000` | Number of points sampled from the curve for rendering. |
| `--no-optimize-cp-weights` | flag | `optimize_cp_weights = True` | Disable optimization of B-spline control-point weights. |
| `--keep-low-weights` | flag | `prune_low_weights = True` | Keep low weights instead of pruning them. |
| `--init-method` | `str` | `tsp` | Initialization method for the single line: `trefoil`, `contour`, or `tsp`. |
| `--n-control-points` | `int` | `385` | Number of control points at the start of optimization. |
| `--width` | `float` or `str` | `1.0` | Stroke width, or the `optim` special mode for varying line width. |
| `--fixed-endpoints` | flag | `False` | Keep endpoints fixed during optimization for easier connection between drawings. |

### SDS Guidance Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--caption` | `str` | `""` | Text prompt used for semantic guidance. |
| `--conditioning-scale` | `float` | `0.5` | Conditioning scale used in the SDS guidance objective. |
| `--condition` | `str` | `depth` | Guidance conditioning type, either `depth` or `canny`. |
| `--lora-model` | `str` | `./SLDgen/guidance/sld-lora.safetensors` | Path to the LoRA weights used for guidance. |
| `--lora-weight` | `float` | `0.1` | Strength of the LoRA adapter. Set to `0` to disable its effect. |

### Loss Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--repulsion-loss-weight` | `float` | `0.004` | Weight of the repulsion loss. |
| `--sparse-loss-weight` | `float` | `2000.0` | Weight of the sparsity loss. |
| `--sparse-loss-type` | `float` | `1.0` | Degree of the sparsity loss; `0.0` corresponds to a standard deviation loss. |
| `--sparse-loss-progressive` | `str` | `linear` | Progressive sparse-loss mode. Any value other than `linear` disables the progressive schedule. |
| `--length-shortening-loss-weight` | `float` | `0.1` | Weight of the length-shortening loss. |

### Metrics Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--aesthetic-predictor-model-path` | `str` | `./SLDgen/metrics/aesthetic_predictor_v2_5.pth` | Path to the aesthetic predictor model weights. |

### Fixed Runtime Settings

These values are set internally by [SLDgen/config.py](SLDgen/config.py) after argument parsing and are not exposed as CLI flags.

| Setting | Value | Description |
| --- | --- | --- |
| `diffusion_model` | `stabilityai/stable-diffusion-3.5-medium` | Diffusion model used for guidance. |
| `diffusion_timesteps` | `1000` | Number of diffusion timesteps. |
| `diffusion_guidance_scale` | `100` | Guidance scale used for diffusion sampling. |
| `vae_path` | `madebyollin/taesd3` | VAE used by the guidance pipeline. |
| `multisteps` | `1` | Number of multistep passes. |
| `negative_caption` | `""` | Negative prompt passed to the guidance model. |


