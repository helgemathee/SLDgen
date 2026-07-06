# SLDgen on RTX 50-series (Blackwell / sm_120) with CUDA 13

This is a fork of [tanguymagne/SLDgen](https://github.com/tanguymagne/SLDgen)
set up to build and run on an **NVIDIA RTX 5090** (compute capability
`sm_120`, Blackwell). SLDgen's own Python is device-agnostic and unchanged —
the porting work lives in the native/CUDA dependencies, which are maintained
in dedicated forks. This file is the rebuild recipe.

> **Why not upstream's instructions?** The README pins CUDA 12.4 + torch
> 2.3.1/cu121, and README step 5 installs `torch==2.3.1 cu121`. **None of
> that supports sm_120** — skip it. sm_120 needs CUDA ≥ 12.8; we use CUDA 13.0
> to match the env's torch (2.12.1+cu130, which already lists sm_120 in
> `torch.cuda.get_arch_list()`).

## Forks (the sources to pull from)

| Component | Fork (branch `sm120-cuda13`) | Upstream |
|-----------|------------------------------|----------|
| SLDgen (this repo) | `git@github.com:helgemathee/SLDgen.git` | tanguymagne/SLDgen |
| diffvg | `git@github.com:helgemathee/diffvg.git` | BachiLi/diffvg |
| pybind11 (diffvg submodule) | `git@github.com:helgemathee/pybind11.git` | pybind/pybind11 |
| fab3dwire (wiregrad) | `git@github.com:helgemathee/fab3dwire.git` | kenji-tojo/fab3dwire |
| Concorde (TSP) | not forked — build from tarball, see below | Georgia Tech |

Each dep fork has an `SM120_CUDA13.md` describing exactly what was changed.

## Environment (conda: `sldgen`)

```bash
conda create -n sldgen python=3.11 && conda activate sldgen
# torch with cu130 (already provides sm_120):  torch==2.12.1+cu130
conda install -y -c conda-forge \
  cuda-toolkit=13.0.2 \   # nvcc 13.0.88, knows compute_120
  "cmake<4" \             # diffvg uses deprecated FindCUDA that cmake 4 removed
  "eigen=3.4" \           # wiregrad rejects eigen 5.x
  ffmpeg
# persisted env vars:
conda env config vars set CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=12.0 \
  CONCORDE_PATH=/home/helge/src/concorde/TSP/concorde
```

Host gcc 15 works with nvcc 13 for sm_120. **For every native build below**, put
nvcc's `cicc` on PATH first, or the build fails with `cicc: not found`:
```bash
export PATH=$CONDA_PREFIX/nvvm/bin:$PATH
```

## Build the native deps

### diffvg  → `import pydiffvg`
```bash
git clone --recursive git@github.com:helgemathee/diffvg.git ~/src/diffvg
cd ~/src/diffvg
export PATH=$CONDA_PREFIX/nvvm/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
rm -rf build && python setup.py install      # NOT pip (poetry backend breaks pip)
pip install cssutils svgpathtools            # pure-python deps missing at import
# verify: cuobjdump build/lib*/diffvg*.so | grep arch  ->  arch = sm_120
```

### wiregrad  → `import wiregrad as wg`  (used in SLDgen/run.py)
```bash
git clone git@github.com:helgemathee/fab3dwire.git ~/src/fab3dwire
cd ~/src/fab3dwire/wiregrad
export PATH=$CONDA_PREFIX/nvvm/bin:$PATH CUDAARCHS=120
pip install "nanobind==2.0.0" "scikit-build>=0.14.0" ninja   # NEVER requirements.txt (pins old torch/triton)
rm -rf _skbuild build dist *.egg-info        # stale _skbuild -> generator mismatch
pip install . --no-build-isolation
pip install cholespy                         # pure-python dep missing at import
```

### Concorde (TSP solver — native C, NOT CUDA)
Not forked; build from the upstream tarball. 2003-era C needs modern-gcc flags.
```bash
mkdir -p ~/src/concorde && cd ~/src/concorde
# place co031219.tgz + qsopt.a + qsopt.h (ubuntu build) here, extract co031219.tgz
export CFLAGS="-O2 -fcommon -std=gnu89 -Wno-implicit-function-declaration -Wno-implicit-int -Wno-return-mismatch"
./configure --host=x86_64-pc-linux-gnu \
  --with-qsopt=/home/helge/src/concorde --enable-ccdefaults   # host+qsopt path must be explicit/absolute
make
# binary: ~/src/concorde/TSP/concorde  (this is $CONCORDE_PATH)
```

## SLDgen Python deps
```bash
cd ~/SLDgen
pip install -r requirements.txt easydict     # easydict is imported but missing from requirements.txt
pip install "transformers>=4.54.1,<5"        # transformers 5.x breaks diffusers 0.34 (FLAX_WEIGHTS_NAME)
# keep numpy <2 (1.26.4) and torch 2.12.1+cu130. Ignore README's torch==2.3.1 cu121 step.
```

## Run
Needs HuggingFace auth + approved gated access to
`stabilityai/stable-diffusion-3.5-medium` (`hf auth login`, request access on
HF). Needs a 24 GB+ GPU (5090 has 32 GB).
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CONCORDE_PATH is persisted in the env vars; nvvm/PATH only matters for rebuilds.
python sldgen.py --target ./data/firefighter.png --num-iter 30    # quick smoke test (~5 it/s)
python sldgen.py --target ./data/firefighter.png                  # full quality (default 4000 iters)
```

## Gotchas cheat-sheet
- `cicc: not found` → `export PATH=$CONDA_PREFIX/nvvm/bin:$PATH` before building.
- diffvg pybind11 "ambiguous template instantiation for factory" → fixed in the
  pybind11 fork (`detail/init.h`); make sure the submodule points at
  `helgemathee/pybind11`.
- wiregrad "generator Ninja does not match Unix Makefiles" → `rm -rf _skbuild`.
- `kaleido` 1.x needs system Chrome → `pip install "kaleido==0.2.1"` (bundles chromium).
- Out-of-memory that isn't SLDgen's fault → check for other GPU hogs (e.g. a
  running ComfyUI); SLDgen itself fits on the 32 GB card.
