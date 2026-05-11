<div align="center">

<h1>SVOO</h1>

<h2>Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering</h2>

<h3>🎉 Accepted to <strong>ICML 2026 Main Track</strong></h3>

<p><strong>Jiayi Luo, Jiayu Chen, Jiankun Wang, Cong Wang, Hanxin Zhu, Qingyun Sun, Chen Gao, Zhibo Chen, Jianxin Li</strong></p>

<p>
  <a href="https://arxiv.org/pdf/2603.18636"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2603.18636-b31b1b.svg"></a>
  <a href="https://icml.cc/"><img alt="Conference" src="https://img.shields.io/badge/ICML-2026%20Main%20Track-4c6ef5.svg"></a>
</p>

</div>

SVOO is a training-free sparse attention method for video generation with offline sparsity profiles and online QK co-clustering.

## Installation

Prerequisites: Linux, Conda, Git, and an NVIDIA GPU with CUDA support.

```bash
git clone https://github.com/Mutual-Luo/SVOO.git
cd svoo
bash scripts/build_env.sh
conda activate svoo
```

`scripts/build_env.sh` creates or reuses a Conda environment, installs CUDA/PyTorch build dependencies, initializes submodules, installs SVOO, installs FlashAttention and FlashInfer, and builds the local CUDA extension. CUDA compute capabilities are detected from the visible NVIDIA GPUs by default.

| Override | Default | Description |
| --- | --- | --- |
| `CONDA_ENV_NAME` | `svoo` | Conda environment name |
| `PYTHON_VERSION` | `3.12.9` | Python version |
| `CUDA_VERSION` | `12.6.0` | Conda CUDA toolkit version |
| `MAX_JOBS` | `nproc` | Parallel build jobs |
| `TORCH_CUDA_ARCH_LIST` | Auto-detected | Override CUDA architectures for cross-build hosts |

FlashAttention, FlashInfer, and the SVOO CUDA extension are required runtime components and are always installed or built by the setup script.

## Model Weights

Choose one directory to store model weights, then reuse it for all commands:

```bash
export MODEL_ROOT=/path/to/models
```

Download all supported public models:

```bash
MODEL_ROOT=/path/to/models bash scripts/download_models.sh
```

Inference scripts look for model folders under `MODEL_ROOT`. If a model is stored somewhere else, pass the exact directory with `MODEL_PATH=/path/to/model`.

## Offline Sparsity Profiles

Canonical profiles are already included. To regenerate them:

```bash
GPUS=0 bash scripts/offline/generate_sparsity_profiles.sh wan21_t2v_14b
GPUS="0 1 2 3" bash scripts/offline/generate_sparsity_profiles.sh all
```

Profiling prompts live in `data/profile_data/prompt.txt`. See `scripts/offline/README.md` for profiling options and output layout.

## Inference

| Task | Command |
| --- | --- |
| Wan T2V | `MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=1.3B bash scripts/inference/wan/wan_t2v_720p_svoo.sh` |
| Wan I2V | `MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh` |
| Wan2.2 T2V A14B | `MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=A14B bash scripts/inference/wan/wan_t2v_720p_svoo.sh` |
| Wan2.2 I2V A14B | `MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=A14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh` |
| HunyuanVideo T2V | `MODEL_ROOT=/path/to/models GPUS=0 bash scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh` |
| HunyuanVideo I2V | `MODEL_ROOT=/path/to/models GPUS=0 bash scripts/inference/hunyuan10/hunyuan10_i2v_720p_svoo.sh` |

To run on a specific GPU, use either `GPUS=7` or `CUDA_VISIBLE_DEVICES=7`.

Demo inputs use this layout:

```text
data/example/<id>/prompt.txt
data/example/<id>/image.jpg
```

Outputs are written to `result/` unless `OUTPUT_DIR` or `OUTPUT_FILE` is set.

### Common Options

| Variable | Description |
| --- | --- |
| `MODEL_ROOT=/path/to/models` | Parent directory containing model folders |
| `MODEL_PATH=/path/to/model` | Exact model directory; overrides `MODEL_ROOT` lookup |
| `GPUS=0` | GPU id used by the launch script |
| `PROMPT_ID=1` | Use `data/example/1/` |
| `PROMPT_FILE=/path/to/prompt.txt` | Override the prompt file |
| `IMAGE_FILE=/path/to/image.jpg` | Override the I2V input image |
| `OUTPUT_DIR=/path/to/results` | Override output directory |
| `OUTPUT_FILE=/path/to/video.mp4` | Override exact output file |
| `SEED=0` | Generation seed |
| `DRY_RUN=1` | Print the command without running inference |

### Memory And Timing

| Variable | Default | Description |
| --- | --- | --- |
| `CPU_OFFLOAD` | `0` | Set `1` to reduce GPU memory usage with CPU offload; this can be slower |
| `SVOO_TRITON_TUNE` | `fixed` | Set `auto` to search the fastest Triton config for the current GPU |

## Acknowledgements

We thank the authors of [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) for their excellent open-source project and inspiring work on training-free sparse attention for video generation, including **SVG1** and **SVG2**.
