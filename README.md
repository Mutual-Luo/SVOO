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

Download all supported public models:

```bash
MODEL_ROOT=/path/to/models bash scripts/download_models.sh
```

Inference scripts first look for local weights under `MODEL_ROOT`. A single model can be passed directly with `MODEL_PATH`.

```bash
MODEL_PATH=/path/to/model GPUS=0 bash scripts/inference/wan/wan_t2v_720p_svoo.sh
```

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
| Wan T2V | `GPUS=0 MODEL_SIZE=1.3B bash scripts/inference/wan/wan_t2v_720p_svoo.sh` |
| Wan I2V | `GPUS=0 MODEL_SIZE=14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh` |
| Wan2.2 T2V A14B | `GPUS=0 MODEL_SIZE=A14B bash scripts/inference/wan/wan_t2v_720p_svoo.sh` |
| Wan2.2 I2V A14B | `GPUS=0 MODEL_SIZE=A14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh` |
| HunyuanVideo T2V | `GPUS=0 bash scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh` |
| HunyuanVideo I2V | `GPUS=0 bash scripts/inference/hunyuan10/hunyuan10_i2v_720p_svoo.sh` |

Demo inputs use this layout:

```text
data/example/<id>/prompt.txt
data/example/<id>/image.jpg
```

Outputs are written to `result/` unless `OUTPUT_DIR` or `OUTPUT_FILE` is set.

### Common Options

| Variable | Description |
| --- | --- |
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
| `SVOO_ENABLE_MEM_SAVE` | `1` | Reduces GPU memory usage by releasing large SVOO intermediates earlier |
| `SVOO_TRITON_WARMUP` | `1` | Required kernel warmup before the progress bar |
| `SVOO_TRITON_TUNE` | `fixed` | Set `auto` to search the fastest Triton config for the current GPU |
| `SVOO_CACHE_ROOT` | `.triton_cache` | Compiler and FlashInfer cache root |

Warmup preserves RNG state and is designed not to affect generated videos. Compilation happens before the inference progress bar.

## Notes

Generated videos, model downloads, compiler caches, and kernel builds are ignored by Git.
