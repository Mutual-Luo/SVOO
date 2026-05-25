# SVOO-EAR

SVOO-EAR is a private integration branch that combines the **SVOO** sparse video-generation inference framework with the **EAR** error-aware routing mechanism from **SVG-EAR**. The goal of this branch is to support side-by-side evaluation of dense attention, SVOO sparse attention, and EAR-enhanced sparse attention on the Wan2.2-I2V-A14B inference path.

> This repository is derived from [SVOO](https://github.com/Mutual-Luo/SVOO) and incorporates EAR-related mechanisms adapted from [SVG-EAR](https://github.com/dyxg/SVG-EAR/tree/pr/ear-wan22-support). SVG-EAR is licensed under the Apache License, Version 2.0. Files modified for EAR integration carry an explicit SVG-EAR license reference in their headers.

## Core Advantages

SVOO-EAR keeps the existing SVOO pipeline intact while adding an EAR mode for Wan image-to-video inference. SVOO contributes offline sparsity profiling, online QK co-clustering, and FlashInfer-backed dynamic block sparse attention. SVG-EAR contributes error-aware block selection and centroid compensation, which are intended to reduce quality loss when attention blocks are pruned aggressively.

| Component | Contribution in SVOO-EAR | Benefit |
| --- | --- | --- |
| SVOO offline profile | Reuses layer/head sparsity profiles and dynamic `min_kc_ratio` selection. | Keeps the original SVOO acceleration strategy and profiling workflow. |
| SVOO online co-clustering | Preserves semantic-aware Q/K permutation and cluster-level block map construction. | Maintains the token grouping structure required by the existing sparse backend. |
| SVG-EAR error-aware routing | Adds EAR block selection with value-statistics-based error estimation. | Prioritizes sparse blocks that are expected to have larger approximation error. |
| SVG-EAR centroid compensation | Adds a pruned sparse forward path that compensates omitted blocks with cluster centroids. | Provides a quality-oriented alternative to the original SVOO sparse path. |
| Wan2.2-I2V script support | Adds `PATTERN=dense|svoo|ear` and `EAR_GAMMA` to the 720p I2V script. | Enables direct quality and performance comparison using one script. |

## Main Runtime Modes

The primary comparison entry point is `scripts/inference/wan/wan_i2v_720p_svoo.sh`. Set `PATTERN` to choose the attention path. The `ear` mode additionally accepts `EAR_GAMMA`, which controls the EAR error-estimation trade-off coefficient exposed by the Wan I2V inference script.

```bash
# Dense baseline
MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=A14B PATTERN=dense bash scripts/inference/wan/wan_i2v_720p_svoo.sh

# Original SVOO sparse path
MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=A14B PATTERN=svoo bash scripts/inference/wan/wan_i2v_720p_svoo.sh

# EAR-enhanced sparse path
MODEL_ROOT=/path/to/models GPUS=0 MODEL_SIZE=A14B PATTERN=ear EAR_GAMMA=1.0 bash scripts/inference/wan/wan_i2v_720p_svoo.sh
```

The script writes outputs into mode-specific result directories by default, making it easier to compare generated video quality across `dense`, `svoo`, and `ear` runs under the same prompt, seed, and model configuration.

## Modified Areas

The Wan2.2-I2V EAR integration primarily touches the attention selector, sparse attention backend, Wan attention processor registration, the top-level Wan I2V CLI, and the 720p inference script.

| File | Purpose |
| --- | --- |
| `svoo/co_clustering.py` | Adds EAR dynamic block selection and EAR pruned sparse FlashInfer forward support. |
| `svoo/models/wan/attention.py` | Adds the Wan EAR attention processor and dispatches `pattern=ear` to EAR selector/backend. |
| `svoo/models/wan/inference.py` | Registers EAR processors in the Wan replacement path and includes EAR in warmup conditions. |
| `wan_i2v_inference.py` | Exposes `--pattern ear` and `--ear_gamma` from the Wan I2V command line. |
| `scripts/inference/wan/wan_i2v_720p_svoo.sh` | Adds one-script comparison support for dense, SVOO, and EAR modes. |

## Validation Status

Before submission, the integration branch was checked with Python syntax compilation, shell syntax validation for the Wan I2V script, patch whitespace validation, and patch applicability validation. Full GPU video-generation validation still needs to be run in the target CUDA/PyTorch environment with the required Wan model weights, FlashInfer backend, and SVOO runtime dependencies.

## Attribution and License Notes

SVOO-EAR is an integration work based on two upstream open-source projects:

| Upstream project | URL | Role in this branch |
| --- | --- | --- |
| SVOO | https://github.com/Mutual-Luo/SVOO | Base repository and sparse video-generation inference framework. |
| SVG-EAR | https://github.com/dyxg/SVG-EAR/tree/pr/ear-wan22-support | Source of the EAR mechanism and related error-aware sparse attention ideas. |

SVG-EAR is distributed under the Apache License, Version 2.0. This branch adds explicit license-reference comments to the modified source files that contain or expose the EAR integration. Users should review the upstream licenses before redistributing this combined work.
