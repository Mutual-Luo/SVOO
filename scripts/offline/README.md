# Offline Sparsity Profiling

Generate the canonical `sparsity_profiles/sparsity_*.csv` files used by SVOO inference.

## Quick Start

```bash
GPUS=0 bash scripts/offline/generate_sparsity_profiles.sh wan21_t2v_14b
GPUS="0 1 2 3" bash scripts/offline/generate_sparsity_profiles.sh all
```

One GPU runs jobs serially. Multiple GPU ids run different model jobs in parallel.

Model keys:

```text
wan21_t2v_1_3b  wan21_t2v_14b  wan21_i2v_14b
wan22_t2v_a14b  wan22_i2v_a14b
hunyuan10_t2v_13b hunyuan10_i2v_13b
```

`hunyuan10_*` means HunyuanVideo 1.0, not HunyuanVideo 1.5.

## Profile Data

```bash
PROFILE_PROMPT_FILE=data/profile_data/prompt.txt
PROFILE_IMAGE_ROOT=data/profile_data/image
```

`prompt.txt` is one prompt per line, and every non-empty line is profiled.
For i2v models, line `N` uses `image/N.jpg`; `.jpeg`, `.png`, and `image/N/image.*` are also accepted.

## Exact vs Fast

By default profiling is exact:

```bash
SPARSITY_QUERY_SAMPLES=0
```

This computes attention sparsity using all query rows, so it is accurate but slow.
To speed it up, sample query rows:

```bash
SPARSITY_QUERY_SAMPLES=4096 GPUS=0 bash scripts/offline/generate_sparsity_profiles.sh wan21_t2v_14b
```

Sampling is only for faster offline estimation. It uses uniformly spaced query rows, then computes the same QK softmax and cumulative-mass metric on that subset.

## Common Options

```bash
MODEL_ID=/path/or/hf-id
MODEL_ROOT=/home/dataset-assist-0/luojy/models  # looks for MODEL_ROOT/<HF repo name>
OFFLINE_PROFILE_ROOT=sparsity_profiles
SPARSITY_THRESHOLD=0.95
SPARSITY_BATCH_SIZE=0      # memory chunk size; 0 auto-selects
RESUME_PROFILE=1           # continue from existing raw logs
RUN_INFERENCE=0            # merge existing raw logs only
```

## Outputs

```text
sparsity_profiles/sparsity_*.csv
sparsity_profiles/runs/<model_key>/runner.log
sparsity_profiles/runs/<model_key>/raw/attention_sparsity-*-th*-q*.txt
sparsity_profiles/runs/<model_key>/logs/*.jsonl
sparsity_profiles/runs/<model_key>/videos/*.mp4
```

The root CSV files are the profiles used by inference. The `runs/` directory is profiling output and can be deleted after the CSVs are produced.

## Metric

For each `(step, layer, head)`, profiling measures the minimum key-token ratio needed to cover `SPARSITY_THRESHOLD` cumulative attention mass. With multiple prompts, the final CSV keeps the maximum value for each `(step, layer, head)` so the profile stays conservative.
