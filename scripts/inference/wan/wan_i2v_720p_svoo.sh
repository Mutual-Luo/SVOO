#!/usr/bin/env bash
set -euo pipefail

# Example:
#   MODEL_ROOT=~/models GPUS=7 RUN_ID=1 MODEL_SIZE=A14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh --top_p_kmeans 1.0 # Dense
#   MODEL_ROOT=~/models GPUS=0 RUN_ID=0 MODEL_SIZE=A14B bash scripts/inference/wan/wan_i2v_720p_svoo.sh                    # Sparse SVOO
#
# User-facing overrides:
#   MODEL_SIZE=14B|A14B        Select Wan2.1 14B or Wan2.2 A14B.
#   MODEL_ROOT=/path/to/models Look for local Hugging Face directories here.
#   MODEL_PATH=/path/to/model  Override the selected model path directly.
#   PROMPT_ID=1                Use data/example/${PROMPT_ID}/prompt.txt and image.jpg.
#   PROMPT_FILE=/path/file     Override the prompt file.
#   IMAGE_FILE=/path/image.jpg Override the input image. IMAGE_PATH also works.
#   OUTPUT_DIR=/path/dir       Override the result directory.
#   OUTPUT_FILE=/path/file.mp4 Override the exact output video path.
#   SVOO_CACHE_ROOT=/path/dir  Override compiler cache root.
#   TRITON_CACHE_DIR=/path/dir Override Triton JIT cache.
#   SVOO_TRITON_WARMUP=1       Compile SVOO Triton kernels before the progress bar.
#   SVOO_TRITON_TUNE=auto      Search the fastest Triton config for the current GPU.
#   SVOO_ENABLE_MEM_SAVE=0|1   Reduce GPU memory usage by releasing large SVOO intermediates earlier.
#   DRY_RUN=1                  Print the command without running inference.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${root}"

python_bin="${PYTHON:-python}"
if [ -z "${CUDA_HOME:-}" ]; then
  python_path="$(command -v "${python_bin}" 2>/dev/null || true)"
  if [ -n "${python_path}" ]; then
    python_prefix="$(cd "$(dirname "${python_path}")/.." && pwd)"
    if [ -x "${python_prefix}/bin/nvcc" ]; then
      export CUDA_HOME="${python_prefix}"
      export CUDA_PATH="${CUDA_HOME}"
    fi
  fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  if [ -z "${FLASHINFER_EXTRA_LDFLAGS:-}" ]; then
    export FLASHINFER_EXTRA_LDFLAGS="-L${CUDA_HOME}/lib -L${CUDA_HOME}/targets/x86_64-linux/lib -L${CUDA_HOME}/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs"
  fi
fi

model_root="${MODEL_ROOT:-${root}/../../models}"
model_size="${MODEL_SIZE:-14B}"
prompt_id="${PROMPT_ID:-1}"
prompt_file="${PROMPT_FILE:-data/example/${prompt_id}/prompt.txt}"
image_file="${IMAGE_FILE:-${IMAGE_PATH:-data/example/${prompt_id}/image.jpg}}"
gpu_id="${GPU:-${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}}"
gpu_id="${gpu_id%%[ ,]*}"
default_cpu_offload=0
default_mem_save=1

cache_root="${SVOO_CACHE_ROOT:-${root}/.triton_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${cache_root}/flashinfer}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

case "${model_size}" in
  14B|14b|wan21_14b|wan21_i2v_14b)
    model_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    model_dir="Wan2.1-I2V-14B-720P-Diffusers"
    output_dir="${OUTPUT_DIR:-result/wan2.1-14B/i2v/svoo}"
    sparsity_csv="sparsity_profiles/sparsity_wan_14B_i2v.csv"

    # SVOO config. min_kc_ratio is the fallback; dynamic_min_kc_ratio_* clips CSV values.
    num_inference_steps=40
    first_times_fp=0.2
    first_layers_fp=0.03
    num_q_centroids=256
    num_k_centroids=1024
    top_p_kmeans=0.90
    min_kc_ratio=0.10
    kmeans_iter_init=2
    kmeans_iter_step=2
    start_reuse_step=9
    reuse_interval=20
    dynamic_min_kc_ratio_min=0.10
    dynamic_min_kc_ratio_max=0.15
    ;;
  A14B|a14b|wan22|wan22_i2v_a14b)
    model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    model_dir="Wan2.2-I2V-A14B-Diffusers"
    output_dir="${OUTPUT_DIR:-result/wan2.2-14B/i2v/svoo}"
    sparsity_csv="sparsity_profiles/sparsity_wan22_A14B_i2v.csv"
    default_mem_save=1

    # SVOO config. min_kc_ratio is the fallback; dynamic_min_kc_ratio_* clips CSV values.
    num_inference_steps=40
    first_times_fp=0.2
    first_layers_fp=0.03
    num_q_centroids=256
    num_k_centroids=1024
    top_p_kmeans=0.90
    min_kc_ratio=0.10
    kmeans_iter_init=2
    kmeans_iter_step=2
    start_reuse_step=9
    reuse_interval=40
    dynamic_min_kc_ratio_min=0.20
    dynamic_min_kc_ratio_max=0.25
    ;;
  *)
    echo "Unknown MODEL_SIZE: ${model_size}" >&2
    exit 1
    ;;
esac

local_model="${model_root}/${model_dir}"
if [ -n "${MODEL_PATH:-}" ]; then
  model_id="${MODEL_PATH}"
elif [ -d "${local_model}" ]; then
  model_id="$(cd "${local_model}" && pwd)"
fi

[ -f "${prompt_file}" ] || { echo "Missing prompt: ${prompt_file}" >&2; exit 1; }
[ -f "${image_file}" ] || { echo "Missing image: ${image_file}" >&2; exit 1; }
prompt="$(< "${prompt_file}")"

resolution="${RESOLUTION:-720p}"
num_frames="${NUM_FRAMES:-81}"
seed="${SEED:-1}"
run_id="${RUN_ID:-0}"
cpu_offload="${CPU_OFFLOAD:-${WAN_CPU_OFFLOAD:-${default_cpu_offload}}}"
export SVOO_ENABLE_MEM_SAVE="${SVOO_ENABLE_MEM_SAVE:-${default_mem_save}}"
output_file="${OUTPUT_FILE:-${output_dir}/${prompt_id}-${run_id}.mp4}"

cmd=(
  "${python_bin}"
  wan_i2v_inference.py
  --model_id "${model_id}"
  --prompt "${prompt}"
  --image_path "${image_file}"
  --num_frames "${num_frames}"
  --seed "${seed}"
  --num_inference_steps "${num_inference_steps}"
  --resolution "${resolution}"
  --cpu_offload "${cpu_offload}"
  --first_times_fp "${first_times_fp}"
  --first_layers_fp "${first_layers_fp}"
  --num_q_centroids "${num_q_centroids}"
  --num_k_centroids "${num_k_centroids}"
  --top_p_kmeans "${top_p_kmeans}"
  --min_kc_ratio "${min_kc_ratio}"
  --kmeans_iter_init "${kmeans_iter_init}"
  --kmeans_iter_step "${kmeans_iter_step}"
  --start_reuse_step "${start_reuse_step}"
  --reuse_interval "${reuse_interval}"
  --use_dynamic_min_kc_ratio
  --sparsity_csv_path "${sparsity_csv}"
  --dynamic_min_kc_ratio_min "${dynamic_min_kc_ratio_min}"
  --dynamic_min_kc_ratio_max "${dynamic_min_kc_ratio_max}"
  --output_file "${output_file}"
)

if [ -n "${HEIGHT:-}${WIDTH:-}" ]; then
  [ -n "${HEIGHT:-}" ] && [ -n "${WIDTH:-}" ] || { echo "HEIGHT and WIDTH must be set together." >&2; exit 1; }
  cmd+=(--height "${HEIGHT}" --width "${WIDTH}")
fi
[ "${SVOO_ZERO_STEP_KMEANS_INIT:-0}" = "1" ] && cmd+=(--zero_step_kmeans_init)
cmd+=("$@")

echo "GPU=${gpu_id} MODEL=${model_id}"
echo "CPU_OFFLOAD=${cpu_offload} SVOO_ENABLE_MEM_SAVE=${SVOO_ENABLE_MEM_SAVE}"
echo "PROMPT=${prompt_file} IMAGE=${image_file}"
echo "OUTPUT=${output_file}"
[ "${DRY_RUN:-0}" = "1" ] && { printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'; exit 0; }

CUDA_VISIBLE_DEVICES="${gpu_id}" ENABLE_FAST_KERNEL="${ENABLE_FAST_KERNEL:-1}" "${cmd[@]}"
