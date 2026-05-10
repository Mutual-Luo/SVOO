#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# key|task|hf_id|local_dir|steps|frames|csv|layers|heads|model_id_env|model_dir_env
MODEL_CONFIGS=(
  "wan21_t2v_1_3b|wan_t2v|Wan-AI/Wan2.1-T2V-1.3B-Diffusers|Wan2.1-T2V-1.3B-Diffusers|50|81|sparsity_wan_1.3B_t2v.csv|30|12|WAN21_T2V_1_3B_MODEL_ID|WAN21_T2V_1_3B_MODEL_DIR"
  "wan21_t2v_14b|wan_t2v|Wan-AI/Wan2.1-T2V-14B-Diffusers|Wan2.1-T2V-14B-Diffusers|50|81|sparsity_wan_14B_t2v.csv|40|40|WAN21_T2V_14B_MODEL_ID|WAN21_T2V_14B_MODEL_DIR"
  "wan21_i2v_14b|wan_i2v|Wan-AI/Wan2.1-I2V-14B-720P-Diffusers|Wan2.1-I2V-14B-720P-Diffusers|40|81|sparsity_wan_14B_i2v.csv|40|40|WAN21_I2V_14B_MODEL_ID|WAN21_I2V_14B_MODEL_DIR"
  "wan22_t2v_a14b|wan_t2v|Wan-AI/Wan2.2-T2V-A14B-Diffusers|Wan2.2-T2V-A14B-Diffusers|40|81|sparsity_wan22_A14B_t2v.csv|40|40|WAN22_T2V_A14B_MODEL_ID|WAN22_T2V_A14B_MODEL_DIR"
  "wan22_i2v_a14b|wan_i2v|Wan-AI/Wan2.2-I2V-A14B-Diffusers|Wan2.2-I2V-A14B-Diffusers|40|81|sparsity_wan22_A14B_i2v.csv|40|40|WAN22_I2V_A14B_MODEL_ID|WAN22_I2V_A14B_MODEL_DIR"
  "hunyuan10_t2v_13b|hunyuan10_t2v|tencent/HunyuanVideo|HunyuanVideo|50|129|sparsity_hunyuan10_13B_t2v.csv|60|24|HUNYUAN10_T2V_MODEL_ID|HUNYUAN10_T2V_MODEL_DIR"
  "hunyuan10_i2v_13b|hunyuan10_i2v|hunyuanvideo-community/HunyuanVideo-I2V|HunyuanVideo-I2V|50|129|sparsity_hunyuan10_13B_i2v.csv|60|24|HUNYUAN10_I2V_MODEL_ID|HUNYUAN10_I2V_MODEL_DIR"
)

OVERRIDE_INFER_STEPS="${INFER_STEPS:-}"
OVERRIDE_NUM_FRAMES="${NUM_FRAMES:-}"
OVERRIDE_EXPECTED_LAYERS="${EXPECTED_LAYERS:-}"
OVERRIDE_EXPECTED_HEADS="${EXPECTED_HEADS:-}"

usage() {
  cat <<EOF
Usage:
  GPUS=0 bash scripts/offline/generate_sparsity_profiles.sh [all|model_key ...]
  GPUS="0 1 2 3" bash scripts/offline/generate_sparsity_profiles.sh all

One GPU runs jobs serially. Multiple GPUs run one queue per GPU in parallel.

Model keys:
$(printf '%s\n' "${MODEL_CONFIGS[@]}" | cut -d'|' -f1 | sed 's/^/  /')

Common overrides:
  GPUS="0"                         GPU ids. Use spaces or commas for multiple GPUs.
  PROFILE_PROMPT_FILE=data/profile_data/prompt.txt
                                    One prompt per line; all non-empty lines are profiled.
  PROFILE_IMAGE_ROOT=data/profile_data/image
                                    For i2v only: image for prompt line N.
  MODEL_ID=/path/or/hf-id          Override model path or Hugging Face id.
  MODEL_ROOT=/home/dataset-assist-0/luojy/models
  OFFLINE_PROFILE_ROOT=sparsity_profiles
  SPARSITY_THRESHOLD=0.95
  SPARSITY_BATCH_SIZE=0            Auto-select largest safe exact query chunk.
  SPARSITY_QUERY_SAMPLES=0         Query rows sampled per layer/head; 0 uses all queries.
  RESUME_PROFILE=1                 Skip complete entries already present in raw txt.
  RUN_INFERENCE=0                  Merge existing raw txt files only.

Output:
  <OFFLINE_PROFILE_ROOT>/sparsity_*.csv
  <OFFLINE_PROFILE_ROOT>/runs/<model_key>/runner.log
  <OFFLINE_PROFILE_ROOT>/runs/<model_key>/{raw,logs,videos}/
EOF
}

model_keys() {
  printf '%s\n' "${MODEL_CONFIGS[@]}" | cut -d'|' -f1
}

load_model() {
  local wanted="$1" row
  for row in "${MODEL_CONFIGS[@]}"; do
    IFS='|' read -r MODEL_KEY TASK MODEL_HF_ID MODEL_DIR_NAME DEFAULT_STEPS DEFAULT_FRAMES \
      OUTPUT_CSV_NAME DEFAULT_LAYERS DEFAULT_HEADS MODEL_ID_ENV MODEL_DIR_ENV <<< "${row}"
    if [ "${MODEL_KEY}" = "${wanted}" ]; then
      return 0
    fi
  done
  return 1
}

resolve_model_id() {
  local model_id_from_env="${!MODEL_ID_ENV:-}"
  local model_dir="${!MODEL_DIR_ENV:-${MODEL_DIR_NAME}}"
  local model_root="${MODEL_ROOT:-/home/dataset-assist-0/luojy/models}"

  if [ -n "${MODEL_ID:-}" ]; then
    MODEL_ID_RESOLVED="${MODEL_ID}"
  elif [ -n "${model_id_from_env}" ]; then
    MODEL_ID_RESOLVED="${model_id_from_env}"
  elif [ -d "${model_root}/${model_dir}" ]; then
    MODEL_ID_RESOLVED="${model_root}/${model_dir}"
  else
    MODEL_ID_RESOLVED="${MODEL_HF_ID}"
  fi
}

setup_env() {
  cd "${PROJECT_ROOT}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.triton_cache}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export SVG_REUSE_DEBUG="${SVG_REUSE_DEBUG:-0}"
  export SVG_ENABLE_MEM_SAVE="${SVG_ENABLE_MEM_SAVE:-0}"
  export FLASHINFER_WORKSPACE_SIZE="${FLASHINFER_WORKSPACE_SIZE:-128000000}"
  export SVOO_SPARSITY_USE_TRITON_PROB_COUNTS="${SVOO_SPARSITY_USE_TRITON_PROB_COUNTS:-1}"
  export SVOO_SPARSITY_TRITON_BOUNDARY_TOL="${SVOO_SPARSITY_TRITON_BOUNDARY_TOL:-1e-6}"

  if [ -z "${FLASH_BASE:-}" ] && [ -d "${PROJECT_ROOT}/svoo/kernels/3rdparty/flashinfer" ]; then
    export FLASH_BASE="${PROJECT_ROOT}/svoo/kernels/3rdparty/flashinfer"
  fi
  if [ -n "${FLASH_BASE:-}" ]; then
    export CPLUS_INCLUDE_PATH="${FLASH_BASE}/3rdparty/cutlass/include:${FLASH_BASE}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  fi
}

run_on_gpu() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" ENABLE_FAST_KERNEL="${ENABLE_FAST_KERNEL}" SVOO_SPARSITY_RESUME="${RESUME_PROFILE}" "$@"
}

find_profile_image() {
  local prompt_id="$1" candidate
  local candidates=(
    "${PROFILE_IMAGE_ROOT}/${prompt_id}.jpg"
    "${PROFILE_IMAGE_ROOT}/${prompt_id}.jpeg"
    "${PROFILE_IMAGE_ROOT}/${prompt_id}.png"
    "${PROFILE_IMAGE_ROOT}/${prompt_id}/image.jpg"
    "${PROFILE_IMAGE_ROOT}/${prompt_id}/image.jpeg"
    "${PROFILE_IMAGE_ROOT}/${prompt_id}/image.png"
  )

  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

build_inference_cmd() {
  local prompt="$1" image_file="$2" output_file="$3" log_file="$4" sparsity_file="$5"
  local common_args=(
    --model_id "${MODEL_ID_RESOLVED}"
    --prompt "${prompt}"
    --seed "${SEED}"
    --num_frames "${INFER_FRAMES_RUN}"
    --num_inference_steps "${INFER_STEPS_RUN}"
    --pattern SVOO
    --first_times_fp "${FIRST_TIMES_FP}"
    --first_layers_fp "${FIRST_LAYERS_FP}"
    --measure_attention_sparsity 1
    --sparsity_output_file "${sparsity_file}"
    --sparsity_batch_size "${SPARSITY_BATCH_SIZE}"
    --sparsity_query_samples "${SPARSITY_QUERY_SAMPLES}"
    --sparsity_threshold "${SPARSITY_THRESHOLD}"
    --sparsity_start_step 1
    --output_file "${output_file}"
    --logging_file "${log_file}"
  )

  case "${TASK}" in
    wan_t2v)
      CMD=(python wan_t2v_inference.py "${common_args[@]}" --height "${HEIGHT}" --width "${WIDTH}" --cpu_offload "${WAN_CPU_OFFLOAD}")
      ;;
    wan_i2v)
      CMD=(python wan_i2v_inference.py "${common_args[@]}" --image_path "${image_file}" --resolution "${RESOLUTION}" --cpu_offload "${WAN_CPU_OFFLOAD}")
      ;;
    hunyuan10_t2v)
      CMD=(python hunyuan10_t2v_inference.py "${common_args[@]}" --height "${HEIGHT}" --width "${WIDTH}")
      [ "${HUNYUAN10_CPU_OFFLOAD}" = "1" ] && CMD+=(--enable_cpu_offload)
      ;;
    hunyuan10_i2v)
      CMD=(python hunyuan10_i2v_inference.py "${common_args[@]}" --image_path "${image_file}" --resolution "${RESOLUTION}")
      [ "${HUNYUAN10_CPU_OFFLOAD}" = "1" ] && CMD+=(--enable_cpu_offload)
      ;;
  esac
}

run_prompt() {
  local gpu="$1" prompt_id="$2" prompt="$3"
  local image_file=""
  local raw_file="${RAW_DIR}/attention_sparsity-${prompt_id}-th${THRESHOLD_TAG}-${QUERY_SAMPLE_TAG}.txt"
  local log_file="${LOG_DIR}/${prompt_id}.jsonl"
  local video_file="${VIDEO_DIR}/${prompt_id}.mp4"

  if [[ "${TASK}" == *i2v ]] && [ "${RUN_INFERENCE}" = "1" ]; then
    if ! image_file="$(find_profile_image "${prompt_id}")"; then
      echo "[${MODEL_KEY}] skip line ${prompt_id}: missing image under ${PROFILE_IMAGE_ROOT}"
      return 1
    fi
  fi

  echo "[${MODEL_KEY}] prompt_line=${prompt_id} -> ${raw_file}"
  [ "${RUN_INFERENCE}" = "1" ] && [ "${RESUME_PROFILE}" != "1" ] && rm -f "${raw_file}"

  if [ "${RUN_INFERENCE}" = "1" ]; then
    build_inference_cmd "${prompt}" "${image_file}" "${video_file}" "${log_file}" "${raw_file}"
    run_on_gpu "${gpu}" "${CMD[@]}"
  fi
}

run_prompts_from_file() {
  local gpu="$1" prompt_file="$2"
  local prompt line_no=0 expected_count=0

  [ -f "${prompt_file}" ] || { echo "Missing PROFILE_PROMPT_FILE: ${prompt_file}" >&2; return 1; }

  while IFS= read -r prompt || [ -n "${prompt}" ]; do
    line_no=$((line_no + 1))
    [ -n "${prompt//[[:space:]]/}" ] || continue
    if run_prompt "${gpu}" "${line_no}" "${prompt}"; then
      expected_count=$((expected_count + 1))
    else
      return 1
    fi
  done < "${prompt_file}"

  [ "${expected_count}" -gt 0 ] || { echo "[${MODEL_KEY}] no valid prompts found in ${prompt_file}" >&2; return 1; }
  EXPECTED_COUNT="${expected_count}"
}

run_model() {
  local key="$1" gpu="$2"
  load_model "${key}" || { echo "Unknown model key: ${key}" >&2; return 1; }
  resolve_model_id

  local profile_root="${OFFLINE_PROFILE_ROOT:-sparsity_profiles}"
  local run_dir="${profile_root}/runs/${MODEL_KEY}"
  local output_csv="${profile_root}/${OUTPUT_CSV_NAME}"
  RAW_DIR="${run_dir}/raw"
  LOG_DIR="${run_dir}/logs"
  VIDEO_DIR="${run_dir}/videos"

  PROFILE_PROMPT_FILE="${PROFILE_PROMPT_FILE:-data/profile_data/prompt.txt}"
  PROFILE_IMAGE_ROOT="${PROFILE_IMAGE_ROOT:-data/profile_data/image}"
  INFER_STEPS_RUN="${OVERRIDE_INFER_STEPS:-${DEFAULT_STEPS}}"
  INFER_FRAMES_RUN="${OVERRIDE_NUM_FRAMES:-${DEFAULT_FRAMES}}"
  EXPECTED_LAYERS_RUN="${OVERRIDE_EXPECTED_LAYERS:-${DEFAULT_LAYERS}}"
  EXPECTED_HEADS_RUN="${OVERRIDE_EXPECTED_HEADS:-${DEFAULT_HEADS}}"
  HEIGHT="${HEIGHT:-720}"
  WIDTH="${WIDTH:-1280}"
  RESOLUTION="${RESOLUTION:-720p}"
  SPARSITY_THRESHOLD="${SPARSITY_THRESHOLD:-0.95}"
  SPARSITY_BATCH_SIZE="${SPARSITY_BATCH_SIZE:-0}"
  SPARSITY_QUERY_SAMPLES="${SPARSITY_QUERY_SAMPLES:-0}"
  case "${SPARSITY_QUERY_SAMPLES}" in
    ''|*[!0-9]*)
      echo "SPARSITY_QUERY_SAMPLES must be a non-negative integer, got: ${SPARSITY_QUERY_SAMPLES}" >&2
      return 1
      ;;
  esac
  FIRST_TIMES_FP="${FIRST_TIMES_FP:-1.0}"
  FIRST_LAYERS_FP="${FIRST_LAYERS_FP:-1.0}"
  SEED="${SEED:-0}"
  WAN_CPU_OFFLOAD="${WAN_CPU_OFFLOAD:-0}"
  HUNYUAN10_CPU_OFFLOAD="${HUNYUAN10_CPU_OFFLOAD:-0}"
  RUN_INFERENCE="${RUN_INFERENCE:-1}"
  MERGE_PROFILE="${MERGE_PROFILE:-1}"
  RESUME_PROFILE="${RESUME_PROFILE:-0}"
  ENABLE_FAST_KERNEL="${ENABLE_FAST_KERNEL:-1}"
  THRESHOLD_TAG="$(awk -v x="${SPARSITY_THRESHOLD}" 'BEGIN { printf "%d", x * 100 + 0.5 }')"
  if [ "${SPARSITY_QUERY_SAMPLES}" -gt 0 ]; then
    QUERY_SAMPLE_TAG="q${SPARSITY_QUERY_SAMPLES}"
  else
    QUERY_SAMPLE_TAG="qall"
  fi

  mkdir -p "$(dirname "${output_csv}")" "${RAW_DIR}" "${LOG_DIR}" "${VIDEO_DIR}" "${TRITON_CACHE_DIR}"

  echo "=== ${MODEL_KEY} ==="
  echo "MODEL_ID=${MODEL_ID_RESOLVED}"
  echo "PROFILE_PROMPT_FILE=${PROFILE_PROMPT_FILE}"
  echo "SPARSITY_QUERY_SAMPLES=${SPARSITY_QUERY_SAMPLES}"
  echo "OUTPUT_CSV=${output_csv}"
  echo "RUN_DIR=${run_dir}"

  run_prompts_from_file "${gpu}" "${PROFILE_PROMPT_FILE}"

  if [ "${MERGE_PROFILE}" = "1" ]; then
    local merge_pattern="attention_sparsity-*-th${THRESHOLD_TAG}-${QUERY_SAMPLE_TAG}.txt"
    local legacy_pattern="attention_sparsity-*-th${THRESHOLD_TAG}.txt"
    if [ "${RUN_INFERENCE}" != "1" ] && ! compgen -G "${RAW_DIR}/${merge_pattern}" >/dev/null; then
      if compgen -G "${RAW_DIR}/${legacy_pattern}" >/dev/null; then
        echo "Using legacy raw sparsity pattern: ${legacy_pattern}"
        merge_pattern="${legacy_pattern}"
      fi
    fi

    python -m svoo.sparsity.merge \
      "${RAW_DIR}" "${output_csv}" \
      --pattern "${merge_pattern}" \
      --expected-count "${EXPECTED_COUNT}" \
      --expected-steps "${INFER_STEPS_RUN}" \
      --expected-layers "${EXPECTED_LAYERS_RUN}" \
      --expected-heads "${EXPECTED_HEADS_RUN}"
  fi
}

run_model_logged() {
  local key="$1" gpu="$2"
  local log="$(profile_root_path)/runs/${key}/runner.log"
  mkdir -p "$(dirname "${log}")"
  echo "[launch] ${key} on GPU ${gpu} -> ${log}"
  (setup_env; run_model "${key}" "${gpu}") >"${log}" 2>&1
}

select_models() {
  SELECTED_MODELS=()
  if [ "$#" -eq 0 ] || [ "${1:-}" = "all" ]; then
    mapfile -t SELECTED_MODELS < <(model_keys)
  else
    local key
    for key in "$@"; do
      key="${key##*/}"
      key="${key%.sh}"
      load_model "${key}" || { echo "Unknown model key: ${key}" >&2; usage; exit 1; }
      SELECTED_MODELS+=("${key}")
    done
  fi
}

parse_gpus() {
  local gpu_text="${GPUS:-0}"
  gpu_text="${gpu_text//,/ }"
  read -r -a GPU_IDS <<< "${gpu_text}"
  [ "${#GPU_IDS[@]}" -gt 0 ] || { echo "GPUS must contain at least one id" >&2; exit 1; }
}

profile_root_path() {
  local root="${OFFLINE_PROFILE_ROOT:-sparsity_profiles}"
  case "${root}" in
    /*) printf '%s\n' "${root}" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "${root}" ;;
  esac
}

main() {
  if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
  fi

  select_models "$@"
  parse_gpus

  local status=0 key i gpu_idx
  if [ "${#GPU_IDS[@]}" -eq 1 ]; then
    for key in "${SELECTED_MODELS[@]}"; do
      if run_model_logged "${key}" "${GPU_IDS[0]}"; then
        echo "[done] ${key}"
      else
        echo "[failed] ${key}" >&2
        status=1
      fi
    done
  else
    local pids=()
    for gpu_idx in "${!GPU_IDS[@]}"; do
      (
        setup_env
        local queue_status=0
        for i in "${!SELECTED_MODELS[@]}"; do
          [ "$((i % ${#GPU_IDS[@]}))" -eq "${gpu_idx}" ] || continue
          if run_model_logged "${SELECTED_MODELS[$i]}" "${GPU_IDS[$gpu_idx]}"; then
            echo "[done] ${SELECTED_MODELS[$i]}"
          else
            echo "[failed] ${SELECTED_MODELS[$i]}" >&2
            queue_status=1
          fi
        done
        exit "${queue_status}"
      ) &
      pids+=("$!")
    done

    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || status=1
    done
  fi

  exit "${status}"
}

main "$@"
