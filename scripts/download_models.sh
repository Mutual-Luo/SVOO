#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-assist-0/luojy/models}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"

case "${RETRY_SLEEP}" in
    ''|*[!0-9]*)
        echo "RETRY_SLEEP must be a non-negative integer number of seconds." >&2
        exit 2
        ;;
esac

mkdir -p "${MODEL_ROOT}"

find_hf_cli() {
    if command -v hf >/dev/null 2>&1 && hf download -h >/dev/null 2>&1; then
        command -v hf
        return 0
    fi
    if command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli download -h >/dev/null 2>&1; then
        command -v huggingface-cli
        return 0
    fi
    return 1
}

HF_CMD="$(find_hf_cli || true)"
if [ -z "${HF_CMD}" ]; then
    echo "Hugging Face CLI not found; installing huggingface_hub[cli] with current python..."
    python -m pip install -U "huggingface_hub[cli]"
    HF_CMD="$(find_hf_cli || true)"
fi

if [ -z "${HF_CMD}" ]; then
    echo "Could not find a Hugging Face CLI with the download command after installation." >&2
    echo "If you need to log in first, run: hf auth login" >&2
    echo "Older installations may use: huggingface-cli login" >&2
    exit 1
fi

echo "Using Hugging Face CLI: ${HF_CMD}"
echo "Login is optional for these public models. If you need it, run: hf auth login"

download_one() {
    local repo_id="$1"
    local revision="${2:-}"
    local local_dir="${MODEL_ROOT}/${repo_id##*/}"
    local attempt=1
    local status=0

    mkdir -p "${local_dir}"

    while true; do
        echo
        if [ -n "${revision}" ]; then
            echo "==> Downloading ${repo_id} (${revision})"
            echo "    to ${local_dir}"
            "${HF_CMD}" download "${repo_id}" --revision "${revision}" --local-dir "${local_dir}"
        else
            echo "==> Downloading ${repo_id}"
            echo "    to ${local_dir}"
            "${HF_CMD}" download "${repo_id}" --local-dir "${local_dir}"
        fi
        status=$?

        if [ "${status}" -eq 0 ]; then
            echo "==> Done: ${repo_id}"
            return 0
        fi

        echo "==> Download failed with exit code ${status}; retrying in ${RETRY_SLEEP}s (attempt ${attempt})..." >&2
        attempt=$((attempt + 1))
        sleep "${RETRY_SLEEP}"
    done
}

download_one "tencent/HunyuanVideo" "refs/pr/18"

download_one "hunyuanvideo-community/HunyuanVideo-I2V"

download_one "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

download_one "Wan-AI/Wan2.1-T2V-14B-Diffusers"

download_one "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

download_one "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

download_one "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

echo
echo "All downloads finished under ${MODEL_ROOT}."
