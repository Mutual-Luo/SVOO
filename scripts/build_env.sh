#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${CONDA_ENV_NAME:-svoo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.9}"
CUDA_VERSION="${CUDA_VERSION:-12.6.0}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-12.6.20}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu126}"
GCC_VERSION="${GCC_VERSION:-11.2.0}"
SYSROOT_VERSION="${SYSROOT_VERSION:-2.28}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

UPDATE_SUBMODULES="${UPDATE_SUBMODULES:-1}"

log() {
    printf '\n==> %s\n' "$*"
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

require_conda() {
    local conda_bin
    conda_bin="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
    if [ -z "${conda_bin}" ]; then
        echo "Conda was not found. Install Miniconda/Mambaforge and ensure conda is on PATH." >&2
        exit 1
    fi
    eval "$("${conda_bin}" shell.bash hook)"
}

conda_env_exists() {
    conda run -n "${ENV_NAME}" python -c "import sys" >/dev/null 2>&1
}

log_nvidia_environment() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "Detected NVIDIA environment"
        nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
    fi
}

detect_torch_cuda_arch_list() {
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        printf '%s\n' "${TORCH_CUDA_ARCH_LIST}"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi was not found. Set TORCH_CUDA_ARCH_LIST manually for this build host." >&2
        exit 1
    fi

    local arch_list
    arch_list="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
            awk '
                {
                    gsub(/[[:space:]]/, "", $0)
                    if ($0 ~ /^[0-9]+[.][0-9]+$/ && !seen[$0]++) {
                        if (out != "") {
                            out = out ";"
                        }
                        out = out $0
                    }
                }
                END { print out }
            '
    )"

    if [ -z "${arch_list}" ]; then
        echo "Could not auto-detect CUDA compute capability. Set TORCH_CUDA_ARCH_LIST manually." >&2
        exit 1
    fi

    printf '%s\n' "${arch_list}"
}

install_conda_packages() {
    log "Installing CUDA toolkit and build tools"
    run conda install -y -n "${ENV_NAME}" -c nvidia \
        "cuda-nvcc=${CUDA_NVCC_VERSION}" \
        "cuda-toolkit=${CUDA_VERSION}"

    run conda install -y -n "${ENV_NAME}" -c conda-forge \
        "gcc_linux-64=${GCC_VERSION}" \
        "gxx_linux-64=${GCC_VERSION}" \
        "sysroot_linux-64=${SYSROOT_VERSION}" \
        binutils_linux-64 \
        cmake \
        git \
        ninja
}

activate_env() {
    log "Activating conda environment: ${ENV_NAME}"
    conda activate "${ENV_NAME}"

    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDA_PATH="${CONDA_PREFIX}"
    export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

    if [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]; then
        export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    fi
    if [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]; then
        export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
        export CUDAHOSTCXX="${CXX}"
    fi

    log_nvidia_environment

    export TORCH_CUDA_ARCH_LIST
    TORCH_CUDA_ARCH_LIST="$(detect_torch_cuda_arch_list)"
    log "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    export MAX_JOBS
}

install_python_packages() {
    log "Installing PyTorch ${TORCH_VERSION} (${TORCH_CUDA_INDEX})"
    run python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"

    log "Installing Python build requirements"
    run python -m pip install -U \
        pip \
        setuptools \
        wheel \
        cmake \
        ninja \
        packaging \
        psutil \
        hatchling \
        editables \
        "huggingface_hub[cli]"

    log "Installing SVOO package"
    run python -m pip install --no-build-isolation -e "${PROJECT_ROOT}"
}

install_submodules() {
    if [ "${UPDATE_SUBMODULES}" != "1" ]; then
        log "Skipping git submodule update"
        return
    fi
    log "Initializing git submodules"
    run git -C "${PROJECT_ROOT}" submodule update --init --recursive
}

build_svoo_kernels() {
    log "Building SVOO CUDA extension"
    run bash "${PROJECT_ROOT}/svoo/kernels/setup.sh"
}

install_flash_attention() {
    log "Installing FlashAttention from local submodule"
    run python -m pip install --no-build-isolation --verbose --editable \
        "${PROJECT_ROOT}/svoo/kernels/3rdparty/flash-attention"
}

install_flashinfer() {
    log "Installing FlashInfer from local submodule"
    run python -m pip install --no-build-isolation --verbose --editable \
        "${PROJECT_ROOT}/svoo/kernels/3rdparty/flashinfer"
}

main() {
    require_conda

    if conda_env_exists; then
        log "Reusing existing conda environment: ${ENV_NAME}"
    else
        log "Creating conda environment: ${ENV_NAME}"
        run conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi

    install_conda_packages
    activate_env
    install_submodules
    install_python_packages
    install_flash_attention
    install_flashinfer
    build_svoo_kernels

    log "Environment is ready"
    printf 'Activate it with:\n  conda activate %s\n' "${ENV_NAME}"
}

main "$@"
