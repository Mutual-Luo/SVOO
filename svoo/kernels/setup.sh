#!/bin/bash
set -euo pipefail

: "${CONDA_PREFIX:?Please activate the svoo conda environment before running this script.}"

CONDA_ENV_PATH=$CONDA_PREFIX

if [[ -z "${CC:-}" && -x "$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-gcc"
fi
if [[ -z "${CXX:-}" && -x "$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CXX="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-g++"
fi
if [[ -z "${CUDAHOSTCXX:-}" && -n "${CXX:-}" ]]; then
    export CUDAHOSTCXX="$CXX"
fi

detect_torch_cuda_arch_list() {
    if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        printf '%s\n' "$TORCH_CUDA_ARCH_LIST"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi was not found. Set TORCH_CUDA_ARCH_LIST manually before building SVOO kernels." >&2
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

    if [[ -z "$arch_list" ]]; then
        echo "Could not auto-detect CUDA compute capability. Set TORCH_CUDA_ARCH_LIST manually before building SVOO kernels." >&2
        exit 1
    fi

    printf '%s\n' "$arch_list"
}

export CUDA_HOME=$CONDA_ENV_PATH
export CUDA_PATH=$CONDA_ENV_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST="$(detect_torch_cuda_arch_list)"
printf 'Using TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"

NVJITLINK_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')")
if [[ -d "$NVJITLINK_PATH" ]]; then
    export LD_LIBRARY_PATH=$NVJITLINK_PATH:${LD_LIBRARY_PATH:-}
fi

CUDA_TARGET_PATH="$CONDA_PREFIX/targets/x86_64-linux"
NVTX_INCLUDE_PATH=$(python - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.nvtx.include")
print(pathlib.Path(spec.origin).parent if spec else "")
PY
)

rm -rf build
mkdir -p build
cd build

cmake_args=(
    ..
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
    -DCAFFE2_USE_CUDA=ON
    -DUSE_CUDA=ON
)

if [[ -n "${CC:-}" ]]; then
    cmake_args+=(-DCMAKE_C_COMPILER="$CC")
fi
if [[ -n "${CXX:-}" ]]; then
    cmake_args+=(-DCMAKE_CXX_COMPILER="$CXX")
fi
if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    cmake_args+=(-DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc")
fi
if [[ -d "$CUDA_TARGET_PATH" ]]; then
    cmake_args+=(
        -DCUDAToolkit_ROOT="$CUDA_TARGET_PATH"
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TARGET_PATH"
        -DCUDA_INCLUDE_DIRS="$CUDA_TARGET_PATH/include"
    )
fi

if [[ -n "$NVTX_INCLUDE_PATH" ]]; then
    cmake_args+=(
        -DUSE_SYSTEM_NVTX=ON
        -Dnvtx3_dir="$NVTX_INCLUDE_PATH"
    )
fi

cmake "${cmake_args[@]}"

make -j"${MAX_JOBS:-$(nproc)}"
