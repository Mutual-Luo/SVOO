conda create -n svoo python==3.12.9
conda activate svoo

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda install nvidia::cuda-nvcc==12.6.20
conda install nvidia::cuda-toolkit==12.6.0
conda install nvidia::cuda==12.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# conda install -c nvidia -c defaults -c conda-forge --override-channels \
#     gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 binutils_linux-64 sysroot_linux-64=2.28
conda install -c conda-forge gcc_linux-64 gxx_linux-64 cmake ninja -y

export TORCH_CUDA_ARCH_LIST="8.0"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
MAX_JOBS=32 python -m pip install -v --no-build-isolation -e .

python -m pip install uv
uv pip install -e "$PROJECT_ROOT"




git -C "$PROJECT_ROOT" submodule update --init --recursive


python -m pip install -U setuptools wheel ninja packaging psutil



cd "$PROJECT_ROOT/svoo/kernels"
python -m pip install -U cmake
bash setup.sh


cd "$PROJECT_ROOT/svoo/kernels/3rdparty/flashinfer"
python -m pip install --no-build-isolation --verbose --editable .
