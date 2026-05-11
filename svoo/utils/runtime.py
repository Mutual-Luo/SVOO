import os

import torch


def configure_cuda_linalg_backend() -> None:
    backend = os.environ.get("SVOO_CUDA_LINALG_BACKEND", "").strip()
    if backend:
        torch.backends.cuda.preferred_linalg_library(backend=backend)
