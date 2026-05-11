import os
import logging

import torch


class _ExactMessageFilter(logging.Filter):
    def __init__(self, blocked_messages: set[str]):
        super().__init__()
        self._blocked_messages = blocked_messages

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in self._blocked_messages


def _suppress_known_startup_warnings() -> None:
    blocked = {"`torch_dtype` is deprecated! Use `dtype` instead!"}
    logger = logging.getLogger("transformers")
    for existing_filter in logger.filters:
        if isinstance(existing_filter, _ExactMessageFilter):
            return
    logger.addFilter(_ExactMessageFilter(blocked))


def configure_cuda_linalg_backend() -> None:
    _suppress_known_startup_warnings()
    backend = os.environ.get("SVOO_CUDA_LINALG_BACKEND", "").strip()
    if backend:
        torch.backends.cuda.preferred_linalg_library(backend=backend)
