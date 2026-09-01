"""FastAPI server + client for Ling-3.0-tiny-int4 (no vLLM / no SGLang)."""

from scripts.ling_server.server import (
    DEFAULT_CACHE_DIR,
    DEFAULT_GPU,
    DEFAULT_PORT,
    LING_MODEL_DIR,
    MAX_COMPLETION_TOKENS,
    app,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_GPU",
    "DEFAULT_PORT",
    "LING_MODEL_DIR",
    "MAX_COMPLETION_TOKENS",
    "app",
]