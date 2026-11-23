"""Where models, datasets and outputs live."""

from __future__ import annotations

import os

NETWORK_VOLUME = "/runpod-volume"
RVC_DIR = os.environ.get("RVC_DIR", "/opt/rvc")


def has_volume() -> bool:
    return os.path.isdir(NETWORK_VOLUME) and os.access(NETWORK_VOLUME, os.W_OK)


def model_cache_dir() -> str:
    if os.environ.get("MODEL_CACHE_DIR"):
        return os.environ["MODEL_CACHE_DIR"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "rvc", "models")
    return "/tmp/rvc/models"


def output_root() -> str:
    if os.environ.get("OUTPUT_ROOT"):
        return os.environ["OUTPUT_ROOT"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "rvc", "runs")
    return "/tmp/rvc/runs"


def configure_caches() -> None:
    if has_volume():
        os.environ.setdefault("HF_HOME", os.path.join(NETWORK_VOLUME, "rvc", "hf-home"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(NETWORK_VOLUME, "rvc", "hf-hub"))
