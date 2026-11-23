"""Pretrained RVC weights: download at job start, cache on the network volume if attached.

Public catalog: lj1995/VoiceConversionWebUI (no Hugging Face token required).
Weights are never baked into the image. The cache mirrors the official assets/ layout
from the README on SHA 81eed5e8f68b6bed1789f682fe78cdd324495afc.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

from .paths import RVC_DIR, model_cache_dir
from .secret_utils import redact

log = logging.getLogger(__name__)

HF_REPO = "lj1995/VoiceConversionWebUI"
UPSTREAM_SHA = "81eed5e8f68b6bed1789f682fe78cdd324495afc"

# Expected sizes from the Researcher spec (HF API, main), used to reject truncated files.
SIZES: dict[str, int] = {
    "hubert_base/pytorch_model.bin": 189_206_711,
    "rmvpe/rmvpe.pt": 181_184_272,
    "pretrained_v2/f0G40k.pth": 73_106_273,
    "pretrained_v2/f0D40k.pth": 142_875_703,
    "pretrained_v2/f0G32k.pth": 73_950_049,
    "pretrained_v2/f0G48k.pth": 75_465_569,
    "pretrained_v2/f0D32k.pth": 142_875_703,
    "pretrained_v2/f0D48k.pth": 142_875_703,
}

# Files that must exist and be non-empty (size not pinned in the spec).
SIDE_FILES = (
    "hubert_base/config.json",
    "hubert_base/preprocessor_config.json",
)

MUTE_ZIP = "mute.zip"
MUTE_ZIP_SIZE = 312_273


class ModelError(RuntimeError):
    pass


def pretrained_files(version: str, sample_rate: str) -> tuple[str, str]:
    folder = "pretrained" if version == "v1" else "pretrained_v2"
    g = f"{folder}/f0G{sample_rate}.pth"
    d = f"{folder}/f0D{sample_rate}.pth"
    return g, d


def required_files(version: str, sample_rate: str) -> list[str]:
    g, d = pretrained_files(version, sample_rate)
    return [
        "hubert_base/pytorch_model.bin",
        *SIDE_FILES,
        "rmvpe/rmvpe.pt",
        g,
        d,
    ]


def cache_path(rel: str) -> str:
    return os.path.join(model_cache_dir(), rel)


def asset_path(rel: str) -> str:
    return os.path.join(RVC_DIR, "assets", rel)


def _link_or_copy(src: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.islink(dest) or os.path.isfile(dest):
        os.remove(dest)
    try:
        os.symlink(os.path.abspath(src), dest)
    except OSError:
        shutil.copy2(src, dest)


def _download_file(repo_path: str, dest: str, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # repo_path is the path inside lj1995/VoiceConversionWebUI.
    # rmvpe.pt lives at the repo root; we store it under rmvpe/rmvpe.pt as the README says.
    hub_name = "rmvpe.pt" if repo_path == "rmvpe/rmvpe.pt" else repo_path
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=hub_name,
        revision="main",
        token=token,
        local_dir=None,
    )
    if os.path.abspath(path) != os.path.abspath(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(path, dest)
    return dest


def _ok_size(rel: str, path: str) -> None:
    size = os.path.getsize(path)
    expected = SIZES.get(rel)
    if expected and size != expected:
        # Allow a small tolerance only if the file is clearly truncated.
        if size < expected * 0.95:
            raise ModelError(f"{rel}: expected {expected} bytes, got {size} (truncated download)")
        log.warning("%s size is %s, spec expected %s", rel, size, expected)
    if size <= 0:
        raise ModelError(f"{rel}: empty file")


def check_access(cfg, token: str | None = None) -> dict[str, Any]:
    """Metadata-only check (no weights) that every component is reachable without a token."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    out: dict[str, Any] = {}
    for rel in required_files(cfg.version, cfg.sample_rate):
        hub_name = "rmvpe.pt" if rel == "rmvpe/rmvpe.pt" else rel
        try:
            meta = get_hf_file_metadata(hf_hub_url(HF_REPO, hub_name), token=token)
            size = meta.size or 0
            expected = SIZES.get(rel)
            rec: dict[str, Any] = {
                "ok": True, "repo": HF_REPO, "file": hub_name,
                "size_gb": round(size / 1e9, 3),
            }
            if expected and size and size != expected:
                rec["size_warning"] = f"expected {expected} bytes, Hub reports {size}"
            out[rel] = rec
        except Exception as exc:
            out[rel] = {"ok": False, "repo": HF_REPO, "file": hub_name,
                        "error": redact(str(exc)).splitlines()[0][:300]}
    return out


def ensure_mute(rvc_dir: str | None = None, token: str | None = None) -> str:
    """logs/mute is required by train.py (two mute rows in the filelist)."""
    root = rvc_dir or RVC_DIR
    mute_wav = os.path.join(root, "logs", "mute", "0_gt_wavs", "mute40k.wav")
    if os.path.isfile(mute_wav):
        return os.path.join(root, "logs", "mute")
    cache = os.path.join(model_cache_dir(), MUTE_ZIP)
    if not os.path.isfile(cache) or os.path.getsize(cache) < MUTE_ZIP_SIZE * 0.9:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=MUTE_ZIP, revision="main", token=token)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        shutil.copy2(path, cache)
    import zipfile
    dest = os.path.join(root, "logs")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(cache) as z:
        z.extractall(dest)
    if not os.path.isfile(mute_wav):
        # Some zips unpack to mute/ at the top; others to logs/mute.
        alt = os.path.join(root, "mute")
        if os.path.isdir(alt) and not os.path.isdir(os.path.join(root, "logs", "mute")):
            shutil.move(alt, os.path.join(root, "logs", "mute"))
    if not os.path.isfile(mute_wav):
        raise ModelError("mute.zip extracted but logs/mute/0_gt_wavs/mute40k.wav is missing")
    return os.path.join(root, "logs", "mute")


def ensure_models(cfg, token: str | None, progress=None) -> tuple[dict[str, str], dict[str, Any]]:
    needed = required_files(cfg.version, cfg.sample_rate)
    cache = model_cache_dir()
    os.makedirs(cache, exist_ok=True)
    paths: dict[str, str] = {}
    info: dict[str, Any] = {"repo": HF_REPO, "cache": cache, "files": {}}
    for rel in needed:
        dest = cache_path(rel)
        if not os.path.isfile(dest):
            if progress:
                progress({"stage": "downloading_models", "file": rel})
            log.info("downloading %s", rel)
            try:
                _download_file(rel, dest, token)
            except Exception as exc:
                raise ModelError(f"failed to download {rel}: {redact(str(exc)).splitlines()[0][:400]}") from exc
        _ok_size(rel, dest)
        _link_or_copy(dest, asset_path(rel))
        paths[rel] = asset_path(rel)
        info["files"][rel] = {"bytes": os.path.getsize(dest), "cached": True}
    ensure_mute(RVC_DIR, token)
    total = sum(v["bytes"] for v in info["files"].values())
    info["total_gb"] = round(total / 1e9, 3)
    return paths, info
