"""Deliver trained RVC weights: HF Hub, S3-compatible bucket, presigned PUT; optional sample."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
MAX_BASE64_BYTES = 20 * 2**20
UPSTREAM_SHA = "81eed5e8f68b6bed1789f682fe78cdd324495afc"


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def encode_file_base64(path: str, max_bytes: int = MAX_BASE64_BYTES) -> str | None:
    size = os.path.getsize(path)
    if size > max_bytes:
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def model_card(cfg, output_name: str, train: dict[str, Any], dataset: dict[str, Any]) -> str:
    lines = [
        "---",
        "tags:",
        "- rvc",
        "- retrieval-based-voice-conversion",
        "- voice-conversion",
        "- runpod",
        f"- {cfg.version}",
        "---",
        "",
        f"# {output_name}",
        "",
        f"**RVC {cfg.version}** ({cfg.sample_rate}, F0={cfg.f0_method}) trained with "
        "[RVC-Project/Retrieval-based-Voice-Conversion-WebUI]"
        f"(https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) "
        f"(`{UPSTREAM_SHA}`) on Runpod Serverless using "
        "[prilojject/runpod-rvc-trainer](https://github.com/prilojject/runpod-rvc-trainer).",
        "",
        f"- version: `{cfg.version}`",
        f"- sample rate: `{cfg.sample_rate}`",
        f"- F0 method: `{cfg.f0_method}`",
        f"- epochs: {cfg.epochs}",
        f"- batch size: {train.get('batch_size', cfg.batch_size)}",
        f"- seed: {cfg.seed}",
        f"- clips: {dataset.get('clips_used')}, audio: {dataset.get('total_seconds')}s",
        "",
        "## How to load the weights",
        "",
        "In the official RVC WebUI, put:",
        "",
        f"- `{output_name}.pth` in `assets/weights/`",
        f"- `{output_name}.index` in `assets/indices/`",
        "",
        "Or with the official CLI (from the RVC project root):",
        "",
        "```bash",
        "python infer/cli.py \\",
        f"  --model /path/to/{output_name}.pth \\",
        "  --input source.wav \\",
        "  --output converted.wav \\",
        f"  --f0-method {cfg.f0_method} \\",
        "  --index-rate 0.75",
        "```",
        "",
        "When `--index` is omitted, the CLI matches an added index by model name. "
        "If `--index-rate` is greater than 0 and the index is missing, the CLI exits with an error.",
        "",
        "This worker is MIT licensed. Upstream RVC is MIT, © 2023 liujing04, 源文雨, Ftps.",
        "",
    ]
    return "\n".join(lines)


def push_to_hf(cfg, token: str, files: list[str], card: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=cfg.hf_repo_id, repo_type="model", private=cfg.hf_private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for path in files:
        ops.append(CommitOperationAdd(path_in_repo=os.path.basename(path), path_or_fileobj=path))
    api.create_commit(
        repo_id=cfg.hf_repo_id, repo_type="model", operations=ops,
        commit_message=f"RVC {cfg.version} {cfg.sample_rate} {cfg.model_name} ({cfg.epochs} ep)",
    )
    return f"https://huggingface.co/{cfg.hf_repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket
    return upload_file_to_bucket(file_name=os.path.basename(path), file_location=path,
                                 bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix)


def put_presigned(url: str, path: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}: {r.text[:200]}")
    return r.status_code


def deliver(cfg, run_name: str, output_name: str, pth_path: str | None, index_path: str | None,
            sample_path: str | None, train: dict[str, Any], dataset: dict[str, Any],
            token: str | None, persistent: bool) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {"persistent_storage": persistent}
    warnings: list[str] = []
    files: list[str] = []
    if pth_path:
        out["pth_file"] = os.path.basename(pth_path)
        out["pth_path"] = pth_path
        out["pth_size_mb"] = round(os.path.getsize(pth_path) / 2**20, 2)
        files.append(pth_path)
        if persistent:
            out["pth_volume_path"] = pth_path
    if index_path and os.path.isfile(index_path):
        out["index_file"] = os.path.basename(index_path)
        out["index_path"] = index_path
        out["index_size_mb"] = round(os.path.getsize(index_path) / 2**20, 2)
        files.append(index_path)
        if persistent:
            out["index_volume_path"] = index_path
    if sample_path and os.path.isfile(sample_path):
        out["sample_file"] = os.path.basename(sample_path)
        out["sample_path"] = sample_path
        files.append(sample_path)

    delivered = persistent and bool(files)
    card = model_card(cfg, output_name, train, dataset)
    card_dir = os.path.dirname(pth_path or index_path or ".")
    with open(os.path.join(card_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    if cfg.hf_repo_id and token and files:
        try:
            out["hf_repo_url"] = push_to_hf(cfg, token, files, card)
            delivered = True
        except Exception as exc:
            out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
            log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url and pth_path:
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, pth_path)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured() and files:
        try:
            prefix = f"rvc/{run_name}"
            for path in files:
                if path.endswith(".pth"):
                    key = "bucket_pth_url"
                elif path.endswith(".index"):
                    key = "bucket_index_url"
                else:
                    key = "bucket_sample_url"
                out[key] = upload_bucket(path, prefix)
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append(
            "the weights were not delivered anywhere persistent: attach a network volume, or set "
            "hf_repo_id (+HF_TOKEN), upload_url, or BUCKET_* env vars. They are lost when the worker stops."
        )
    return out, warnings
