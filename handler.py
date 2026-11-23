"""Runpod Serverless entry point: headless RVC training (voice → voice, .pth + .index).

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_url": "https://.../voice.zip"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset  # noqa: E402
from src.paths import RVC_DIR, configure_caches, has_volume, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("rvc-worker")

# Official default on CUDA: fp16 (is_half=true). Do not force bf16.
IS_HALF = True


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Always start the next job on a fresh worker (frees GPU memory, drops partial downloads).
    result["refresh_worker"] = True
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    output_name = cfg.model_name
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/rvc/work", run_name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "model_name": output_name,
        "version": cfg.version,
        "sample_rate": cfg.sample_rate,
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append(
            "no network volume: pretrained weights are re-downloaded on every cold start "
            "(attach one to cache them under /runpod-volume/rvc/models)"
        )

    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, work)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if all(v.get("ok") for v in access.values()) else "error",
            "model_access": access,
            "model_download_gb": round(sum(v.get("size_gb") or 0 for v in access.values() if v.get("ok")), 2),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "pretrained files not reachable: " + "; ".join(
                f"{k}: {v.get('error')}" for k, v in access.items() if not v.get("ok"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    from src.models import ModelError, ensure_models
    from src.train import (
        TrainingError, copy_artifacts, find_added_index, find_small_model, gpu_info,
        maybe_infer_sample, run_pipeline,
    )

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu["gpu"]:
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}
    if gpu.get("supported") is False:
        return {"error": f"GPU {gpu['gpu']} (compute {gpu.get('compute_capability')}) is not supported by this "
                         "image's PyTorch/CUDA build (Blackwell MIG GPUs are excluded in hub.json); "
                         "redeploy on Ampere/Ada (RTX 4090, L40S, A6000 Ada, A6000, A40)",
                **{k: result[k] for k in ("run_name", "config")}, "gpu": gpu}

    try:
        _paths, minfo = ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}
    result["models"] = minfo

    os.makedirs(run_dir, exist_ok=True)
    exp_name = output_name
    exp_dir = os.path.join(RVC_DIR, "logs", exp_name)
    shutil.rmtree(exp_dir, ignore_errors=True)
    os.makedirs(exp_dir, exist_ok=True)
    # Official savee() writes assets/weights/<exp>.pth; start from a clean slot.
    stale_pth = os.path.join(RVC_DIR, "assets", "weights", f"{exp_name}.pth")
    if os.path.isfile(stale_pth) or os.path.islink(stale_pth):
        os.remove(stale_pth)
    log_path = os.path.join(run_dir, "train.log")
    timeout_s = int(os.environ.get("TRAIN_TIMEOUT_SECONDS", "0")) or None
    workers = min(4, os.cpu_count() or 2)

    try:
        training = run_pipeline(
            cfg, dataset_dir=report.audio_dir, exp_name=exp_name,
            log_path=log_path, progress=progress, timeout_s=timeout_s, workers=workers,
        )
    except TrainingError as exc:
        return {"error": redact(str(exc))[:4000], "run_name": run_name, "gpu": gpu, "log_path": log_path}
    except Exception as exc:
        log.error("training crashed: %s", redact(traceback.format_exc()))
        return {"error": f"training crashed: {redact(str(exc)).splitlines()[0][:500]}", "run_name": run_name}

    result["training"] = training
    pth_src = find_small_model(exp_name)
    index_src = find_added_index(exp_name, cfg.version)
    if not pth_src:
        return {"error": "training finished but no inference-ready .pth was written to assets/weights/",
                "run_name": run_name, "training": training, "log_path": log_path}
    if not index_src:
        result["warnings"].append("training finished but no .index was written; the listing is incomplete without it")

    copied = copy_artifacts(pth_src, index_src, run_dir, output_name)
    pth_path = copied.get("pth")
    index_path = copied.get("index")

    sample_path = None
    if cfg.num_samples > 0 and pth_path:
        try:
            source = cfg.sample_audio_url
            local_src = os.path.join(report.audio_dir, report.example_files[0]) if report.example_files else ""
            if source:
                # Reuse the dataset downloader for a single file by wrapping is messy; prefer local clip.
                import requests as _req
                dest = os.path.join(work, "sample_src.wav")
                with _req.get(source, timeout=(15, 120), stream=True) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                local_src = dest
            if local_src and os.path.isfile(local_src):
                sample_path = maybe_infer_sample(
                    cfg, pth_path=pth_path, index_path=index_path, source_wav=local_src,
                    out_dir=os.path.join(run_dir, "samples"),
                    log_path=log_path, progress=progress, timeout_s=timeout_s,
                )
                if sample_path:
                    shutil.copy2(sample_path, os.path.join(run_dir, "sample.wav"))
                    sample_path = os.path.join(run_dir, "sample.wav")
        except Exception as exc:
            result["warnings"].append(f"sample inference failed: {redact(str(exc)).splitlines()[0][:240]}")

    from src.outputs import deliver, encode_file_base64
    progress({"stage": "uploading"})
    artifacts, warns = deliver(
        cfg, run_name, output_name, pth_path, index_path, sample_path,
        training, report.as_dict(), token, persistent,
    )
    result["warnings"] += warns
    if cfg.return_sample_base64 and sample_path:
        b64 = encode_file_base64(sample_path)
        if b64:
            artifacts["sample_base64"] = b64
        else:
            result["warnings"].append("return_sample_base64 skipped: sample is larger than 20 MiB")
    result["artifacts"] = artifacts
    result.update({"status": "completed", "elapsed_seconds": round(time.time() - started, 1)})
    shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
