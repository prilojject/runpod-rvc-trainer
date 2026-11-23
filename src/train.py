"""Headless wrapper around the official RVC CLI (docs/en/cli.md on the pinned SHA).

All commands run from the RVC clone root. Gradio webui.py is never started.
On one GPU we do not launch extra extract processes or a DDP trainer.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable

from .filelist import write_filelist_and_config
from .models import pretrained_files
from .paths import RVC_DIR
from .secret_utils import redact

log = logging.getLogger(__name__)

ProgressFn = Callable[[dict], None]


class TrainingError(RuntimeError):
    pass


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None, "vram_gb": None, "compute_capability": None, "supported": None}
    try:
        import torch
    except Exception:
        return info
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}.{minor}"
    # Official image / Hub listing exclude Blackwell MIG slices; Ada/Ampere 24/48 GB are the target.
    unsupported = major >= 10  # sm_100+ is not in this worker's supported gpuIds
    info.update({
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / 2**30, 2),
        "compute_capability": cc,
        "supported": not unsupported,
        "cuda": torch.version.cuda,
        "is_half": True,
    })
    return info


def official_pythonpath(rvc_dir: str = RVC_DIR) -> str:
    parts = [rvc_dir]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def run_command(
    cmd: list[str],
    log_path: str,
    progress: ProgressFn | None,
    stage: str,
    timeout_s: int | None = None,
    extra_env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = official_pythonpath()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log.info("running %s: %s", stage, " ".join(cmd))
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n===== {stage} =====\n{' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd or RVC_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise TrainingError(f"{stage} failed to start: {redact(str(exc))}") from exc
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                if progress and should_forward(line):
                    progress({"stage": stage, "line": line.strip()[:240]})
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise TrainingError(f"{stage} exceeded TRAIN_TIMEOUT_SECONDS") from exc
    elapsed = round(time.time() - started, 2)
    if code != 0:
        tail = _tail(log_path, 40)
        raise TrainingError(f"{stage} exited {code} after {elapsed}s\n{tail}")
    return {"stage": stage, "exit_code": code, "seconds": elapsed, "cmd": cmd}


def should_forward(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(re.search(r"epoch|train|index|loss|error|warning|saved|preprocess|extract", s, re.I))


def _tail(path: str, n: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return redact("".join(lines[-n:]))
    except Exception:
        return ""


def preprocess_command(dataset_dir: str, sample_rate_hz: int, exp_name: str, workers: int = 4) -> list[str]:
    exp_dir = os.path.join("logs", exp_name)
    return [
        "python", "train/preprocess.py",
        dataset_dir, str(sample_rate_hz), str(max(1, workers)), exp_dir, "False", "3.7",
    ]


def extract_f0_command(exp_name: str, f0_method: str, is_half: bool = True) -> list[str]:
    exp_dir = os.path.join("logs", exp_name)
    if f0_method == "rmvpe":
        # CUDA RMVPE, one process, GPU 0, half-precision (docs/en/cli.md).
        return [
            "python", "train/dataset/extract_f0.py",
            "cuda", "1", "0", "0", exp_dir, "true" if is_half else "false",
        ]
    # CPU PM fallback.
    return ["python", "train/dataset/extract_f0.py", "cpu", exp_dir, "4", "pm"]


def extract_hubert_command(exp_name: str, version: str, is_half: bool = True) -> list[str]:
    exp_dir = os.path.join("logs", exp_name)
    return [
        "python", "train/dataset/extract_hubert_feature.py",
        "cuda:0", "1", "0", "0", exp_dir, version, "true" if is_half else "false",
    ]


def train_command(
    exp_name: str,
    sample_rate: str,
    version: str,
    epochs: int,
    save_every: int,
    batch_size: int,
    pretrain_g: str,
    pretrain_d: str,
) -> list[str]:
    # 1 GPU, no DDP launcher; -l 1 keeps only the latest large ckpt; -sw 1 writes the small infer model.
    return [
        "python", "train/train.py",
        "-e", exp_name,
        "-sr", sample_rate,
        "-f0", "1",
        "-bs", str(batch_size),
        "-g", "0",
        "-te", str(epochs),
        "-se", str(save_every),
        "-pg", pretrain_g,
        "-pd", pretrain_d,
        "-l", "1",
        "-c", "0",
        "-sw", "1",
        "-v", version,
    ]


def index_command(exp_name: str, version: str, workers: int = 4) -> list[str]:
    return [
        "python", "train/train_index.py",
        exp_name, version, "assets/indices", str(max(1, workers)), "single",
    ]


def infer_command(
    model_path: str,
    input_wav: str,
    output_wav: str,
    f0_method: str,
    index_rate: float,
    pitch: int,
    index_path: str | None = None,
) -> list[str]:
    cmd = [
        "python", "infer/cli.py",
        "--model", model_path,
        "--input", input_wav,
        "--output", output_wav,
        "--f0-method", f0_method,
        "--index-rate", str(index_rate),
        "--pitch", str(pitch),
    ]
    if index_path:
        cmd.extend(["--index", index_path])
    return cmd


def find_small_model(exp_name: str, rvc_dir: str = RVC_DIR) -> str | None:
    # Official savee() writes assets/weights/<exp_name>.pth
    direct = os.path.join(rvc_dir, "assets", "weights", f"{exp_name}.pth")
    if os.path.isfile(direct) and os.path.getsize(direct) > 1024:
        return direct
    weights = os.path.join(rvc_dir, "assets", "weights")
    if not os.path.isdir(weights):
        return None
    candidates = [
        os.path.join(weights, n) for n in os.listdir(weights)
        if n.endswith(".pth") and os.path.isfile(os.path.join(weights, n))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def find_added_index(exp_name: str, version: str, rvc_dir: str = RVC_DIR) -> str | None:
    # Official train_index.py links to assets/indices/<exp_name>_added_IVF*_..._<exp>_<ver>.index
    # and also writes the added index under logs/<exp>/.
    roots = [
        os.path.join(rvc_dir, "assets", "indices"),
        os.path.join(rvc_dir, "logs", exp_name),
    ]
    matches: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.endswith(".index"):
                continue
            if "added_IVF" in name and exp_name in name and version in name:
                matches.append(os.path.join(root, name))
            elif name.startswith(f"{exp_name}_") and "added_IVF" in name:
                matches.append(os.path.join(root, name))
    if not matches:
        # last resort: any added index in those folders
        for root in roots:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                if "added_IVF" in name and name.endswith(".index"):
                    matches.append(os.path.join(root, name))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def copy_artifacts(pth_src: str, index_src: str | None, run_dir: str, model_name: str) -> dict[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    pth_dst = os.path.join(run_dir, f"{model_name}.pth")
    shutil.copy2(pth_src, pth_dst)
    out = {"pth": pth_dst}
    if index_src and os.path.isfile(index_src):
        # Keep a stable, easy-to-find name next to the .pth.
        idx_dst = os.path.join(run_dir, f"{model_name}.index")
        shutil.copy2(index_src, idx_dst)
        out["index"] = idx_dst
        out["index_original_name"] = os.path.basename(index_src)
    return out


def maybe_infer_sample(
    cfg,
    pth_path: str,
    index_path: str | None,
    source_wav: str,
    out_dir: str,
    log_path: str,
    progress: ProgressFn | None,
    timeout_s: int | None,
) -> str | None:
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "sample.wav")
    cmd = infer_command(
        pth_path, source_wav, dest, cfg.f0_method, cfg.index_rate, cfg.pitch,
        index_path=index_path,
    )
    # If the index is missing, official CLI errors when index-rate > 0. Fall back to 0.
    if not index_path:
        cmd = infer_command(pth_path, source_wav, dest, cfg.f0_method, 0.0, cfg.pitch, index_path=None)
    run_command(cmd, log_path, progress, "infer_sample", timeout_s=timeout_s)
    return dest if os.path.isfile(dest) else None


def run_pipeline(
    cfg,
    dataset_dir: str,
    exp_name: str,
    log_path: str,
    progress: ProgressFn,
    timeout_s: int | None,
    workers: int = 4,
) -> dict[str, Any]:
    training: dict[str, Any] = {
        "version": cfg.version,
        "sample_rate": cfg.sample_rate,
        "epochs": cfg.epochs,
        "save_every_epoch": cfg.save_every_epoch,
        "batch_size": cfg.batch_size,
        "f0_method": cfg.f0_method,
        "is_half": True,
        "stages": {},
    }
    g_rel, d_rel = pretrained_files(cfg.version, cfg.sample_rate)
    pretrain_g = os.path.join("assets", g_rel)
    pretrain_d = os.path.join("assets", d_rel)

    progress({"stage": "preprocess"})
    training["stages"]["preprocess"] = run_command(
        preprocess_command(dataset_dir, cfg.sample_rate_hz, exp_name, workers),
        log_path, progress, "preprocess", timeout_s=timeout_s,
    )

    progress({"stage": "extract_f0"})
    training["stages"]["extract_f0"] = run_command(
        extract_f0_command(exp_name, cfg.f0_method, is_half=True),
        log_path, progress, "extract_f0", timeout_s=timeout_s,
    )

    progress({"stage": "extract_hubert"})
    training["stages"]["extract_hubert"] = run_command(
        extract_hubert_command(exp_name, cfg.version, is_half=True),
        log_path, progress, "extract_hubert", timeout_s=timeout_s,
    )

    progress({"stage": "filelist"})
    filelist_path, config_path, n_rows = write_filelist_and_config(
        RVC_DIR, exp_name, cfg.version, cfg.sample_rate, if_f0=True, speaker_id=0,
    )
    training["filelist_rows"] = n_rows
    training["filelist"] = filelist_path
    training["config_json"] = config_path
    if n_rows < 1:
        raise TrainingError("filelist is empty after extract; not enough valid clips")
    # Avoid a batch larger than the (tiny) training set — official DataLoader would stall or fail.
    batch = min(cfg.batch_size, max(1, n_rows))
    if batch != cfg.batch_size:
        training["batch_size_adjusted"] = batch
        log.warning("batch_size lowered from %s to %s to match filelist size", cfg.batch_size, batch)
    training["batch_size"] = batch

    progress({"stage": "train"})
    training["stages"]["train"] = run_command(
        train_command(
            exp_name, cfg.sample_rate, cfg.version, cfg.epochs,
            cfg.save_every_epoch, batch, pretrain_g, pretrain_d,
        ),
        log_path, progress, "train", timeout_s=timeout_s,
    )

    progress({"stage": "index"})
    training["stages"]["index"] = run_command(
        index_command(exp_name, cfg.version, workers),
        log_path, progress, "index", timeout_s=timeout_s,
    )
    return training
