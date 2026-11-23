"""Fetch, unpack and validate training audio for a single-speaker RVC dataset.

Transcripts are not required: this is voice conversion, not TTS.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import wave
import zipfile
from dataclasses import dataclass, field
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".wave", ".m4a")
MAX_DOWNLOAD_BYTES = 2 * 2**30
MAX_UNPACKED_BYTES = 2 * 2**30


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    source: str
    clips_found: int = 0
    clips_used: int = 0
    total_seconds: float = 0.0
    audio_dir: str = ""
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    example_files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["skipped"] = self.skipped[:20]
        return d


def _download(url: str, dest: str) -> str:
    log.info("downloading dataset archive")
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code != 200:
            raise DatasetError(f"dataset_url returned HTTP {r.status_code}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise DatasetError("dataset archive is larger than 2 GiB")
                f.write(chunk)
    return dest


def _safe_member(base: str, name: str) -> str | None:
    # Zip-slip: refuse members that would escape dest (including names with .. or abs paths).
    target = os.path.realpath(os.path.join(base, name))
    root = os.path.realpath(base)
    if not target.startswith(root + os.sep) and target != root:
        return None
    return target


def _extract(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    total = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                total += m.file_size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.filename)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                total += m.size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.name)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    continue
                with src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except tarfile.ReadError as exc:
        raise DatasetError("dataset_url must point to a .zip or .tar(.gz) archive") from exc


def _is_junk(name: str) -> bool:
    base = os.path.basename(name)
    rel = name.replace("\\", "/")
    return base.startswith(".") or base.startswith("__MACOSX") or "/__MACOSX/" in rel


def _duration_seconds(path: str) -> float | None:
    low = path.lower()
    if low.endswith((".wav", ".wave")):
        try:
            with wave.open(path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate() or 1
                return frames / float(rate)
        except Exception:
            return None
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _collect_audio(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        for f in files:
            if _is_junk(f) or not f.lower().endswith(AUDIO_EXTS):
                continue
            out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def flatten_audio(src_root: str, dest_dir: str) -> list[str]:
    """Copy audio into a single directory, as official single-speaker preprocessing expects."""
    os.makedirs(dest_dir, exist_ok=True)
    used: list[str] = []
    seen: set[str] = set()
    for path in _collect_audio(src_root):
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        dest_name = base
        i = 1
        while dest_name.lower() in seen:
            dest_name = f"{name}_{i}{ext}"
            i += 1
        seen.add(dest_name.lower())
        dest = os.path.join(dest_dir, dest_name)
        shutil.copy2(path, dest)
        used.append(dest)
    return used


def prepare_dataset(cfg, work: str) -> DatasetReport:
    raw_dir = os.path.join(work, "archive")
    audio_dir = os.path.join(work, "audio")
    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(work, "dataset.bin")
    try:
        _download(cfg.dataset_url, archive)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"failed to download dataset_url: {redact(str(exc)).splitlines()[0][:300]}") from exc
    try:
        _extract(archive, raw_dir)
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass

    found = _collect_audio(raw_dir)
    report = DatasetReport(source=cfg.dataset_url, clips_found=len(found), audio_dir=audio_dir)
    if not found:
        raise DatasetError("no wav/flac/mp3 clips found in the archive")

    copied = flatten_audio(raw_dir, audio_dir)
    used = 0
    total = 0.0
    for path in copied:
        dur = _duration_seconds(path)
        if dur is not None:
            if dur <= 0.05:
                report.skipped.append(f"{os.path.basename(path)}: too short ({dur:.2f}s)")
                os.remove(path)
                continue
            total += dur
        used += 1
        if len(report.example_files) < 5:
            report.example_files.append(os.path.basename(path))
    report.clips_used = used
    report.total_seconds = round(total, 2)
    if used < 1:
        raise DatasetError("no usable audio clips remained after filtering")
    if total and total < 8:
        report.warnings.append(
            f"only {total:.1f}s of audio; official RVC advice is about 10 minutes of clean speech. "
            "A short set is fine for a smoke test but will not produce a usable voice."
        )
    elif total and total < 60:
        report.warnings.append(
            f"only {total:.1f}s of audio; official RVC advice is about 10 minutes of clean, low-noise speech."
        )
    log.info("dataset ready: %s clips, %.1fs", used, total)
    return report
