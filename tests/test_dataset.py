import io
import os
import wave
import zipfile
import struct

import pytest

from src.dataset import DatasetError, _extract, _safe_member, flatten_audio, prepare_dataset
from src.config import resolve_config


def _wav_bytes(seconds=0.5, rate=16000, freq=440):
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * __import__("math").sin(2 * 3.14159 * freq * i / rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


def test_zip_slip_is_rejected(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("ok.wav", b"RIFF")
        z.writestr("../escape.wav", b"nope")
        z.writestr("/tmp/abs.wav", b"nope")
    dest = tmp_path / "out"
    _extract(str(zpath), str(dest))
    # Safe member is kept (even if not a valid wav — extract doesn't parse audio).
    found = []
    for root, _, files in os.walk(dest):
        found.extend(os.path.join(root, f) for f in files)
    rels = [os.path.relpath(p, dest) for p in found]
    assert any(name.endswith("ok.wav") for name in rels)
    assert not any("escape.wav" in name for name in rels)
    assert not (tmp_path / "escape.wav").exists()


def test_safe_member():
    base = "/tmp/rvc-safe"
    assert _safe_member(base, "a/b.wav")
    assert _safe_member(base, "../x.wav") is None
    assert _safe_member(base, "/etc/passwd") is None


def test_flatten_audio_and_prepare(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    (raw / "sub" / "a.wav").write_bytes(_wav_bytes(1.0))
    (raw / "b.flac").write_bytes(b"not-really-flac")  # duration unknown, still kept
    (raw / "__MACOSX" / "._a.wav").parent.mkdir(exist_ok=True)
    (raw / "__MACOSX" / "._a.wav").write_bytes(b"junk")
    dest = tmp_path / "audio"
    used = flatten_audio(str(raw), str(dest))
    names = {os.path.basename(p) for p in used}
    assert "a.wav" in names and "b.flac" in names
    assert not any(p.startswith(".") or "__MACOSX" in p for p in names)


def test_prepare_rejects_empty_archive(tmp_path, monkeypatch):
    zpath = tmp_path / "empty.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("readme.txt", "no audio")

    def fake_download(url, dest):
        import shutil
        shutil.copy2(zpath, dest)
        return dest

    monkeypatch.setattr("src.dataset._download", fake_download)
    cfg = resolve_config({"dataset_url": "https://example.com/empty.zip"})
    with pytest.raises(DatasetError, match="no wav/flac/mp3"):
        prepare_dataset(cfg, str(tmp_path / "work"))
