import json
import os
import random
from pathlib import Path

import pytest

from src.filelist import (
    FilelistError,
    build_filelist_lines,
    official_config_relpath,
    overlapping_names,
    write_filelist_and_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _touch(path: Path, data: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fake_exp(root: Path, names=("clip_0", "clip_1"), version="v2"):
    exp = root / "logs" / "zxowl"
    feat = "3_feature768" if version == "v2" else "3_feature256"
    for n in names:
        _touch(exp / "0_gt_wavs" / f"{n}.wav")
        _touch(exp / feat / f"{n}.npy")
        _touch(exp / "2a_f0" / f"{n}.wav.npy")
        _touch(exp / "2b-f0nsf" / f"{n}.wav.npy")
    # Official configs used by write_config_json
    cfg_src = FIXTURES / "configs"
    dest = root / "configs"
    if not dest.exists():
        os.makedirs(dest / "v1", exist_ok=True)
        os.makedirs(dest / "v2", exist_ok=True)
        for p in (cfg_src / "v1").glob("*.json"):
            (dest / "v1" / p.name).write_text(p.read_text())
        for p in (cfg_src / "v2").glob("*.json"):
            (dest / "v2" / p.name).write_text(p.read_text())
    return exp


def test_config_path_follows_webui():
    # WebUI: if version == v1 or sr == 40k: v1/<sr>.json else v2/<sr>.json
    assert official_config_relpath("v2", "40k") == os.path.join("v1", "40k.json")
    assert official_config_relpath("v1", "40k") == os.path.join("v1", "40k.json")
    assert official_config_relpath("v2", "48k") == os.path.join("v2", "48k.json")
    assert official_config_relpath("v2", "32k") == os.path.join("v2", "32k.json")
    assert official_config_relpath("v1", "32k") == os.path.join("v1", "32k.json")
    assert not (FIXTURES / "configs" / "v2" / "40k.json").exists()


def test_overlapping_names_require_all_branches(tmp_path):
    root = tmp_path / "rvc"
    exp = _fake_exp(root, names=("keep", "partial"))
    # Drop F0 for "partial" so it must be excluded (WebUI set intersection).
    os.remove(exp / "2a_f0" / "partial.wav.npy")
    names = overlapping_names(str(exp), "v2", if_f0=True)
    assert names == {"keep"}


def test_filelist_format_and_mute_rows(tmp_path):
    root = tmp_path / "rvc"
    _fake_exp(root, names=("aaa", "bbb"))
    rng = random.Random(0)
    lines = build_filelist_lines(
        str(root), str(root / "logs" / "zxowl"), "v2", "40k", if_f0=True, speaker_id=0, rng=rng,
    )
    assert len(lines) == 4  # 2 clips + 2 mute
    mute = [ln for ln in lines if "/logs/mute/" in ln]
    assert len(mute) == 2 and mute[0] == mute[1]
    assert mute[0].endswith("|0")
    assert "mute40k.wav" in mute[0]
    assert "3_feature768/mute.npy" in mute[0]
    assert "2a_f0/mute.wav.npy" in mute[0]
    assert "2b-f0nsf/mute.wav.npy" in mute[0]
    data = [ln for ln in lines if "/logs/mute/" not in ln]
    for ln in data:
        parts = ln.split("|")
        assert len(parts) == 5
        wav, feat, coarse, cont, spk = parts
        assert wav.endswith(".wav") and feat.endswith(".npy")
        assert coarse.endswith(".wav.npy") and cont.endswith(".wav.npy")
        assert spk == "0"
        stem = os.path.basename(wav)[:-4]
        assert stem in {"aaa", "bbb"}
        assert os.path.basename(feat) == f"{stem}.npy"


def test_write_filelist_and_v2_40k_config(tmp_path):
    root = tmp_path / "rvc"
    _fake_exp(root)
    fl, cfg_path, n = write_filelist_and_config(
        str(root), "zxowl", "v2", "40k", if_f0=True, speaker_id=0, rng=random.Random(1),
    )
    assert n == 2
    assert os.path.isfile(fl) and os.path.isfile(cfg_path)
    cfg = json.loads(Path(cfg_path).read_text())
    assert "speaker_info" not in cfg
    assert cfg["data"]["sampling_rate"] == 40000
    # Sorted keys + trailing newline, as WebUI writes it.
    raw = Path(cfg_path).read_text()
    assert raw.endswith("\n")
    assert list(cfg.keys()) == sorted(cfg.keys())


def test_empty_intersection_raises(tmp_path):
    root = tmp_path / "rvc"
    exp = root / "logs" / "empty"
    (exp / "0_gt_wavs").mkdir(parents=True)
    (exp / "3_feature768").mkdir(parents=True)
    (exp / "2a_f0").mkdir(parents=True)
    (exp / "2b-f0nsf").mkdir(parents=True)
    with pytest.raises(FilelistError):
        build_filelist_lines(str(root), str(exp), "v2", "40k")
