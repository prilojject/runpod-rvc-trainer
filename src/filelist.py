"""Build logs/<exp>/filelist.txt and config.json with the same logic as the official WebUI.

Source: RVC-Project webui.py::run_train_model on SHA 81eed5e8f68b6bed1789f682fe78cdd324495afc.
There is no official prepare_filelist.py; a pure CLI workflow must do this before train.py.

Single-speaker F0 record (docs/en/cli.md):

    wav_path|HuBERT_feature_path|coarse_F0_path|continuous_F0_path|speaker_ID

Directory names after extract on this SHA:

    0_gt_wavs / 2a_f0 / 2b-f0nsf / 3_feature768 (v2) or 3_feature256 (v1)

config.json: WebUI uses configs/v1/<sr>.json when version is v1 *or* sr is 40k
(there is no configs/v2/40k.json); otherwise configs/v2/<sr>.json.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


class FilelistError(RuntimeError):
    pass


def feature_dir_name(version: str) -> str:
    return "3_feature256" if version == "v1" else "3_feature768"


def feature_dim(version: str) -> int:
    return 256 if version == "v1" else 768


def official_config_relpath(version: str, sample_rate: str) -> str:
    # Exact WebUI branch: if version19 == "v1" or sr2 == "40k": v1/<sr>.json else v2/<sr>.json
    if version == "v1" or sample_rate == "40k":
        return os.path.join("v1", f"{sample_rate}.json")
    return os.path.join("v2", f"{sample_rate}.json")


def stem_set(directory: str) -> set[str]:
    # WebUI uses name.split(".")[0], not a full suffix strip.
    return {name.split(".")[0] for name in os.listdir(directory)}


def overlapping_names(exp_dir: str, version: str, if_f0: bool) -> set[str]:
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feat_dir = os.path.join(exp_dir, feature_dir_name(version))
    if not os.path.isdir(gt_wavs_dir) or not os.path.isdir(feat_dir):
        return set()
    names = stem_set(gt_wavs_dir) & stem_set(feat_dir)
    if if_f0:
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
        if not os.path.isdir(f0_dir) or not os.path.isdir(f0nsf_dir):
            return set()
        names &= stem_set(f0_dir) & stem_set(f0nsf_dir)
    return names


def _win_escape(path: str) -> str:
    # Kept for parity with webui.py (no-op on Linux paths).
    return path.replace("\\", "\\\\")


def build_filelist_lines(
    rvc_dir: str,
    exp_dir: str,
    version: str,
    sample_rate: str,
    if_f0: bool = True,
    speaker_id: int = 0,
    rng: random.Random | None = None,
) -> list[str]:
    """Return shuffled filelist lines, including the two official mute rows."""
    names = overlapping_names(exp_dir, version, if_f0)
    if not names:
        raise FilelistError(
            "no overlapping wav/feature/F0 files to train on; finish preprocess and extract first"
        )

    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feat_dir = os.path.join(exp_dir, feature_dir_name(version))
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    fea_dim = feature_dim(version)
    opt: list[str] = []
    for name in sorted(names):
        if if_f0:
            line = "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
                _win_escape(gt_wavs_dir),
                name,
                _win_escape(feat_dir),
                name,
                _win_escape(f0_dir),
                name,
                _win_escape(f0nsf_dir),
                name,
                speaker_id,
            )
        else:
            line = "%s/%s.wav|%s/%s.npy|%s" % (
                _win_escape(gt_wavs_dir),
                name,
                _win_escape(feat_dir),
                name,
                speaker_id,
            )
        opt.append(line)

    now_dir = rvc_dir
    if if_f0:
        mute_line = (
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|"
            "%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (now_dir, sample_rate, now_dir, fea_dim, now_dir, now_dir, speaker_id)
        )
    else:
        mute_line = "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s" % (
            now_dir, sample_rate, now_dir, fea_dim, speaker_id
        )
    # Official WebUI appends the mute row twice.
    opt.append(mute_line)
    opt.append(mute_line)

    shuffler = rng.shuffle if rng is not None else random.shuffle
    shuffler(opt)
    return opt


def load_official_config(rvc_dir: str, version: str, sample_rate: str) -> dict[str, Any]:
    rel = official_config_relpath(version, sample_rate)
    path = os.path.join(rvc_dir, "configs", rel)
    if not os.path.isfile(path):
        raise FilelistError(f"official config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_config_json(
    rvc_dir: str,
    exp_dir: str,
    version: str,
    sample_rate: str,
) -> dict[str, Any]:
    config_save_path = os.path.join(exp_dir, "config.json")
    if Path(config_save_path).exists():
        with open(config_save_path, encoding="utf-8") as fh:
            config_data = json.load(fh)
    else:
        config_data = load_official_config(rvc_dir, version, sample_rate)
    # Single-speaker path: WebUI pops speaker_info if present.
    config_data.pop("speaker_info", None)
    os.makedirs(exp_dir, exist_ok=True)
    with open(config_save_path, "w", encoding="utf-8") as fh:
        json.dump(config_data, fh, ensure_ascii=False, indent=4, sort_keys=True)
        fh.write("\n")
    return config_data


def write_filelist_and_config(
    rvc_dir: str,
    exp_name: str,
    version: str,
    sample_rate: str,
    if_f0: bool = True,
    speaker_id: int = 0,
    rng: random.Random | None = None,
) -> tuple[str, str, int]:
    """Write filelist.txt and config.json under logs/<exp_name>.

    Returns (filelist_path, config_path, n_training_rows_without_mute).
    """
    exp_dir = os.path.join(rvc_dir, "logs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    lines = build_filelist_lines(
        rvc_dir, exp_dir, version, sample_rate, if_f0=if_f0, speaker_id=speaker_id, rng=rng,
    )
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    with open(filelist_path, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))
    write_config_json(rvc_dir, exp_dir, version, sample_rate)
    n_train = max(0, len(lines) - 2)
    return filelist_path, os.path.join(exp_dir, "config.json"), n_train
