"""Input schema for the headless RVC trainer.

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

SUPPORTED_VERSIONS = ("v2", "v1")
SAMPLE_RATES = ("40k", "32k", "48k")
F0_METHODS = ("rmvpe", "pm")
# Official CLI default is 200 epochs (quality). Smoke tests use 2.
EPOCH_DEFAULT = 200
EPOCH_MAX = 1200
BATCH_DEFAULT = 8
SAVE_EVERY_DEFAULT = 5

PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "version": "v2",
        "sample_rate": "40k",
        "epochs": EPOCH_DEFAULT,
        "save_every_epoch": SAVE_EVERY_DEFAULT,
        "batch_size": BATCH_DEFAULT,
        "f0_method": "rmvpe",
    },
    "quick": {
        "version": "v2",
        "sample_rate": "40k",
        "epochs": 2,
        "save_every_epoch": 2,
        "batch_size": 4,
        "f0_method": "rmvpe",
        "num_samples": 0,
    },
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


PARAMS: list[Param] = [
    Param("version", str, "v2", "VERSION", choices=SUPPORTED_VERSIONS,
          help="RVC architecture. v2 is the default. v1 is optional and uses pretrained/ rather than pretrained_v2/."),
    Param("sample_rate", str, "40k", "SAMPLE_RATE", choices=SAMPLE_RATES,
          help="Training sample rate. 40k is the default v2 pretrain. Also 32k and 48k."),
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL to a .zip / .tar(.gz) of wav/flac/mp3 clips of one target speaker. Transcripts are not needed."),
    Param("model_name", str, "zxowl", "MODEL_NAME",
          help="Experiment / weight name. The inference-ready file is written as assets/weights/<model_name>.pth."),
    Param("run_name", str, "", "RUN_NAME",
          help="Folder name for outputs. Default: the Runpod job id."),
    Param("preset", str, "default", "PRESET", choices=tuple(PRESETS),
          help="Named defaults. 'default' is v2/40k/200 epochs; 'quick' is a 2-epoch smoke test."),
    Param("epochs", int, EPOCH_DEFAULT, "EPOCHS", min=1, max=EPOCH_MAX,
          help="Total training epochs (official default 200, max 1200). Use 2 for a smoke test."),
    Param("save_every_epoch", int, SAVE_EVERY_DEFAULT, "SAVE_EVERY_EPOCH", min=1, max=EPOCH_MAX,
          help="Checkpoint interval in epochs. Official CLI example uses 5."),
    Param("batch_size", int, BATCH_DEFAULT, "BATCH_SIZE", min=1, max=32,
          help="Per-GPU batch size. Official default is 8; the quick preset uses 4."),
    Param("f0_method", str, "rmvpe", "F0_METHOD", choices=F0_METHODS,
          help="Pitch extractor. Default rmvpe (CUDA, half-precision). pm is a CPU fallback."),
    Param("index_rate", float, 0.75, "INDEX_RATE", min=0.0, max=1.0,
          help="Retrieval blend for an optional sample conversion after training."),
    Param("pitch", int, 0, "PITCH", min=-24, max=24,
          help="Pitch shift in semitones for the optional sample conversion."),
    Param("num_samples", int, 0, "NUM_SAMPLES", min=0, max=4,
          help="Optional short conversions after training (0 = skip). Smoke tests use 0."),
    Param("sample_audio_url", str, "", "SAMPLE_AUDIO_URL",
          help="Optional source wav URL for a sample conversion. Otherwise one clip from the dataset is used."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO",
          help="Upload the .pth + .index + model card here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the .pth file."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET",
          help="If BUCKET_* env vars are set, upload the artifacts and return URLs."),
    Param("return_sample_base64", bool, False, "RETURN_SAMPLE_BASE64",
          help="Include a short sample wav as base64 in the job output (only if num_samples > 0)."),
    Param("seed", int, 1234, "SEED", min=0),
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate the dataset, check that pretrained weights are reachable. No GPU work."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "exp_name": "model_name",
    "experiment": "model_name",
    "name": "model_name",
    "sr": "sample_rate",
    "sampleRate": "sample_rate",
    "total_epoch": "epochs",
    "total_epochs": "epochs",
    "save_every": "save_every_epoch",
    "f0": "f0_method",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
    "return_samples_base64": "return_sample_base64",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def sample_rate_hz(self) -> int:
        return {"32k": 32000, "40k": 40000, "48k": 48000}[self.values["sample_rate"]]

    @property
    def if_f0(self) -> int:
        return 1

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        return out


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def _normalize_sample_rate(value: str) -> str:
    raw = value.strip().lower().replace("hz", "")
    aliases = {
        "40k": "40k", "40000": "40k", "40": "40k",
        "32k": "32k", "32000": "32k", "32": "32k",
        "48k": "48k", "48000": "48k", "48": "48k",
    }
    if raw not in aliases:
        raise ValueError(f"sample_rate: expected 32k, 40k or 48k, got {value!r}")
    return aliases[raw]


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = dict(os.environ if env is None else env)
    raw_in = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    # Apply aliases on the job input only.
    normalized_input: dict[str, Any] = {}
    for key, val in raw_in.items():
        name = ALIASES.get(key, key)
        if name in normalized_input and name != key:
            warnings.append(f"alias {key} ignored; {name} already set")
            continue
        normalized_input[name] = val

    preset_name = "default"
    for candidate, source in (
        (normalized_input.get("preset"), "input"),
        (env.get("PRESET"), "env"),
    ):
        if candidate is not None and str(candidate).strip() != "":
            name = str(candidate).strip().lower()
            if name not in PRESETS:
                errors.append(f"preset: unknown value {candidate!r} (use default or quick)")
            else:
                preset_name = name
            break

    preset = dict(PRESETS[preset_name])

    for param in PARAMS:
        raw = None
        source = "default"
        if param.name in normalized_input and normalized_input[param.name] is not None and normalized_input[param.name] != "":
            raw = normalized_input[param.name]
            source = "input"
        elif param.env and env.get(param.env) not in (None, ""):
            raw = env[param.env]
            source = "env"
        elif param.name in preset:
            raw = preset[param.name]
            source = f"preset:{preset_name}"
        else:
            raw = param.default
            source = "default"

        if raw is None:
            values[param.name] = param.default
            sources[param.name] = source
            continue
        try:
            val = _coerce(param, raw)
            if param.name == "sample_rate":
                val = _normalize_sample_rate(str(val))
            if param.name == "version":
                val = str(val).strip().lower()
                if val not in SUPPORTED_VERSIONS:
                    raise ValueError(f"version: expected v2 or v1, got {raw!r}")
            if param.name == "f0_method":
                val = str(val).strip().lower()
            if param.choices and val not in param.choices:
                raise ValueError(f"{param.name}: expected one of {param.choices}, got {val!r}")
            if param.min is not None and isinstance(val, (int, float)) and val < param.min:
                raise ValueError(f"{param.name}: {val} is below the minimum {param.min}")
            if param.max is not None and isinstance(val, (int, float)) and val > param.max:
                raise ValueError(f"{param.name}: {val} is above the maximum {param.max}")
            values[param.name] = val
            sources[param.name] = source
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ConfigError(errors)

    if values["save_every_epoch"] > values["epochs"]:
        values["save_every_epoch"] = values["epochs"]
        warnings.append(
            f"save_every_epoch lowered to {values['epochs']} so it does not exceed epochs"
        )

    if not values["dataset_url"]:
        errors.append("dataset_url is required")
    elif not URL_RE.match(values["dataset_url"]):
        errors.append("dataset_url must be an http(s) URL")

    if values["sample_audio_url"] and not URL_RE.match(values["sample_audio_url"]):
        errors.append("sample_audio_url must be an http(s) URL")

    if values["upload_url"] and not URL_RE.match(values["upload_url"]):
        errors.append("upload_url must be an http(s) URL")

    if values["hf_repo_id"] and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id must look like 'user/repo'")

    if values["run_name"] and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name may only contain letters, digits, '.', '_' and '-'")
    if not MODEL_NAME_RE.match(values["model_name"]):
        errors.append("model_name may only contain letters, digits, '.', '_' and '-'")

    # Tokens must never arrive in the job input.
    for banned in ("hf_token", "token", "authorization", "api_key", "secret", "password",
                   "aws_secret_access_key", "bucket_secret_access_key"):
        if banned in {k.lower() for k in raw_in}:
            errors.append("secrets must not be sent in the job input; use Runpod secrets / env")
            break

    if values["version"] == "v1":
        warnings.append(
            "version=v1 is optional; default training is v2. v1 uses assets/pretrained/ weights."
        )

    if errors:
        raise ConfigError(errors)

    values["preset"] = preset_name
    sources.setdefault("preset", "default" if preset_name == "default" else sources.get("preset", "input"))
    return ResolvedConfig(values=values, sources=sources, warnings=warnings)
