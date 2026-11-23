import pytest

from src.config import ConfigError, resolve_config


def test_defaults():
    cfg = resolve_config({"dataset_url": "https://example.com/voice.zip"})
    assert cfg.version == "v2"
    assert cfg.sample_rate == "40k"
    assert cfg.sample_rate_hz == 40000
    assert cfg.epochs == 200
    assert cfg.batch_size == 8
    assert cfg.f0_method == "rmvpe"
    assert cfg.if_f0 == 1
    assert cfg.num_samples == 0
    assert cfg.model_name == "zxowl"


def test_priority_input_over_env_over_preset():
    cfg = resolve_config(
        {"dataset_url": "https://example.com/a.zip", "epochs": 3, "preset": "quick"},
        env={"EPOCHS": "9", "VERSION": "v1", "DATASET_URL": "https://example.com/ignored.zip"},
    )
    assert cfg.epochs == 3 and cfg.sources["epochs"] == "input"
    assert cfg.dataset_url.endswith("a.zip") and cfg.sources["dataset_url"] == "input"
    assert cfg.version == "v1" and cfg.sources["version"] == "env"


def test_env_beats_preset():
    cfg = resolve_config(
        {"dataset_url": "https://example.com/a.zip", "preset": "quick"},
        env={"VERSION": "v1"},
    )
    assert cfg.version == "v1"
    assert cfg.epochs == 2  # from quick preset
    assert cfg.batch_size == 4


def test_quick_preset():
    cfg = resolve_config({"dataset_url": "https://example.com/a.zip", "preset": "quick"})
    assert cfg.epochs == 2 and cfg.save_every_epoch == 2 and cfg.batch_size == 4


def test_sample_rate_aliases():
    for raw, expect in (("40000", "40k"), ("40", "40k"), ("32k", "32k"), ("48000", "48k")):
        cfg = resolve_config({"dataset_url": "https://example.com/a.zip", "sample_rate": raw})
        assert cfg.sample_rate == expect


def test_save_every_capped():
    cfg = resolve_config({"dataset_url": "https://example.com/a.zip", "epochs": 2, "save_every_epoch": 10})
    assert cfg.save_every_epoch == 2
    assert any("save_every_epoch" in w for w in cfg.warnings)


def test_rejects_bad_url_and_repo():
    with pytest.raises(ConfigError):
        resolve_config({})
    with pytest.raises(ConfigError):
        resolve_config({"dataset_url": "ftp://x/y.zip"})
    with pytest.raises(ConfigError):
        resolve_config({"dataset_url": "https://example.com/a.zip", "hf_repo_id": "not-a-repo"})
    with pytest.raises(ConfigError):
        resolve_config({"dataset_url": "https://example.com/a.zip", "model_name": "has spaces"})


def test_rejects_secrets_in_input():
    with pytest.raises(ConfigError) as exc:
        resolve_config({"dataset_url": "https://example.com/a.zip", "hf_token": "should-not-be-here"})
    assert "secrets" in str(exc.value).lower()


def test_public_dict_redacts_upload_url():
    cfg = resolve_config({"dataset_url": "https://example.com/a.zip", "upload_url": "https://s3.example/put"})
    assert cfg.public_dict()["upload_url"] == "<presigned url>"
    assert cfg.upload_url.startswith("https://")
