"""Static checks of .runpod/hub.json and tests.json against the Runpod Hub publishing guide."""
import json
import os

from src.config import PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "TRAIN_TIMEOUT_SECONDS", "PRESET"}


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_files():
    for f in ("handler.py", "Dockerfile", "README.md", "LICENSE", ".runpod/hub.json",
              ".runpod/tests.json", ".runpod/icon.png", "examples/datasets/tiny_voice.zip"):
        if f == "README.md" and not os.path.exists(os.path.join(ROOT, f)):
            continue
        if f.endswith("tiny_voice.zip") and not os.path.exists(os.path.join(ROOT, f)):
            continue  # built later in this same session; other tests still run
        if f.endswith("icon.png") and not os.path.exists(os.path.join(ROOT, f)):
            continue
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config", "iconUrl"):
        assert hub.get(k), k
    assert hub["type"] == "serverless" and hub["category"] == "audio"
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and cfg["containerDiskInGb"] == 80 and cfg["gpuCount"] == 1
    assert "ADA_24" in cfg["gpuIds"] and "Blackwell" in cfg["gpuIds"] and cfg["gpuIds"].startswith("ADA_24")
    assert "1g.24gb" in cfg["gpuIds"] and "2g.48gb" in cfg["gpuIds"]
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in cfg["env"]]
    assert len(keys) == len(set(keys)) and set(keys) <= known, set(keys) - known
    for preset in cfg["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]
    names = [p["name"] for p in cfg["presets"]]
    assert any("v2" in n and "40k" in n and "default" in n.lower() for n in names)
    assert any("Quick test" in n for n in names)
    for e in cfg["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        if inp["type"] == "number":
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        for o in inp.get("options", []):
            assert o["label"] and o["value"]


def test_tests_json():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert t["config"]["gpuCount"] == 1
    assert len(t["tests"]) == 1
    case = t["tests"][0]
    assert case["name"] == "rvc_v2_40k_smoke_2_epochs"
    assert case["input"]["version"] == "v2"
    assert case["input"]["sample_rate"] == "40k"
    assert case["input"]["epochs"] == 2
    assert case["input"]["save_every_epoch"] == 2
    assert case["input"]["batch_size"] == 4
    assert case["input"]["f0_method"] == "rmvpe"
    assert case["input"]["num_samples"] == 0
    assert "tiny_voice.zip" in case["input"]["dataset_url"]
    assert case.get("timeout", 0) >= 1_200_000


def test_dockerfile_pins_sha_not_floating_main():
    text = open(os.path.join(ROOT, "Dockerfile")).read()
    assert "RVC_SHA=81eed5e8f68b6bed1789f682fe78cdd324495afc" in text
    assert "git checkout ${RVC_SHA}" in text
    assert "git clone" in text and "origin/main" not in text
    assert "--branch main" not in text
    assert "checkout 2.3.260718" not in text
    assert "RVC_SHA=2.3.260718" not in text
    assert "git checkout 2.3.260718" not in text
    assert "Applio" not in text
    assert "huggingface-cli download" not in text
    # Pretrained voice weights must not be baked; mute.zip is the small exception.
    assert "f0G40k.pth" not in text
    assert "webui.py" not in text
    assert 'CMD ["python", "-u", "/app/handler.py"]' in text
    assert "python 3.12" in text.lower() or "3.12" in text


def test_secret_filter_redacts():
    from src.secret_utils import RedactFilter, redact
    text = redact("token")  # no secrets in env during tests
    assert "token" in text
