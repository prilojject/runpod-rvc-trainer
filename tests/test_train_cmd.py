from src.train import (
    extract_f0_command,
    extract_hubert_command,
    index_command,
    infer_command,
    preprocess_command,
    train_command,
)


def test_preprocess_cli():
    cmd = preprocess_command("/data/audio", 40000, "zxowl", workers=4)
    assert cmd[:3] == ["python", "-P", "train/preprocess.py"]
    assert cmd[3:7] == ["/data/audio", "40000", "4", "logs/zxowl"]
    assert cmd[7] == "False" and cmd[8] == "3.7"


def test_extract_f0_rmvpe_cuda_half():
    cmd = extract_f0_command("zxowl", "rmvpe", is_half=True)
    assert cmd == ["python", "-P", "train/dataset/extract_f0.py", "cuda", "1", "0", "0", "logs/zxowl", "true"]


def test_extract_f0_pm_cpu():
    cmd = extract_f0_command("zxowl", "pm", is_half=True)
    assert cmd == ["python", "-P", "train/dataset/extract_f0.py", "cpu", "logs/zxowl", "4", "pm"]


def test_extract_hubert_v2_half():
    cmd = extract_hubert_command("zxowl", "v2", is_half=True)
    assert cmd == [
        "python", "-P", "train/dataset/extract_hubert_feature.py",
        "cuda:0", "1", "0", "0", "logs/zxowl", "v2", "true",
    ]


def test_train_cli_single_gpu_small_model():
    cmd = train_command("zxowl", "40k", "v2", 2, 2, 4,
                        "assets/pretrained_v2/f0G40k.pth", "assets/pretrained_v2/f0D40k.pth")
    assert cmd[0:3] == ["python", "-P", "train/train.py"]
    assert "-e" in cmd and cmd[cmd.index("-e") + 1] == "zxowl"
    assert "-sr" in cmd and cmd[cmd.index("-sr") + 1] == "40k"
    assert "-f0" in cmd and cmd[cmd.index("-f0") + 1] == "1"
    assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "0"
    assert "-te" in cmd and cmd[cmd.index("-te") + 1] == "2"
    assert "-se" in cmd and cmd[cmd.index("-se") + 1] == "2"
    assert "-bs" in cmd and cmd[cmd.index("-bs") + 1] == "4"
    assert "-l" in cmd and cmd[cmd.index("-l") + 1] == "1"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "0"
    assert "-sw" in cmd and cmd[cmd.index("-sw") + 1] == "1"
    assert "-v" in cmd and cmd[cmd.index("-v") + 1] == "v2"
    assert "deepspeed" not in " ".join(cmd)
    assert "torchrun" not in " ".join(cmd)


def test_index_single():
    cmd = index_command("zxowl", "v2", 4)
    assert cmd == ["python", "-P", "train/train_index.py", "zxowl", "v2", "assets/indices", "4", "single"]


def test_infer_cli():
    cmd = infer_command("/w/zxowl.pth", "in.wav", "out.wav", "rmvpe", 0.75, 0, index_path="/w/zxowl.index")
    assert cmd[0:3] == ["python", "-P", "infer/cli.py"]
    assert "--f0-method" in cmd and cmd[cmd.index("--f0-method") + 1] == "rmvpe"
    assert "--index-rate" in cmd and cmd[cmd.index("--index-rate") + 1] == "0.75"
    assert "--index" in cmd
