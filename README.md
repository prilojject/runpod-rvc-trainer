# RVC Voice Conversion Trainer for Runpod

[![Runpod](https://img.shields.io/badge/Runpod-Hub-7c3aed)](https://console.runpod.io/hub/listing/prilojject/runpod-rvc-trainer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A **Runpod Serverless** worker that trains an official [Retrieval-based Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) model from a zip of one speaker's audio and returns an inference-ready **`.pth` + `.index`**.

It is a headless wrapper around the official CLI (`docs/en/cli.md`). The Gradio `webui.py` is never started.

This is **not** [runpod-gpt-sovits-trainer](https://github.com/prilojject/runpod-gpt-sovits-trainer) (few-shot TTS / GPT+SoVITS voice cloning) and **not** [runpod-kokoro-tts](https://github.com/prilojject/runpod-kokoro-tts) (TTS inference only).

## What you send / what you get

**Send** a `dataset_url` pointing to a `.zip` or `.tar.gz` of wav/flac/mp3 clips of **one target speaker**. Transcripts are not needed (this is voice conversion, not TTS). Official advice is about 10 minutes of clean, low-noise speech.

```
voice.zip
  001.wav
  002.wav
  ...
```

**Get** an inference-ready small model (`-sw 1`):

- `<model_name>.pth` — put it in `assets/weights/`
- `<model_name>.index` — put it in `assets/indices/`

Delivered to a Hugging Face repo, an S3-compatible bucket, a presigned PUT URL, or a Runpod network volume (`/runpod-volume/rvc/runs/<run_name>/`). If none of those are configured, the job still returns the file names and sizes, and a warning that the weights will disappear when the worker stops.

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| Architecture | **v2** | v1 is optional |
| Sample rate | **40 kHz** | also `32k` / `48k` |
| F0 | **rmvpe**, half-precision on CUDA | `pm` is a CPU fallback |
| Epochs | 200 | use 2 for a smoke test |
| Batch size | 8 | lowered automatically if the sliced set is smaller |
| GPU | 24 GB (RTX 4090 class) | plenty of headroom |

Public pretrained weights come from [`lj1995/VoiceConversionWebUI`](https://huggingface.co/lj1995/VoiceConversionWebUI) (~0.55 GiB, no token). They are downloaded at job start and cached on a network volume when one is attached. They are not baked into the image.

Upstream is pinned to SHA `81eed5e8f68b6bed1789f682fe78cdd324495afc` (the commit that added the official CLI). Tag `2.3.260718` is not used: it does not include that CLI.

## Example job

```json
{
  "input": {
    "dataset_url": "https://example.com/voice.zip",
    "model_name": "my_speaker",
    "version": "v2",
    "sample_rate": "40k",
    "epochs": 200,
    "batch_size": 8,
    "f0_method": "rmvpe",
    "hf_repo_id": "you/my-rvc-speaker"
  }
}
```

Hub presets:

1. **RVC v2 40k (default)** — 200 epochs, batch 8, rmvpe
2. **Quick test** — 2 epochs, batch 4 (quality of the voice will be poor; this only checks the pipeline)

A tiny public-domain LibriSpeech sample lives at [`examples/datasets/tiny_voice.zip`](examples/datasets/tiny_voice.zip) for the Hub smoke test. Do not expect a usable voice from it.

## How to run the weights

In the official RVC WebUI, drop the `.pth` into `assets/weights/` and the `.index` into `assets/indices/`.

Or, from the official project root:

```bash
python infer/cli.py \
  --model assets/weights/my_speaker.pth \
  --input source.wav \
  --output converted.wav \
  --f0-method rmvpe \
  --index-rate 0.75
```

If `--index-rate` is greater than 0 and the index cannot be found, the official CLI exits with an error instead of silently disabling retrieval. Pass `--index-rate 0` to skip retrieval.

## FAQ

**Does this clone a voice from a few seconds of audio?**  
RVC is voice *conversion* (source speech → target timbre), not few-shot TTS. Plan on minutes of clean speech. For few-shot TTS, use [runpod-gpt-sovits-trainer](https://github.com/prilojject/runpod-gpt-sovits-trainer).

**Why is there no Gradio UI?**  
This worker is meant for Runpod Serverless jobs. The official CLI (`train/preprocess.py`, `train/dataset/extract_*.py`, `train/train.py`, `train/train_index.py`, `infer/cli.py`) is what runs.

**Can I train several speakers into one model?**  
Not in this version. The official CLI supports multi-speaker training; this worker keeps the single-speaker path.

**Is Applio included?**  
No. This listing uses the official RVC-Project pipeline and the canonical `.pth` / `.index` format.

**Do I need a Hugging Face token for the pretrained files?**  
No. A token is only required if you set `hf_repo_id` to upload the result.

**What about source separation (PyMSS / UVR)?**  
Not in this worker. Send already-clean vocal clips.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The GPU smoke test is `rvc_v2_40k_smoke_2_epochs` in [`.runpod/tests.json`](.runpod/tests.json): v2 / 40k / 2 epochs / no samples, on an RTX 4090.

## License

MIT. Upstream RVC is also MIT, © 2023 liujing04, 源文雨, Ftps. See [NOTICE](NOTICE) and [LICENSE.upstream](LICENSE.upstream).
