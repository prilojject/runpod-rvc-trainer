# Agent notes — runpod-rvc-trainer

## What this worker is

A Runpod Serverless **headless RVC trainer**. Input: zip of one speaker's audio. Output: inference-ready `.pth` + `.index`. Official CLI only; do not start `webui.py` or any Gradio server.

Do not fold this into `runpod-gpt-sovits-trainer`. Do not confuse it with `runpod-kokoro-tts` (inference only). Do not switch the upstream to Applio.

## Pinned upstream

- Repo: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
- SHA: `81eed5e8f68b6bed1789f682fe78cdd324495afc` (written to `/opt/rvc.commit`)
- Do **not** pin tag `2.3.260718` (no official CLI) and do **not** clone a floating `main`.
- The SHA is the Docker pin only. GitHub releases of *this* repo must use `targetCommitish=main` (a SHA target breaks Runpod Hub's `git clone -b`).

## Defaults to keep

- RVC v2, 40 kHz, F0=rmvpe, `is_half=true` (fp16, not bf16)
- Public weights from `lj1995/VoiceConversionWebUI`, no token, official `assets/` layout
- `logs/mute` from official `mute.zip` (required by the filelist)
- `filelist.txt` / `config.json` must match `webui.py::run_train_model` on the pinned SHA
- For v2 + 40k the official config path is `configs/v1/40k.json` (there is no `v2/40k.json`)
- One GPU, no DDP / torchrun / deepspeed launcher
- Do not pass `--shm-size` via Runpod `dockerArgs`
- Do not bake pretrained voice weights into the image

## Tests

- Unit tests (`tests/`) must run without a GPU and without pretrained weights.
- The Hub smoke test is one case only: v2 / 40k / 2 epochs / tiny_voice.zip / no samples / RTX 4090 / timeout 20 min.
- After a paid GPU test: delete the endpoint, pod and registry auth. Confirm with GraphQL that `endpoints=[]`. Leave protected pods `92anyumh6q7jrf` and `oejzh2afh8hcym` (and volumes `13faqbphuo`, `fv2c638zex`) untouched.

## Secrets

Never print `RUNPOD_API_KEY`, `HF_TOKEN`, `gh` tokens, or `BUCKET_*` values. Never accept tokens in the job input.
