# Headless RVC trainer for Runpod Serverless.
# No pretrained voice weights are baked in. The tiny official mute.zip (required by
# train.py filelist rows) is unpacked into /opt/rvc/logs/mute at build time.
#
# Official RVC targets Python 3.12. Pin an explicit SHA of upstream main that contains
# the official CLI — never a floating main, and never tag 2.3.260718 (that tag has no CLI).
# This SHA is only the Docker pin; our GitHub release for the Hub must use targetCommitish=main.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG RVC_SHA=81eed5e8f68b6bed1789f682fe78cdd324495afc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    RVC_DIR=/opt/rvc \
    PYTHONPATH=/opt/rvc

# git: clone upstream; ffmpeg + libsndfile: audio I/O; python3.12 is the Ubuntu 24.04 default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 libsndfile1-dev ca-certificates \
        python3 python3-venv python3-pip python3-dev wget unzip \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"

# Clone the official repo and check out the pinned SHA. Do not clone a floating main.
# Skip the RVCRealtimeVST submodule (realtime GUI, not used).
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git ${RVC_DIR} \
    && cd ${RVC_DIR} \
    && git checkout ${RVC_SHA} \
    && git rev-parse HEAD > /opt/rvc.commit \
    && test "$(cat /opt/rvc.commit)" = "${RVC_SHA}" \
    && rm -rf .git RVCRealtimeVST

# Stage 1: official torch 2.7.1+cu128 from download.pytorch.org (no PKU/NJU mirrors).
# Stage 2: official requirments_cu128_py312.txt minus the Chinese index, Gradio WebUI,
#          PyMSS, GUI packages, onnxruntime-gpu and FCPE (not used by default rmvpe training).
# faiss-cpu is the official pin and is enough to build the .index (this is not GAN training).
# Do not upgrade torch after this step (constraints).
WORKDIR ${RVC_DIR}
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && python -c "from pathlib import Path; src=Path('requirments_cu128_py312.txt').read_text(); skip=('--index-url','pymss','FreeSimpleGUI','gradio','altair','anyio','fastapi','httpx','markdown-it-py','mdit-py-plugins','pydantic','pyparsing','starlette','uvicorn','websockets','sounddevice','onnxruntime','coloredlogs','torchfcpe'); Path('/tmp/rvc-requirements.txt').write_text('\n'.join(l for l in src.splitlines() if not l.strip().startswith(skip))+'\n')" \
    && python -m pip install -r /tmp/rvc-requirements.txt -c /tmp/constraints.txt

# Tiny official mute set (~312 KiB). Required by the WebUI filelist (two mute rows).
RUN python -m pip install 'huggingface_hub>=0.26.0' \
    && python -c "from huggingface_hub import hf_hub_download; from pathlib import Path; import zipfile; p=hf_hub_download('lj1995/VoiceConversionWebUI','mute.zip',revision='main'); dest=Path('/opt/rvc/logs'); dest.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(p).extractall(dest); assert (dest/'mute'/'0_gt_wavs'/'mute40k.wav').is_file()" \
    && mkdir -p /opt/rvc/assets/weights /opt/rvc/assets/indices /opt/rvc/assets/hubert_base /opt/rvc/assets/rmvpe /opt/rvc/assets/pretrained_v2

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && python -m pip freeze > /app/requirements.lock \
    && python -c "import sys,torch,torchaudio,faiss,librosa,runpod; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| python', sys.version.split()[0], '| runpod', runpod.__version__, '| rvc sha', open('/opt/rvc.commit').read().strip())"

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
# Gradio WebUI is never started; the handler calls official CLI scripts headlessly.
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
