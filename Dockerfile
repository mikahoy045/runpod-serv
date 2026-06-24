FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1 \
    build-essential \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install comfy-cli
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
RUN /usr/bin/yes | comfy --workspace /comfyui install \
    --cuda-version 12.8 --nvidia --skip-torch-or-directml
RUN comfy tracking disable

WORKDIR /comfyui

RUN pip install runpod requests websocket-client boto3

RUN cd custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && \
    pip install -r requirements.txt 2>/dev/null || true

RUN cd custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt 2>/dev/null || true

RUN cd custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt 2>/dev/null || true

RUN cd custom_nodes && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    cd rgthree-comfy && \
    pip install -r requirements.txt 2>/dev/null || true

RUN pip cache purge

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY src/extra_model_paths.yaml /comfyui/extra_model_paths.yaml

WORKDIR /app
COPY worker/ /app/
RUN chmod +x /app/start.sh

ENV LTX_WORKFLOW_PATH=/app/workflow_base.json

CMD ["/app/start.sh"]
