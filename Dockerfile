FROM ghcr.io/e-dream-ai/gpu-container-ltx:latest

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN wget -nv -O /comfyui/models/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors \
    https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors

WORKDIR /app
COPY worker/ /app/
RUN chmod +x /app/start.sh

ENV LTX_WORKFLOW_PATH=/app/workflow_base.json

CMD ["/app/start.sh"]
