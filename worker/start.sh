#!/usr/bin/env bash
set -e

TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
[ -n "$TCMALLOC" ] && export LD_PRELOAD="${TCMALLOC}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

VOLUME_ROOT="${LTX_VOLUME_ROOT:-/runpod-volume}"
V11_NAME="${LTX_V11_MODEL_NAME:-ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors}"
V11_URL="${LTX_V11_MODEL_URL:-https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors}"
V11_DIR="${VOLUME_ROOT}/models/diffusion_models"
V11_PATH="${V11_DIR}/${V11_NAME}"

if [ -d "$VOLUME_ROOT" ]; then
  if [ ! -s "$V11_PATH" ]; then
    echo "ltx-batch: v1.1 transformer missing on network volume, fetching once -> ${V11_PATH}"
    mkdir -p "$V11_DIR"
    wget -nv -O "${V11_PATH}.part" "$V11_URL"
    mv "${V11_PATH}.part" "$V11_PATH"
    echo "ltx-batch: v1.1 transformer ready on volume"
  else
    echo "ltx-batch: v1.1 transformer already present on volume"
  fi
else
  echo "ltx-batch: WARNING no network volume mounted at ${VOLUME_ROOT}; v1.1 transformer unavailable"
fi

echo "ltx-batch: starting ComfyUI"
python3 /comfyui/main.py --disable-auto-launch --disable-metadata --preview-method taesd \
  --extra-model-paths-config /comfyui/extra_model_paths.yaml &

echo "ltx-batch: starting batch handler"
python3 -u /app/rp_handler.py
