#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "ltx-batch: boot $(date -u +%H:%M:%S)"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
[ -n "$TCMALLOC" ] && export LD_PRELOAD="${TCMALLOC}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

VOLUME_ROOT="${LTX_VOLUME_ROOT:-/runpod-volume}"
V11_NAME="${LTX_V11_MODEL_NAME:-ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors}"
V11_URL="${LTX_V11_MODEL_URL:-https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors}"
if [ -d "$VOLUME_ROOT" ]; then
  V11_DIR="${VOLUME_ROOT}/models/diffusion_models"
else
  V11_DIR="/comfyui/models/diffusion_models"
fi
V11_PATH="${V11_DIR}/${V11_NAME}"

if [ ! -s "$V11_PATH" ]; then
  echo "ltx-batch: downloading v1.1 transformer -> ${V11_PATH}"
  mkdir -p "$V11_DIR"
  if wget -q --tries=5 --timeout=60 --continue -O "${V11_PATH}.part" "$V11_URL"; then
    mv "${V11_PATH}.part" "$V11_PATH"
    echo "ltx-batch: v1.1 transformer ready"
  else
    echo "ltx-batch: WARNING v1.1 download failed; renders will fail until model is present"
  fi
else
  echo "ltx-batch: v1.1 transformer already present at ${V11_PATH}"
fi

echo "ltx-batch: starting ComfyUI"
python3 /comfyui/main.py --disable-auto-launch --disable-metadata --preview-method taesd \
  --extra-model-paths-config /comfyui/extra_model_paths.yaml &

echo "ltx-batch: starting batch handler"
exec python3 -u "${SCRIPT_DIR}/rp_handler.py"
