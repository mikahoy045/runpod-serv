#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "ltx-batch: boot $(date -u +%H:%M:%S)"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
[ -n "$TCMALLOC" ] && export LD_PRELOAD="${TCMALLOC}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

VOLUME_ROOT="${LTX_VOLUME_ROOT:-/runpod-volume}"
if [ -d "$VOLUME_ROOT" ]; then
  MODELS_ROOT="${VOLUME_ROOT}/models"
else
  MODELS_ROOT="/comfyui/models"
fi
READY_FILE="${MODELS_ROOT}/.models_ready"
export LTX_MODELS_READY_FILE="$READY_FILE"

HF_BASE="${LTX_HF_BASE:-https://huggingface.co}"
KIJAI="${HF_BASE}/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG="${HF_BASE}/Comfy-Org/ltx-2/resolve/main"
UPSCALER="${HF_BASE}/Lightricks/LTX-2.3/resolve/main"
LORA_BASE="${HF_BASE}/Lightricks"
TAEHV_BASE="${LTX_TAEHV_BASE:-https://github.com/madebyollin/taehv/raw/main/safetensors}"

read -r -d '' MODELS <<EOF
diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors|${KIJAI}/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors
text_encoders/gemma_3_12B_it_fpmixed.safetensors|${COMFYORG}/split_files/text_encoders/gemma_3_12B_it_fpmixed.safetensors
text_encoders/ltx-2.3_text_projection_bf16.safetensors|${KIJAI}/text_encoders/ltx-2.3_text_projection_bf16.safetensors
vae/LTX23_video_vae_bf16.safetensors|${KIJAI}/vae/LTX23_video_vae_bf16.safetensors
vae/LTX23_audio_vae_bf16.safetensors|${KIJAI}/vae/LTX23_audio_vae_bf16.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors|${UPSCALER}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
vae_approx/taeltx2_3.safetensors|${TAEHV_BASE}/taeltx2_3.safetensors
loras/ltx-2-19b-lora-camera-control-static.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors
loras/ltx-2-19b-lora-camera-control-dolly-in.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors
loras/ltx-2-19b-lora-camera-control-dolly-out.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors
loras/ltx-2-19b-lora-camera-control-dolly-left.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors
loras/ltx-2-19b-lora-camera-control-dolly-right.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors
loras/ltx-2-19b-lora-camera-control-jib-up.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors
loras/ltx-2-19b-lora-camera-control-jib-down.safetensors|${LORA_BASE}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors
EOF

fetch_model() {
  local rel="$1" url="$2"
  local dest="${MODELS_ROOT}/${rel}"
  if [ -s "$dest" ]; then
    echo "ltx-batch: present ${rel}"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "ltx-batch: fetching ${rel}"
  if wget -q --tries=5 --timeout=120 --continue -O "${dest}.part" "$url"; then
    mv "${dest}.part" "$dest"
    echo "ltx-batch: done ${rel}"
    return 0
  fi
  rm -f "${dest}.part"
  echo "ltx-batch: FAILED ${rel}"
  return 1
}

populate_models() {
  mkdir -p "$MODELS_ROOT"
  rm -f "$READY_FILE"
  local missing=0
  while IFS='|' read -r rel url; do
    [ -z "$rel" ] && continue
    fetch_model "$rel" "$url" || missing=1
  done <<< "$MODELS"
  if [ "$missing" = "0" ]; then
    touch "$READY_FILE"
    echo "ltx-batch: ALL MODELS READY"
  else
    echo "ltx-batch: WARNING some models missing; renders needing them will fail"
  fi
}

populate_models &

echo "ltx-batch: starting ComfyUI"
python3 /comfyui/main.py --disable-auto-launch --disable-metadata --preview-method taesd \
  --extra-model-paths-config /comfyui/extra_model_paths.yaml &

echo "ltx-batch: starting batch handler"
exec python3 -u "${SCRIPT_DIR}/rp_handler.py"
