#!/usr/bin/env bash
set -e

TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
[ -n "$TCMALLOC" ] && export LD_PRELOAD="${TCMALLOC}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"backend:cudaMallocAsync"}

echo "ltx-batch: starting ComfyUI"
python3 /comfyui/main.py --disable-auto-launch --disable-metadata --preview-method taesd &

echo "ltx-batch: starting batch handler"
python3 -u /app/rp_handler.py
