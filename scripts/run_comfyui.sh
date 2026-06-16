#!/usr/bin/env bash
# scripts/run_comfyui.sh [extra comfyui args...]
# Launches ComfyUI for Wan2.2 synthetic-clip generation.
# - Pinned to GPU 1 (no desktop load) by default; override with GPU=0.
# - Saved videos land in wan2.2-lora/generated/clips/<category>/ (via filename_prefix).
# - Models/LoRAs are wired in ComfyUI/extra_model_paths.yaml (stock Wan2.2, no Remix).
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMFY_DIR="$(cd "$BASE_DIR/.." && pwd)/ComfyUI"
OUT_DIR="$BASE_DIR/generated/clips"
GPU="${GPU:-1}"
PORT="${PORT:-8188}"

mkdir -p "$OUT_DIR"
# expandable_segments helps the 14B T2V fit alongside both experts on 24GB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "ComfyUI -> http://127.0.0.1:$PORT   (GPU $GPU, output: $OUT_DIR)"
cd "$COMFY_DIR"
# --lowvram: stream the 14B UNet from the 128GB system RAM (won't co-fit with the
#            text encoder on 24GB otherwise).
# --disable-cuda-malloc: use the native caching allocator so expandable_segments
#            takes effect; fixes the "Allocation on device" OOM during LoRA patching.
# (The workflow also runs the umt5 text encoder on CPU — CLIPLoader device=cpu —
#  which frees ~11GB of VRAM. Together these let a 14B T2V + LoRA fit on one 24GB card.)
CUDA_VISIBLE_DEVICES="$GPU" exec ./.venv/bin/python main.py \
  --port "$PORT" \
  --output-directory "$OUT_DIR" \
  --lowvram \
  --disable-cuda-malloc \
  "$@"
