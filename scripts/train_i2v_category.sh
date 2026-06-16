#!/usr/bin/env bash
# scripts/train_i2v_category.sh CATEGORY [STEPS]
# Trains a single category I2V LoRA (dual-noise) on Wan2.2 I2V-A14B using musubi-tuner.
# High-noise -> GPU 0, low-noise -> GPU 1, in parallel.
# Differences from the T2V script: --task i2v-A14B and the i2v DiT weights. The
# --i2v flag goes ONLY on latent caching (caches the conditioning frame); the
# training script infers i2v mode from --task i2v-A14B and REJECTS --i2v.
# Wan2.2 I2V needs NO CLIP model (unlike Wan2.1). Run from project root:
#   bash scripts/train_i2v_category.sh fighting 3000
set -e

CATEGORY="${1:?usage: train_i2v_category.sh CATEGORY [STEPS]}"
STEPS="${2:-3000}"

MUSUBI="$(cd "$(dirname "$0")/../.." && pwd)/musubi-tuner"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Same fragmentation fix that made the 24GB training fit (see train_category.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIT_HIGH="$BASE_DIR/models/wan2.2-i2v/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
DIT_LOW="$BASE_DIR/models/wan2.2-i2v/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
VAE="$BASE_DIR/models/wan2.2-t2v/split_files/vae/wan_2.1_vae.safetensors"
T5="$BASE_DIR/models/wan2.2-t2v/models_t5_umt5-xxl-enc-bf16.pth"

CONFIG="$BASE_DIR/configs/${CATEGORY}_i2v_dataset.toml"
OUT_DIR="$BASE_DIR/loras/${CATEGORY}_i2v"
NAME="${CATEGORY}_i2v_lora_r32"
CACHE_DIR="$BASE_DIR/datasets/processed/$CATEGORY/cache_i2v"

COMMON_ARGS=(
  --task i2v-A14B
  --vae "$VAE"
  --sdpa --mixed_precision fp16 --fp8_base
  --blocks_to_swap 30
  --optimizer_type adamw8bit
  --gradient_checkpointing
  --network_module networks.lora_wan --network_dim 32 --network_alpha 32
  --timestep_sampling shift --discrete_flow_shift 3.0
  --save_every_n_steps 300 --seed 42
)

if [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
  echo "===== Pre-caching (I2V): $CATEGORY ====="
  uv run python "$MUSUBI/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "$CONFIG" --vae "$VAE" --i2v
  uv run python "$MUSUBI/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "$CONFIG" --t5 "$T5" --batch_size 16
else
  echo "===== I2V cache present for $CATEGORY ($(ls "$CACHE_DIR" | wc -l) files) — skipping pre-cache ====="
fi

echo "===== Training I2V $CATEGORY ($STEPS steps) — high-noise (GPU 0) + low-noise (GPU 1) ====="
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
  --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
  "${COMMON_ARGS[@]}" \
  --dit "$DIT_HIGH" \
  --dataset_config "$CONFIG" \
  --learning_rate 2e-4 \
  --min_timestep 900 --max_timestep 1000 \
  --max_train_steps "$STEPS" \
  --output_dir "$OUT_DIR" --output_name "${NAME}_high" \
  --log_with tensorboard --logging_dir "$BASE_DIR/logs/${CATEGORY}_i2v" &
PID_HIGH=$!

CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
  "${COMMON_ARGS[@]}" \
  --dit "$DIT_LOW" \
  --dataset_config "$CONFIG" \
  --learning_rate 2e-5 \
  --min_timestep 0 --max_timestep 900 \
  --max_train_steps "$STEPS" \
  --output_dir "$OUT_DIR" --output_name "${NAME}_low" \
  --log_with tensorboard --logging_dir "$BASE_DIR/logs/${CATEGORY}_i2v" &
PID_LOW=$!

wait $PID_HIGH; HIGH_RC=$?
wait $PID_LOW;  LOW_RC=$?
echo "===== Done: ${CATEGORY}_i2v (high rc=$HIGH_RC, low rc=$LOW_RC) ====="
exit $(( HIGH_RC | LOW_RC ))
