#!/usr/bin/env bash
# scripts/train_all.sh
# Trains all four category LoRAs using musubi-tuner.
# High-noise (GPU 0) and low-noise (GPU 1) experts run simultaneously per category.
# Run from project root: bash scripts/train_all.sh
set -e

MUSUBI="$(cd .. && pwd)/musubi-tuner"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Model paths — verify these match your download after Task 1 Step 4
DIT_HIGH="$BASE_DIR/models/wan2.2-t2v/wan2.2_t2v_high_noise_14B_fp16.safetensors"
DIT_LOW="$BASE_DIR/models/wan2.2-t2v/wan2.2_t2v_low_noise_14B_fp16.safetensors"
VAE="$BASE_DIR/models/wan2.2-t2v/wan_2.1_vae.safetensors"
T5="$BASE_DIR/models/wan2.2-t2v/models_t5_umt5-xxl-enc-bf16.pth"

COMMON_ARGS=(
  --task t2v-A14B
  --vae "$VAE"
  --sdpa --mixed_precision bf16 --fp8_base
  --optimizer_type adamw8bit
  --gradient_checkpointing
  --network_module networks.lora_wan --network_dim 32 --network_alpha 32
  --timestep_sampling shift --discrete_flow_shift 3.0
  --save_every_n_steps 300 --seed 42
)

train_category() {
  local category=$1
  local steps=${2:-3000}
  local config="$BASE_DIR/configs/${category}_dataset.toml"
  local out_dir="$BASE_DIR/loras/$category"
  local name="${category}_lora_r32"

  echo "===== Pre-caching: $category ====="
  cd "$MUSUBI"
  uv run python cache_latents.py \
    --task t2v-A14B --dit "$DIT_HIGH" --vae "$VAE" \
    --dataset_config "$config"
  uv run python cache_text_encoder_outputs.py \
    --task t2v-A14B --t5xxl "$T5" \
    --dataset_config "$config"

  echo "===== Training $category — high-noise (GPU 0) + low-noise (GPU 1) ====="
  CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/wan_train_network.py \
    "${COMMON_ARGS[@]}" \
    --dit "$DIT_HIGH" \
    --dataset_config "$config" \
    --learning_rate 2e-4 \
    --min_timestep 900 --max_timestep 1000 \
    --max_train_steps "$steps" \
    --output_dir "$out_dir" --output_name "${name}_high" \
    --log_with tensorboard --logging_dir "$BASE_DIR/logs/$category" &

  CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
    --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/wan_train_network.py \
    "${COMMON_ARGS[@]}" \
    --dit "$DIT_LOW" \
    --dataset_config "$config" \
    --learning_rate 2e-5 \
    --min_timestep 0 --max_timestep 900 \
    --max_train_steps "$steps" \
    --output_dir "$out_dir" --output_name "${name}_low" \
    --log_with tensorboard --logging_dir "$BASE_DIR/logs/$category" &

  wait
  cd "$BASE_DIR"
  echo "===== Done: $category ====="
}

train_category fighting  3000
train_category vandalism 2500
train_category stabbing  3000
train_category shooting  3000

echo "All LoRAs trained."
