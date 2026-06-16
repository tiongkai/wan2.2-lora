#!/usr/bin/env bash
# scripts/train_i2v_77f.sh [STEPS]
# Trains the fighting I2V LoRA at 77 frames / 480x832 (~4.8s @16fps — the longest
# clean clip our 5.0s training videos allow: at Wan's native 16fps a 5.0s clip is
# 80 frames, and 77 is the largest valid 4n+1 length <= 80).
# Same dual-noise setup; higher block-swap for the heavier activation load.
# --i2v goes ONLY on caching; training infers i2v from --task i2v-A14B.
set -e

STEPS="${1:-3000}"
MUSUBI="$(cd "$(dirname "$0")/../.." && pwd)/musubi-tuner"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIT_HIGH="$BASE_DIR/models/wan2.2-i2v/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
DIT_LOW="$BASE_DIR/models/wan2.2-i2v/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
VAE="$BASE_DIR/models/wan2.2-t2v/split_files/vae/wan_2.1_vae.safetensors"
T5="$BASE_DIR/models/wan2.2-t2v/models_t5_umt5-xxl-enc-bf16.pth"

CONFIG="$BASE_DIR/configs/fighting_i2v_77f_dataset.toml"
OUT_DIR="$BASE_DIR/loras/fighting_i2v_77f"
NAME="fighting_i2v_77f_lora_r32"
CACHE_DIR="$BASE_DIR/datasets/processed/fighting/cache_i2v_77f"

COMMON_ARGS=(
  --task i2v-A14B
  --vae "$VAE"
  --sdpa --mixed_precision fp16 --fp8_base
  --blocks_to_swap 39
  --optimizer_type adamw8bit
  --gradient_checkpointing
  --network_module networks.lora_wan --network_dim 32 --network_alpha 32
  --timestep_sampling shift --discrete_flow_shift 3.0
  --save_every_n_steps 300 --seed 42
)

if [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
  echo "===== Pre-caching (I2V 77f @480x832) ====="
  uv run python "$MUSUBI/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "$CONFIG" --vae "$VAE" --i2v
  uv run python "$MUSUBI/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "$CONFIG" --t5 "$T5" --batch_size 16
else
  echo "===== 77f cache present ($(ls "$CACHE_DIR" | wc -l) files) — skipping pre-cache ====="
fi

echo "===== Training fighting_i2v_77f ($STEPS steps) — high (GPU 0) + low (GPU 1) ====="
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
  --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
  "${COMMON_ARGS[@]}" \
  --dit "$DIT_HIGH" --dataset_config "$CONFIG" \
  --learning_rate 2e-4 --min_timestep 900 --max_timestep 1000 \
  --max_train_steps "$STEPS" \
  --output_dir "$OUT_DIR" --output_name "${NAME}_high" \
  --log_with tensorboard --logging_dir "$BASE_DIR/logs/fighting_i2v_77f" &
PID_HIGH=$!

CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
  "${COMMON_ARGS[@]}" \
  --dit "$DIT_LOW" --dataset_config "$CONFIG" \
  --learning_rate 2e-5 --min_timestep 0 --max_timestep 900 \
  --max_train_steps "$STEPS" \
  --output_dir "$OUT_DIR" --output_name "${NAME}_low" \
  --log_with tensorboard --logging_dir "$BASE_DIR/logs/fighting_i2v_77f" &
PID_LOW=$!

wait $PID_HIGH; HIGH_RC=$?
wait $PID_LOW;  LOW_RC=$?
echo "===== Done: fighting_i2v_77f (high rc=$HIGH_RC, low rc=$LOW_RC) ====="
exit $(( HIGH_RC | LOW_RC ))
