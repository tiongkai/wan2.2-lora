---
title: Training Parameters Reference
type: concept
tags: [training, parameters, learning-rate, rank, optimizer, configuration]
created: 2026-05-12
updated: 2026-05-12
---

# Training Parameters Reference

A comprehensive reference for Wan 2.2 LoRA training parameters. Settings vary by LoRA type, tool, and hardware — this page collects community-validated configurations from multiple sources.

## Quick Reference: Recommended Settings by LoRA Type

### Identity/Character LoRA [wavespeed-training-settings]

| Parameter | Value |
|-----------|-------|
| Rank (dim) | 16 |
| Alpha | 16 |
| Learning rate | 5e-5 |
| Optimizer | AdamW (weight_decay 0.01) |
| Scheduler | Cosine with 5% warmup |
| Batch size | 2-4 (A100/4090) |
| Dataset | 12-20 images |
| Steps | 2000-3000 |

### Style LoRA [wavespeed-training-settings]

| Parameter | Value |
|-----------|-------|
| Rank (dim) | 32 |
| Alpha | 32 |
| Learning rate | 7e-5 to 1e-4 |
| Optimizer | AdamW (weight_decay 0.01) |
| Scheduler | Cosine with 5% warmup |
| Batch size | 2-4 |
| Dataset | 30-50 images |
| Steps | 3000-4000 |

### General/Motion LoRA [apatero-best-practices]

| Parameter | Value |
|-----------|-------|
| Rank (dim) | 32-64 |
| Alpha | Match rank |
| Learning rate | 0.0001-0.0003 |
| Batch size | 1-2 |
| Steps | 2000-4000 |
| Dataset | 15-30 video clips |

## Tool-Specific Configurations

### Musubi-Tuner (kohya-ss)

```bash
accelerate launch --mixed_precision bf16 wan_train_network.py \
  --task t2v-A14B \
  --dit wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --vae wan_2.1_vae.safetensors \
  --dataset_config dataset.toml \
  --sdpa --mixed_precision bf16 --fp8_base \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --gradient_checkpointing \
  --network_module networks.lora_wan --network_dim 32 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
  --output_dir output --output_name my-lora
```

**Wan 2.2 dual-noise handling** (train high and low separately):

| Expert | --min_timestep | --max_timestep | Learning Rate |
|--------|---------------|----------------|---------------|
| High-noise | 900 | 1000 | 2e-4 |
| Low-noise | 0 | 900 | 2e-5 |

**Alternative community settings**: LR 3e-5 with `loraplus_lr_ratio=4`, 1600 steps, `discrete_flow_shift 12`. High and low noise back-to-back in ~12 hours. [musubi-community-discussions]

**Discrete flow shift**:
- 7.0 — faster training
- 3.0 — if details not learned well, lower the shift
- 12.0 — used with lower LR + loraplus

**Task flags**:
- `t2v-A14B` — Wan 2.2 text-to-video
- `i2v-A14B` — Wan 2.2 image-to-video
- `t2v-14B` — Wan 2.1 text-to-video
- `i2v-14B` — Wan 2.1 image-to-video

### AI Toolkit (Ostris)

```yaml
config:
  name: wan22-i2v-lora
  process:
    - type: sd_trainer
      training_folder: output
      device: cuda:0
      network:
        type: lora
        linear: 64
        linear_alpha: 64
      save:
        dtype: float16
        save_every: 500
      datasets:
        - folder_path: /path/to/dataset
          num_frames: 81
          resolution: [512, 768]
          caption_dropout_rate: 0.05
      train:
        batch_size: 1
        steps: 3000
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer:
          type: adamw
          lr: 0.0002
        dtype: bf16
      model:
        name_or_path: Wan2.2-I2V-A14B-Diffusers-bf16
        arch: wan22_14b_i2v
        quantize: true
        qtype: qfloat8
      model_kwargs:
        train_high_noise: true
        train_low_noise: true
      sample:
        sampler: flowmatch
        num_frames: 81
        fps: 16
```

Key advantage: handles both high-noise and low-noise training with simple flags.

### Diffusion-Pipe

```bash
cd /workspace/diffusion-pipe/
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_video.toml
```

Uses TOML config files and DeepSpeed for training. See [[diffusion-pipe]] for details.

## Parameter Deep Dive

### Learning Rate

The most sensitive parameter. WAN 2.2 punishes high learning rates with plasticky skin and loss spikes.

| LR Range | Use Case | Risk |
|----------|----------|------|
| 3e-5 to 5e-5 | Identity, conservative | Slow convergence |
| 7e-5 to 1e-4 | Style | Moderate |
| 1e-4 to 3e-4 | General, motion | Can overfit quickly |
| > 3e-4 | Not recommended | Loss spikes, artifacts |

### Rank (Network Dimension)

Higher rank = more capacity to learn, but also more VRAM and higher overfitting risk.

| Rank | Use Case |
|------|----------|
| 16 | Identity LoRA (minimal, focused) |
| 32 | Style, general purpose |
| 64 | Complex styles, motion (AI Toolkit default) |
| > 64 | Rarely needed, high overfitting risk |

### Training Steps

| Steps | Outcome |
|-------|---------|
| < 1500 | Typically underfit |
| 2000-3000 | Good for identity with clean dataset |
| 3000-4000 | Good for style and motion |
| 4000-5000 | May improve quality, check for overfitting |
| > 5000 | High overfitting risk unless dataset is large |

### Optimizer

**AdamW** is the standard. Use `weight_decay 0.01` for regularization. `adamw8bit` reduces VRAM in [[musubi-tuner]].

### Scheduler

- **Cosine** with 5% warmup — most common, well-tested
- **Polynomial** (power 8) — used in some Civitai guides

## Contradictions Between Sources

| Parameter | WaveSpeed | Apatero | Musubi Community |
|-----------|-----------|---------|-----------------|
| LR (identity) | 5e-5 | 0.0001-0.0003 | 2e-4 (or 3e-5 + loraplus) |
| Rank | 16 (identity) | 32-64 | 32 |
| Steps | Not specified | 2000-4000 | 1600 (with loraplus) |

**Guidance**: WaveSpeed's lower LR + lower rank is more conservative and safer from overfitting. Higher settings from Apatero/musubi-tuner learn faster but need careful monitoring. Start conservative, increase if underfit.

## Related

- [[dual-noise-architecture]] — Why Wan 2.2 needs special handling
- [[hardware-requirements]] — How parameters interact with VRAM
- [[overfitting-and-troubleshooting]] — When parameter choices go wrong
- [[musubi-tuner]] — Tool-specific configuration details
- [[ai-toolkit]] — Tool-specific configuration details
