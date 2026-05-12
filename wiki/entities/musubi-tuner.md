---
title: Musubi-Tuner (kohya-ss)
type: entity
tags: [tool, training, musubi-tuner, kohya-ss]
created: 2026-05-12
updated: 2026-05-12
---

# Musubi-Tuner

Musubi-tuner is a community-maintained LoRA training toolkit by **kohya-ss**. It is the most widely used and community-tested tool for Wan 2.1/2.2 LoRA training.

## Key Info

- **Repository**: https://github.com/kohya-ss/musubi-tuner
- **Author**: kohya-ss (unofficial, not affiliated with Wan team)
- **Supported models**: HunyuanVideo, Wan 2.1/2.2, FramePack, FLUX.1 Kontext, others
- **Config format**: TOML (dataset config) + CLI arguments (training config)
- **Launch**: via `accelerate launch`

## Features

- FP8 base model quantization (`--fp8_base`)
- Block swapping (GPU↔CPU, saves VRAM at cost of speed)
- Latent and text encoder pre-caching
- Gradient checkpointing
- AdamW 8-bit optimizer (`adamw8bit`)
- `networks.lora_wan` network module
- SECourses 1-click installer with presets available

## Wan 2.2 Task Flags

| Flag | Model |
|------|-------|
| `t2v-A14B` | Wan 2.2 T2V |
| `i2v-A14B` | Wan 2.2 I2V |
| `t2v-1.3B` | Wan 2.1 T2V 1.3B |
| `t2v-14B` | Wan 2.1 T2V 14B |
| `i2v-14B` | Wan 2.1 I2V 14B |
| `t2v-1.3B-FC` | Wan 2.1-Fun Control |
| `t2v-14B-FC` | Wan 2.1-Fun Control |

## Installation

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
pip install -r requirements.txt
```

## Pre-Training Steps

Before training, you must cache latents and text encoder outputs:

```bash
# Cache latents
python cache_latents.py --task t2v-A14B --dataset_config dataset.toml ...

# Cache text encoder
python cache_text_encoder_outputs.py --task t2v-A14B --dataset_config dataset.toml ...
```

**Note**: Wan 2.2 does **not** require a CLIP model (unlike Wan 2.1).

## Dataset Config (TOML)

```toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "train/video"
cache_directory = "train/cache_video"
frame_extraction = "uniform"
source_fps = 16.0
target_frames = [57]
max_frames = 57
resolution = [384, 384]
```

## Training Command (Wan 2.2 T2V)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/wan_train_network.py \
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

## Dual-Noise Training (Wan 2.2)

Train high-noise and low-noise experts separately:

**High-noise** (layout/composition):
```bash
--dit wan2.2_t2v_high_noise_14B_fp16.safetensors \
--min_timestep 900 --max_timestep 1000 \
--learning_rate 2e-4
```

**Low-noise** (details/refinement):
```bash
--dit wan2.2_t2v_low_noise_14B_fp16.safetensors \
--min_timestep 0 --max_timestep 900 \
--learning_rate 2e-5
```

See [[dual-noise-architecture]] for why this is necessary.

## Community Resources

- **Docs**: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md
- **Discussion #182**: Training rules of the trade
- **Discussion #455**: Wan 2.2 general discussion, advice, questions
- **SECourses**: 1-click installer with presets

## Strengths

- Most community support and tested configurations
- Fine-grained control over every parameter
- Extensive VRAM optimization options
- Well-documented

## Limitations

- Not user-friendly for beginners
- Wan 2.2 dual-noise requires separate training runs (more complex workflow)
- CLI-only (no web UI)

## Related

- [[ai-toolkit]] — Alternative with simpler dual-noise handling
- [[diffusion-pipe]] — Alternative using DeepSpeed
- [[training-parameters]] — Parameter reference
- [[training-tools-comparison]] — Side-by-side comparison
