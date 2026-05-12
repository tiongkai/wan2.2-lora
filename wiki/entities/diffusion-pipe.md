---
title: Diffusion-Pipe
type: entity
tags: [tool, training, diffusion-pipe, deepspeed]
created: 2026-05-12
updated: 2026-05-12
---

# Diffusion-Pipe

Diffusion-pipe is a training tool that uses **DeepSpeed** for distributed and memory-efficient LoRA training. It's more advanced and less user-friendly than [[ai-toolkit]] or [[musubi-tuner]].

## Key Info

- **Repository**: https://github.com/tdrussell/diffusion-pipe
- **Config format**: TOML
- **Launch**: via DeepSpeed
- **Audience**: Advanced users with technical experience

## Requirements

- Text encoder model
- VAE model
- Wan model weights
- DeepSpeed installed

## Training Command

```bash
cd /workspace/diffusion-pipe/
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_video.toml
```

## Usage Context

- Commonly used on cloud platforms (RunPod)
- Works alongside [[comfyui]] for inference
- Used in the Civitai local training guide for Wan 2.2 on RTX 3090
- Trains at small resolutions (256×256) for memory efficiency

## Typical Training Setup

From community guides:
- RTX 3090, 256×256 resolution
- 4-8 second clips, ~20 clips
- Training takes 10-20 hours
- TOML config for dataset and training parameters

## Strengths

- DeepSpeed optimizations for distributed training
- Can work on lower VRAM with aggressive settings
- Good for multi-GPU setups

## Limitations

- "Not exactly user-friendly" — assumes baseline technical experience
- Less documentation than [[musubi-tuner]] or [[ai-toolkit]]
- Wan 2.2 dual-noise handling requires separate runs
- Smaller community

## Related

- [[musubi-tuner]] — More user-friendly alternative
- [[ai-toolkit]] — Simplest workflow alternative
- [[training-tools-comparison]] — Side-by-side comparison
