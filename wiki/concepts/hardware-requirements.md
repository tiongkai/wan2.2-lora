---
title: Hardware Requirements
type: concept
tags: [hardware, vram, gpu, cloud, training-time]
created: 2026-05-12
updated: 2026-05-12
---

# Hardware Requirements

LoRA training VRAM requirements depend on model size, training resolution, batch size, and quantization settings. Wan 2.2's [[dual-noise-architecture]] doesn't require both experts in memory simultaneously — each expert is ~14B active.

## VRAM Requirements

| Training Type | Minimum VRAM | Recommended | Notes |
|--------------|-------------|-------------|-------|
| Image LoRA | ~12 GB | 16+ GB | With FP8 quantization |
| Video LoRA (T2V) | ~24 GB | 24+ GB | RTX 4090/3090 viable |
| Video LoRA (I2V) | ~24 GB | 48+ GB | A6000 or cloud recommended |
| I2V with FP8 + block-swap | ~16 GB | 24 GB | Community-reported, tight |

## GPU Comparison

| GPU | VRAM | Training Time (approx) | Cost Tier | Notes |
|-----|------|----------------------|-----------|-------|
| RTX 3090 | 24 GB | 10-20 hours | Consumer | 256x256 small videos, viable but slow |
| RTX 4090 | 24 GB | Faster than 3090 | Consumer | Recommended for local training |
| RTX 5090 | 24-32 GB | Fast | Consumer | Newest generation |
| A6000 | 48 GB | ~24 hours (person LoRA) | Prosumer | Higher resolution possible |
| A100 | 80 GB | Fastest, batch 2-4 | Cloud | Best for large experiments |
| H100/H200 | 80+ GB | Fastest | Cloud | Available via RunComfy, RunPod |

## Training Time Estimates

| Scenario | GPU | Resolution | Steps | Time |
|----------|-----|-----------|-------|------|
| Wan 2.1 1.3B anime | RTX 3090 | 512px | 3500 | ~2.5 hours |
| Wan 2.2 14B person | A6000 | 768px | 3000 | ~24 hours |
| Wan 2.2 14B person | Consumer 24GB | 768px | 3000 | 2-3 days |
| Wan 2.2 dual-noise (both) | Any 24GB | 360-512px | 1600 ea | ~12 hours total |
| Civitai guide (diffusion-pipe) | RTX 3090 | 256x256 | Full | 10-20 hours |

## VRAM Optimization Techniques

### FP8 Quantization
Both [[musubi-tuner]] (`--fp8_base`) and [[ai-toolkit]] (`quantize: true, qtype: qfloat8`) support FP8 quantization of the base model, reducing VRAM by ~40%.

### Block Swapping
[[musubi-tuner]] supports block swapping — moving transformer blocks between GPU and CPU during training. Saves VRAM at the cost of speed.

### Gradient Checkpointing
All major tools support gradient checkpointing (`--gradient_checkpointing` / `gradient_checkpointing: true`), trading compute for VRAM.

### Latent/Text Encoder Caching
Pre-caching latents and text encoder outputs before training avoids loading these models during training. Required by [[musubi-tuner]], supported by [[ai-toolkit]].

### 4-bit ARA Quantization
[[ai-toolkit]] supports 4-bit ARA quantization for even lower VRAM requirements. Community-reported I2V training on 16GB with this approach.

## Cloud Options

| Provider | Typical GPU | Approximate Cost |
|----------|------------|-----------------|
| RunPod | A100, H100 | $1-3/hr |
| RunComfy | H100, H200 | Varies |
| Lambda Labs | A100, H100 | $1-2/hr |
| Vast.ai | Various | $0.30-2/hr |

## Recommendations

- **Budget local (24GB)**: RTX 3090/4090, train at 512px or lower, use FP8 + gradient checkpointing. Expect multi-hour to multi-day training.
- **Comfortable local (48GB)**: A6000, train at 768px, more flexibility with batch size.
- **Cloud**: Rent an A100/H100 for a few hours when you need speed or higher resolution.

## Related

- [[resolution-guide]] — Resolution choices affect VRAM
- [[training-parameters]] — Batch size and quantization settings
- [[musubi-tuner]] — VRAM optimization flags
- [[ai-toolkit]] — Quantization options
