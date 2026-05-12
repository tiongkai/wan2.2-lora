---
title: Wan 2.2
type: entity
tags: [model, wan-2-2, video-generation, moe]
created: 2026-05-12
updated: 2026-05-12
---

# Wan 2.2

Wan 2.2 is a large-scale video generation model built on the diffusion transformer (DiT) paradigm. It is the successor to [[wan-2-1]], introducing a [[dual-noise-architecture]] with Mixture-of-Experts (MoE).

## Architecture

- **Type**: Diffusion Transformer (DiT) with MoE
- **Experts**: Two specialized models — high-noise (~14B params) and low-noise (~14B params)
- **Total parameters**: ~27B (but only 14B active per denoising step)
- **Design**: High-noise expert handles layout/composition; low-noise expert handles fine details
- **Base resolution**: 720p, 24fps

## Variants

| Variant | Task | Model ID |
|---------|------|----------|
| T2V-A14B | Text-to-video | `Wan-AI/Wan2.2-T2V-A14B` |
| I2V-A14B | Image-to-video | `Wan-AI/Wan2.2-I2V-A14B` |
| S2V-14B | Subject-to-video | `Wan-AI/Wan2.2-S2V-14B` |
| T2I-A14B | Text-to-image | `Wan-AI/Wan2.2-T2I-A14B` |
| 5B | Smaller, faster variant | Various |

## Model Files

| File | Purpose |
|------|---------|
| `wan2.2_t2v_high_noise_14B_fp16.safetensors` | High-noise expert (T2V) |
| `wan2.2_t2v_low_noise_14B_fp16.safetensors` | Low-noise expert (T2V) |
| `wan_2.1_vae.safetensors` | VAE (shared with Wan 2.1) |
| T5-based text encoder | Text conditioning |

## Key Characteristics

- "Behaves like a very opinionated SDXL checkpoint" — pushes toward crisp portraits, smooth gradients, cinematic light
- No CLIP model required (unlike [[wan-2-1]])
- Diffusers integration available (`WanPipeline`)
- Works with ComfyUI for inference

## LoRA Training

See [[training-parameters]] for recommended settings. Key considerations:
- Must train **both** high-noise and low-noise experts
- [[ai-toolkit]] handles both with `train_high_noise`/`train_low_noise` flags
- [[musubi-tuner]] trains them separately with different timestep ranges and learning rates
- See [[dual-noise-architecture]] for full details

## Overfitting Behavior

Wan 2.2 has specific overfitting tendencies:
- Color clustering (amplifies dataset color bias)
- Background homogenization (defaults to shallow DoF/bokeh)
- Skin plasticity (over-smoothing)
- See [[overfitting-and-troubleshooting]]

## HuggingFace

Models available at: `Wan-AI/` organization on HuggingFace

## Related

- [[wan-2-1]] — Predecessor model
- [[wan21-vs-wan22]] — Detailed comparison
- [[dual-noise-architecture]] — MoE architecture details
