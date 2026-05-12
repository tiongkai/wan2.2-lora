---
title: Wan 2.1
type: entity
tags: [model, wan-2-1, video-generation]
created: 2026-05-12
updated: 2026-05-12
---

# Wan 2.1

Wan 2.1 is a video generation model built on the diffusion transformer paradigm, released **February 25, 2025**. It is the predecessor to [[wan-2-2]].

## Architecture

- **Type**: Diffusion Transformer (DiT) — single model (no MoE)
- **Variants**: 1.3B (lightweight) and 14B (full)
- **Integration**: Added to HuggingFace Diffusers on March 3, 2025

## Variants

| Variant | Task | Parameters | Model ID |
|---------|------|-----------|----------|
| T2V-1.3B | Text-to-video | 1.3B | `Wan-AI/Wan2.1-T2V-1.3B` |
| T2V-14B | Text-to-video | 14B | `Wan-AI/Wan2.1-T2V-14B` |
| I2V-14B | Image-to-video | 14B | `Wan-AI/Wan2.1-I2V-14B` |
| T2V-14B-Diffusers | Diffusers format | 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |

## Key Differences from Wan 2.2

- Single model (not dual-expert) — simpler LoRA training
- Requires CLIP model (Wan 2.2 does not)
- 1.3B variant available (faster experiments, lower VRAM)
- Training scripts/guides are more mature and widely tested

## Official Resources

- **GitHub**: https://github.com/Wan-Video/Wan2.1
- **HuggingFace**: Wan-AI organization
- Officially references DiffSynth-Studio for additional features

## LoRA Training

Simpler than Wan 2.2 — single model, no dual-noise handling needed.

Example with [[musubi-tuner]]:
```bash
--task t2v-1.3B   # or t2v-14B, i2v-14B
--dit wan2.1_xxx_bf16.safetensors
```

The 1.3B variant is excellent for experimentation:
- Lower VRAM requirements
- Faster training (~2.5 hours for 3500 steps at 512px on RTX 3090)
- Good for testing dataset quality and training parameters before scaling to 14B

## Related

- [[wan-2-2]] — Successor model
- [[wan21-vs-wan22]] — Detailed comparison
- [[training-parameters]] — Training configuration
