---
title: Wan 2.1 vs Wan 2.2 for LoRA Training
type: synthesis
tags: [comparison, wan-2-1, wan-2-2, migration]
created: 2026-05-12
updated: 2026-05-12
---

# Wan 2.1 vs Wan 2.2 for LoRA Training

This page covers the key differences between Wan 2.1 and Wan 2.2 that affect LoRA training.

## Architecture Comparison

| Aspect | [[wan-2-1]] | [[wan-2-2]] |
|--------|-----------|-----------|
| Architecture | Single DiT | MoE (dual-expert DiT) |
| Active parameters | 14B | 14B per step |
| Total parameters | 14B | ~27B |
| Experts | 1 | 2 (high-noise + low-noise) |
| Smallest variant | 1.3B | 5B |
| CLIP required | Yes | No |
| VAE | wan_2.1_vae | Same VAE (shared) |

## LoRA Training Differences

| Aspect | Wan 2.1 | Wan 2.2 |
|--------|---------|---------|
| Models to train | 1 | 2 (or 1 with AI Toolkit) |
| Training runs | 1 | 1-2 depending on tool |
| LoRA output files | 1 | 1-2 depending on tool |
| Training time | 1x | ~2x (two experts) |
| Complexity | Lower | Higher |
| Minimum VRAM | 24 GB (14B T2V) | 24 GB (14B T2V) |

## Parameter Differences

| Parameter | Wan 2.1 | Wan 2.2 |
|-----------|---------|---------|
| Learning rate | ~0.0002 | 0.0001-0.0003 (varies by expert) |
| Steps | 3000-3500 | 2000-5000 |
| Dataset size | 10-15 images | 10-30 images/clips |
| Resolution | 512px common | 768px sweet spot |
| Task flag (musubi) | `t2v-14B` | `t2v-A14B` |

## Dual-Noise Impact

The biggest practical change. In Wan 2.1, you train once against one model. In Wan 2.2:

**With [[musubi-tuner]]**: Two separate training runs
- High-noise: `--min_timestep 900 --max_timestep 1000`, LR 2e-4
- Low-noise: `--min_timestep 0 --max_timestep 900`, LR 2e-5

**With [[ai-toolkit]]**: One training run
- `train_high_noise: true, train_low_noise: true`

## Quality Differences

Wan 2.2 produces higher quality base outputs but is "more opinionated":
- Pushes toward crisp portraits, cinematic light, smooth gradients
- Color clustering from training data gets amplified
- Backgrounds tend to homogenize to shallow DoF
- Regularization images more important than with Wan 2.1

## Migration Guide

If you're moving from Wan 2.1 to Wan 2.2 training:

1. **Tool**: Same tools work, but update to latest versions
2. **Dataset**: Can reuse, but consider adding regularization images
3. **Config**: Must update task flag (`t2v-14B` → `t2v-A14B`)
4. **Workflow**: Plan for dual-noise training (two runs or AI Toolkit)
5. **CLIP**: No longer needed for pre-caching
6. **Monitoring**: Watch more carefully for overfit (Wan 2.2 hides it better)

## When to Use Wan 2.1

Wan 2.1 is still useful for:
- **Experimentation**: 1.3B variant trains in ~2.5 hours
- **Testing datasets**: Quick validation before committing to Wan 2.2
- **Simpler workflow**: When dual-noise complexity isn't worth it
- **Mature guides**: More community-tested configurations available

## Cross-References

- [[wan-2-1]] — Full model details
- [[wan-2-2]] — Full model details
- [[dual-noise-architecture]] — The MoE system explained
- [[training-parameters]] — Settings for both versions
