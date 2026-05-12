---
title: Dual-Noise Architecture (Wan 2.2 MoE)
type: concept
tags: [wan-2-2, architecture, moe, mixture-of-experts, dual-noise]
created: 2026-05-12
updated: 2026-05-12
---

# Dual-Noise Architecture

Wan 2.2 introduces a **Mixture-of-Experts (MoE)** architecture into video diffusion. This is the single biggest architectural change from [[wan-2-1]] and the primary reason LoRA training for Wan 2.2 requires a different approach.

## How It Works

Instead of one diffusion transformer handling all denoising steps, Wan 2.2 uses **two specialized expert models**:

| Expert | Denoising Phase | Focus | Parameters |
|--------|----------------|-------|------------|
| **High-noise** | Early steps (timesteps ~900-1000) | Overall layout, composition, structure | ~14B |
| **Low-noise** | Later steps (timesteps ~0-900) | Fine details, textures, refinement | ~14B |

**Total parameters**: ~27B, but only **14B active per denoising step** — each step routes to one expert.

## Why This Matters for LoRA Training

You need separate LoRA weights for each expert, because they handle fundamentally different aspects of generation:

- The high-noise LoRA teaches your concept's **shape, layout, and structure**
- The low-noise LoRA teaches your concept's **details, textures, and fine features**

If you only train one expert, the LoRA won't fully transfer the concept.

## Tool-Specific Approaches

### AI Toolkit (easiest)

AI Toolkit handles both experts transparently with two flags:

```yaml
model_kwargs:
  train_high_noise: true
  train_low_noise: true
```

This trains a single LoRA file that works for both experts. See [[ai-toolkit]].

### Musubi-Tuner (most control)

Musubi-tuner trains each expert separately. You run two training jobs:

**High-noise expert:**
```bash
--dit wan2.2_t2v_high_noise_14B_fp16.safetensors \
--min_timestep 900 --max_timestep 1000 \
--learning_rate 2e-4
```

**Low-noise expert:**
```bash
--dit wan2.2_t2v_low_noise_14B_fp16.safetensors \
--min_timestep 0 --max_timestep 900 \
--learning_rate 2e-5
```

Note the 10x difference in learning rate — the low-noise expert needs gentler training. This produces **two separate LoRA files** that are both loaded at inference. See [[musubi-tuner]].

### Diffusion-Pipe

Similar to musubi-tuner, requires separate training runs. See [[diffusion-pipe]].

## Model Files

For Wan 2.2 A14B, you need these model files:

| File | Purpose |
|------|---------|
| `wan2.2_t2v_high_noise_14B_fp16.safetensors` | High-noise expert (T2V) |
| `wan2.2_t2v_low_noise_14B_fp16.safetensors` | Low-noise expert (T2V) |
| `wan_2.1_vae.safetensors` | VAE (shared with Wan 2.1) |
| Text encoder model | T5-based encoder |

## Comparison with Wan 2.1

| Aspect | Wan 2.1 | Wan 2.2 |
|--------|---------|---------|
| Architecture | Single transformer | Dual-expert MoE |
| Models to train | 1 | 2 (or 1 with AI Toolkit) |
| Active params/step | 14B | 14B (same efficiency) |
| Total params | 14B | 27B |
| LoRA output | 1 file | 1-2 files depending on tool |
| Training time | 1x | ~2x (two experts to train) |

See [[wan21-vs-wan22]] for a full comparison.

## Related

- [[wan-2-2]] — The full model description
- [[training-parameters]] — How to configure dual-noise training
- [[musubi-tuner]] — Separate training approach
- [[ai-toolkit]] — Unified training approach
