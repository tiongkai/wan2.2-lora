---
title: Training Tools Comparison
type: synthesis
tags: [comparison, tools, musubi-tuner, ai-toolkit, diffusion-pipe]
created: 2026-05-12
updated: 2026-05-12
---

# Training Tools Comparison

Three major tools support Wan 2.2 LoRA training. This page compares them across key dimensions.

## Feature Comparison

| Feature | [[musubi-tuner]] | [[ai-toolkit]] | [[diffusion-pipe]] |
|---------|-----------------|----------------|-------------------|
| **Author** | kohya-ss | Ostris | tdrussell |
| **Config format** | TOML + CLI | YAML | TOML |
| **Wan 2.2 support** | Full | Full | Partial |
| **Dual-noise handling** | Separate runs | Single run (flags) | Separate runs |
| **Quantization** | FP8 | FP8, 4-bit ARA | Limited |
| **Block swapping** | Yes | No | No |
| **Pre-caching** | Required (manual) | Automatic | Manual |
| **Web UI** | No (SECourses installer) | RunComfy | No |
| **Community size** | Largest | Growing | Small |
| **Documentation** | Good | Good (+ YouTube) | Basic |
| **Multi-GPU** | Via accelerate | Via config | Via DeepSpeed |
| **Beginner friendly** | Medium | **Best** | Lowest |

## Wan 2.2 Dual-Noise Workflow

This is the biggest differentiator:

### Musubi-Tuner
```
Train high-noise expert → Train low-noise expert → Two LoRA files
```
Two separate training runs with different configs. More control but more work.

### AI Toolkit
```
Train (both experts) → One LoRA file
```
Single run with `train_high_noise: true, train_low_noise: true`. Simpler but less control.

### Diffusion-Pipe
```
Train high-noise expert → Train low-noise expert → Two LoRA files
```
Similar to musubi-tuner but with DeepSpeed backend.

## VRAM Efficiency

| Technique | Musubi-Tuner | AI Toolkit | Diffusion-Pipe |
|-----------|-------------|------------|----------------|
| FP8 base | `--fp8_base` | `qtype: qfloat8` | Limited |
| 4-bit | No | `ARA` | No |
| Block swap | Yes | No | No |
| Gradient checkpoint | Yes | Yes | Yes |
| Latent caching | Yes (manual) | Yes (auto) | Yes (manual) |
| **Min viable VRAM** | ~16 GB (with tricks) | ~16 GB (with 4-bit) | ~24 GB |

## Recommendation Matrix

| Scenario | Recommended Tool |
|----------|-----------------|
| First LoRA training ever | [[ai-toolkit]] |
| Maximum control over training | [[musubi-tuner]] |
| Need community help | [[musubi-tuner]] |
| Want simplest Wan 2.2 workflow | [[ai-toolkit]] |
| Multi-GPU distributed training | [[diffusion-pipe]] |
| Lowest VRAM (16 GB) | [[ai-toolkit]] (4-bit) or [[musubi-tuner]] (block swap) |
| Cloud training (RunPod) | [[musubi-tuner]] or [[diffusion-pipe]] |
| Cloud training (RunComfy) | [[ai-toolkit]] |

## Community Activity

- **Musubi-tuner**: Most active GitHub discussions, many Issues with community responses, SECourses presets
- **AI Toolkit**: YouTube tutorials from creator, RunComfy integration, growing community
- **Diffusion-pipe**: Smaller community, more DIY

## Cross-References

- [[musubi-tuner]] — Detailed entity page
- [[ai-toolkit]] — Detailed entity page
- [[diffusion-pipe]] — Detailed entity page
- [[training-parameters]] — Tool-specific config examples
