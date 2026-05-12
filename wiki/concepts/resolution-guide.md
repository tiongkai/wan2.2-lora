---
title: Resolution Guide
type: concept
tags: [resolution, training, vram, quality]
created: 2026-05-12
updated: 2026-05-12
---

# Resolution Guide

Training resolution is a critical tradeoff between quality, VRAM, training time, and overfitting risk. Wan 2.2 is built around **720p, 24fps** for T2V/I2V.

## Resolution Tiers

| Resolution | VRAM Needed | Training Time | Detail Quality | Best For |
|-----------|-------------|---------------|----------------|----------|
| 256×256 | ~16 GB | Fast | Low | Experiments, prototyping on 3090 |
| 384×384 | ~18 GB | Moderate | Medium | Budget training with musubi-tuner |
| 512×512 | ~20 GB | Moderate | Medium | Budget hardware, acceptable quality |
| 544×960 | ~22 GB | Moderate | Medium-high | Portrait aspect ratio, good tradeoff |
| 768×768 | ~24 GB | Long | High | **Sweet spot** — optimal balance |
| 1024×1024 | 24+ GB | Very long (2-3x) | Very high | When detail matters, large dataset required |

## The Golden Rule

> **Never train at a resolution higher than your source material.**

Training at 1024px from 720p source videos teaches the LoRA to reproduce compression artifacts and upscaling patterns instead of actual character/style details. Match training resolution to source resolution or go lower.

## Resolution and Dataset Size

Higher resolutions need proportionally larger datasets to avoid overfitting:

| Resolution | Minimum Clips | Recommended Clips |
|-----------|--------------|-------------------|
| 256-384px | 12-15 | 20 |
| 512px | 15-20 | 20-30 |
| 768px | 20-25 | 30 |
| 1024px | 35+ | 50+ |

Training at 1024×1024 with only 15 clips (~150 frames) causes **severe overfitting** — the LoRA reproduces training examples exactly instead of generalizing. [apatero-best-practices]

## Aspect Ratios

- **Square** (768×768, 512×512) — general purpose, most common
- **Portrait** (544×960) — good for character/person LoRAs
- **Landscape** — less common for training, can work for scene LoRAs

Some tools support **bucketing** (variable aspect ratios within a training run). [[musubi-tuner]] supports this with `enable_bucket = true`.

## Training vs Inference Resolution

You can train at a lower resolution and generate at a higher resolution during inference. For example:
- Train at 544×960
- Generate at 720p or higher

This works because LoRA learns concept features, not resolution-specific patterns (as long as training data was clean).

## Frames and Resolution

For video LoRAs, both spatial resolution and frame count affect VRAM:

| Resolution | Frames | Approximate VRAM |
|-----------|--------|-----------------|
| 384×384 | 57 frames | ~20 GB |
| 512×768 | 81 frames | ~24 GB |
| 512×512 | 33 frames | ~18 GB |
| 360×360 | 65 frames | ~16 GB |

## Recommendations

- **Start at 512-768px** for your first LoRA training run
- **Use 768×768** if you have 24GB VRAM and a good dataset (20+ clips)
- **Drop to 384-512px** if VRAM is tight or dataset is small
- **Only use 1024px** if you have 35+ high-quality clips and hardware headroom
- **Match source resolution** — never exceed it

## Related

- [[hardware-requirements]] — How resolution affects VRAM
- [[dataset-preparation]] — Dataset size requirements per resolution
- [[overfitting-and-troubleshooting]] — Resolution-related overfitting
