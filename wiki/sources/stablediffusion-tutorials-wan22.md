---
title: "StableDiffusionTutorials: Wan 2.2 LoRA Training"
type: source
tags: [tutorial, ai-toolkit, dual-transformer, windows, linux]
source_url: https://www.stablediffusiontutorials.com/2025/10/wan2.2-lora-training.html
source_date: 2025-12
created: 2026-05-12
updated: 2026-05-12
---

# StableDiffusionTutorials — Wan 2.2 LoRA Training

## Key Takeaways

- Uses [[ai-toolkit]] for training
- Emphasizes the key Wan 2.2 difference: **two transformer models** (high/low noise) that must be trained separately
- Recommends 3000-5000 training steps
- Training process is "pretty much similar to Wan 2.1" except for the dual transformers

## Practical Notes

- Works on both Windows and Linux
- Due to the dual-transformer architecture, requires a more powerful machine setup
- Also has a companion guide for Wan 2.1 training at the same site

## Companion Guide

The site also has a Wan 2.1 LoRA training guide: https://www.stablediffusiontutorials.com/2025/03/wan-lora-train.html

That guide documents:
- Training a style LoRA with 14 images on RTX 3090
- 512 resolution, 3500 steps, ~2.5 hours
- Uses Wan 2.1 1.3B model

## Cross-References

- [[ai-toolkit]] — Training tool used
- [[dual-noise-architecture]] — The key Wan 2.2 difference highlighted
- [[wan-2-1]] — Companion guide for previous version
