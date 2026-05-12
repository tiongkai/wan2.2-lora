---
title: "Civitai: WAN 2.2 Local LoRA Training Guide"
type: source
tags: [civitai, local-training, diffusion-pipe, rtx-3090]
source_url: https://civitai.com/articles/18985/wan-22-local-lora-training-guide-windowslinux
source_date: 2025-09
created: 2026-05-12
updated: 2026-05-12
---

# Civitai — WAN 2.2 Local LoRA Training Guide

## Key Takeaways

- Demonstrates Wan 2.2 LoRA training on consumer hardware (RTX 3090)
- Uses [[diffusion-pipe]] as the training tool
- Trains at very low resolution (256×256) to fit in 24GB VRAM
- Practical proof that consumer GPU training works, just slowly

## Training Setup

| Parameter | Value |
|-----------|-------|
| GPU | RTX 3090 (24 GB) |
| Tool | diffusion-pipe |
| Resolution | 256×256 |
| Clip length | 4-8 seconds |
| Dataset size | ~20 clips (12-15 minimum) |
| Training time | 10-20 hours |

## Notable Observations

- "Not exactly user-friendly" — assumes baseline technical experience
- 20 clips is a good target, but 12-15 can work if quality is high
- Small square videos at 256×256 are a practical starting point on consumer hardware

## Cross-References

- [[diffusion-pipe]] — Training tool used
- [[hardware-requirements]] — RTX 3090 training times
- [[resolution-guide]] — Low-resolution training tradeoffs
