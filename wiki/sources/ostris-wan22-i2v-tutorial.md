---
title: "YouTube: Ostris — Train Wan 2.2 I2V 14B LoRA with AI Toolkit"
type: source
tags: [youtube, ostris, ai-toolkit, i2v, tutorial, video]
source_url: https://www.youtube.com/watch?v=2d6A_l8c_x8
source_date: 2025-08-20
created: 2026-05-12
updated: 2026-05-12
---

# Ostris — Train Wan 2.2 I2V 14B LoRA with AI Toolkit

## Key Takeaways

- Step-by-step video tutorial by the creator of [[ai-toolkit]]
- Trains an **orbital shot** I2V LoRA as a practical demo
- Covers dataset design for motion, style, and character LoRAs
- Explains Wan 2.2's dual high/low-noise experts
- Shows multi-stage training settings
- Recommends frame counts and resolutions for different LoRA types
- Can run on 24GB local GPUs or cloud (H100/H200 via RunComfy)

## Related Tutorials by Ostris

| Video | Model | Type | Date |
|-------|-------|------|------|
| [Wan 2.2 I2V 14B](https://www.youtube.com/watch?v=2d6A_l8c_x8) | Wan 2.2 14B | I2V | Aug 2025 |
| [Wan 2.2 5B I2V](https://www.youtube.com/watch?v=9ATaQdin1sA) | Wan 2.2 5B | I2V | Jul 2025 |
| [Wan 2.1 I2V (dolly zoom)](https://www.youtube.com/watch?v=_swwOO95FrY) | Wan 2.1 14B | I2V | Jul 2025 |
| [Wan 2.1 Character (images)](https://www.youtube.com/watch?v=oJdT5dzrNEY) | Wan 2.1 | Character | Jul 2025 |

## Practical Details

- The orbital shot demo shows motion LoRA training workflow end-to-end
- Demonstrates AI Toolkit's unified high/low noise handling
- Shows how to configure `train_high_noise: true, train_low_noise: true`

## Why This Source Matters

Ostris is the creator of AI Toolkit, making this the **authoritative tutorial** for that tool's Wan 2.2 workflow. The orbital shot demo is a practical, reproducible example.

## Cross-References

- [[ai-toolkit]] — The tool demonstrated
- [[dual-noise-architecture]] — Explained in the tutorial
- [[training-parameters]] — Config examples from this tutorial
- [[dataset-preparation]] — Dataset design guidance
