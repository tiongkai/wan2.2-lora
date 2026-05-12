---
title: "Apatero Blog: Train Wan 2.2 LoRAs Best Practices"
type: source
tags: [apatero, best-practices, video-clips, motion]
source_url: https://www.apatero.com/blog/train-wan-22-loras-best-practices-2025
source_date: 2025-11
created: 2026-05-12
updated: 2026-05-12
---

# Apatero Blog — Train Wan 2.2 LoRAs Best Practices

## Key Takeaways

- **Video clips are mandatory for motion LoRAs** — cannot train motion from still images
- Training at 1024×1024 with only 15 clips causes severe overfitting
- Temporal consistency in training data is critical
- Provides general parameter ranges for Wan 2.2

## Critical Insight

> "You cannot train effective Wan 2.2 LoRAs on still images hoping the model figures out motion — video clips showing actual movement are mandatory."

This is one of the most important practical findings for video LoRA training.

## Recommended Parameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001-0.0003 |
| Steps | 2000-4000 |
| Batch size | 1-2 |
| Network dim | 32-64 |

## Overfitting Warning

1024×1024 with 15 clips (~150 frames) = severe overfitting. Minimum 35 clips (500+ frames) for 1024px training, or reduce resolution.

## Cross-References

- [[dataset-preparation]] — Video vs image requirements
- [[resolution-guide]] — Resolution-dataset size relationship
- [[training-parameters]] — Parameter ranges from this source
- [[overfitting-and-troubleshooting]] — Resolution-related overfitting
