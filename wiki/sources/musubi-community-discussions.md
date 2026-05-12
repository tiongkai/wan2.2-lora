---
title: "Musubi-Tuner Community Discussions"
type: source
tags: [musubi-tuner, community, github, settings, experiments]
source_url: https://github.com/kohya-ss/musubi-tuner/discussions/182
source_date: 2025
created: 2026-05-12
updated: 2026-05-12
---

# Musubi-Tuner Community Discussions

## Key Takeaways

- Community-validated training settings from real experiments
- Multiple approaches to learning rate / discrete flow shift tradeoffs
- Practical dual-noise training workflows (high/low noise split)
- Honest reports of what works and what doesn't

## Sources

- **Discussion #182**: "WAN training — The rules of the trade"
- **Discussion #455**: "Wan 2.2 training and general discussion"

## Community-Validated Settings

### Standard Approach
- LR: 2e-4
- Timestep sampling: shift
- Discrete flow shift: 7.0 (faster training)
- If details not learned well: lower shift to ~3.0

### Alternative (Lower LR + LoRA Plus)
- LR: 3e-5 with `loraplus_lr_ratio=4`
- Steps: 1600
- Discrete flow shift: 12
- Both high and low noise back-to-back in ~12 hours
- Resolutions: 360×360×65 and 512×512×33 for high noise

### Dual-Noise Split
- High-noise: `--min_timestep 900 --max_timestep 1000`, LR 2e-4
- Low-noise: `--min_timestep 0 --max_timestep 900`, LR 2e-5
- 10x learning rate difference between experts

## Civitai Community Settings (Discussion #455 reference)

From the Civitai WAN 2.2 workflow TLDR:
- Task: `t2v-A14B`
- LR: 3e-4
- Network dim: 16, alpha: 16
- Discrete flow shift: 1.0
- Optimizer: adamw with weight_decay 0.1
- Scheduler: polynomial (power 8)
- max_grad_norm: 0
- FP16 mixed precision

## Notable Issues

- **Issue #569**: Wan 2.2 training on high AND low noise generating only one LoRA
- **Issue #621**: Wan 2.2 I2V LoRA training having no effect (comparison with AI Toolkit)
- **Issue #416**: I2V LoRA throws errors but T2V trains fine

## Cross-References

- [[musubi-tuner]] — The tool discussed
- [[training-parameters]] — Settings adopted from these discussions
- [[dual-noise-architecture]] — Dual-noise split settings
