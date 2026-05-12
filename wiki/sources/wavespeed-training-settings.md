---
title: "WaveSpeed Blog: WAN 2.2 LoRA Training Settings"
type: source
tags: [wavespeed, training-settings, identity, style, overfitting]
source_url: https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/
source_date: 2026-01
created: 2026-05-12
updated: 2026-05-12
---

# WaveSpeed Blog — WAN 2.2 LoRA Training Settings

## Key Takeaways

- Provides the most conservative and well-tested settings for identity and style LoRAs
- Introduces the insight that **"WAN hides overfit behind pretty samples"** — a critical watchpoint
- Recommends **regularization images** as a specific fix for Wan 2.2's opinionated aesthetic
- Distinguishes clearly between identity and style LoRA configurations

## Recommended Settings

### Identity LoRA
- Rank 16, alpha 16
- LR 5e-5
- AdamW, weight_decay 0.01
- Cosine scheduler, 5% warmup
- Batch size 2-4 (A100/4090)
- 12-20 images

### Style LoRA
- Rank 32, alpha 32
- LR 7e-5 to 1e-4
- 30-50 images

## Overfitting Insights

- **Prompt inertia**: output barely changes with different prompts
- **Skin plasticity**: pores vanish, especially cheeks/foreheads
- **Color clustering**: warm dataset → everything goes warm
- **Background homogenization**: defaults to shallow DoF and soft bokeh

## Key Fix: Regularization Images

Mix 10-20% of training batches with regularization images throughout training (not just at start). This gives the model "permission" to maintain base model variety while the LoRA holds identity.

## Notable Quote

> "WAN punishes high LR with plasticky skin and loss spikes."

## Cross-References

- [[training-parameters]] — Settings adopted from this source
- [[overfitting-and-troubleshooting]] — Overfitting detection methods from this source
