---
title: LoRA (Low-Rank Adaptation)
type: concept
tags: [lora, fine-tuning, fundamentals]
created: 2026-05-12
updated: 2026-05-12
---

# LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that adapts a pretrained model by injecting small, trainable low-rank matrices into its attention layers. Instead of retraining all parameters, LoRA modifies **less than 1%** of the model's weights — enabling domain adaptation on a single consumer GPU.

## How It Works

A full weight matrix W (e.g., 4096x4096 in an attention layer) is approximated as:

```
W' = W + ΔW = W + B × A
```

Where B (4096 × rank) and A (rank × 4096) are small trainable matrices. Only B and A are updated during training; the original W stays frozen.

## Key Parameters

| Parameter | What It Controls | Typical Values |
|-----------|-----------------|----------------|
| **Rank (dim)** | Capacity of the adaptation. Higher = more expressive but more VRAM and overfitting risk | 16 (identity), 32 (style), 32-64 (general) |
| **Alpha** | Scaling factor for the LoRA update. Usually set equal to rank | Match rank value |
| **Learning Rate** | How fast the LoRA weights update | 5e-5 to 3e-4 depending on use case |

## LoRA in Video Diffusion Models

In [[wan-2-2]] and [[wan-2-1]], LoRA modules are injected into the attention layers of the diffusion transformer (DiT). This lets you teach the model new:

- **Characters/identities** — consistent faces, bodies, distinctive features
- **Styles** — visual aesthetics (anime, cinematic, painterly)
- **Motions** — camera movements (orbital, dolly zoom), action types
- **I2V behaviors** — how to animate a reference image

## LoRA vs Full Fine-Tuning

| Aspect | LoRA | Full Fine-Tuning |
|--------|------|-----------------|
| Parameters trained | < 1% | 100% |
| VRAM required | 12-24 GB | 80+ GB |
| Training time | Hours to days | Days to weeks |
| Risk of catastrophic forgetting | Low | High |
| Output file size | ~50-200 MB | Full model size (14B+) |
| Composability | Can stack/merge multiple LoRAs | One model at a time |

## Wan 2.2 Considerations

Because [[wan-2-2]] uses a [[dual-noise-architecture]], you need LoRA modules for **both** the high-noise and low-noise expert models. See [[training-parameters]] for tool-specific approaches to handling this.

## Related

- [[training-parameters]] — Detailed parameter recommendations
- [[dataset-preparation]] — What data to feed the training
- [[overfitting-and-troubleshooting]] — When LoRA training goes wrong
