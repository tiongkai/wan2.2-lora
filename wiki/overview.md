---
title: Wan 2.2 LoRA Training — Overview
type: overview
tags: [overview, wan-2-2, lora, video-generation]
created: 2026-05-12
updated: 2026-05-12
---

# Wan 2.2 LoRA Training Wiki

This wiki is a compounding knowledge base on training LoRA (Low-Rank Adaptation) models for the **Wan 2.1/2.2** video generation model family. It covers tools, techniques, dataset preparation, training configurations, and community-validated best practices.

## Why LoRA?

Wan 2.2 is a powerful video generation model, but its pretrained weights may not produce the specific characters, styles, or motions you need. LoRA lets you adapt the model by training less than 1% of its parameters — making domain-specific fine-tuning possible on a single consumer GPU without retraining the full 14B+ backbone.

## What Makes Wan 2.2 Different

Wan 2.2 introduces a [[dual-noise-architecture]] — a Mixture-of-Experts (MoE) system with two 14B-parameter expert models:

- **High-noise expert**: handles early denoising (overall layout and composition)
- **Low-noise expert**: handles late denoising (fine details and refinement)

This means LoRA training for Wan 2.2 requires training **both experts**, either separately (using [[musubi-tuner]]) or simultaneously (using [[ai-toolkit]]). This is the single biggest difference from [[wan-2-1]] LoRA training.

## Quick Start Path

1. **Choose a tool**: [[ai-toolkit]] (easiest, YAML config) or [[musubi-tuner]] (most flexible, community-driven) — see [[training-tools-comparison]]
2. **Prepare your dataset**: 15-30 video clips or images with text captions — see [[dataset-preparation]] and [[captioning]]
3. **Set training parameters**: start with community-validated defaults — see [[training-parameters]]
4. **Match resolution to hardware**: 768x768 is the sweet spot for 24GB VRAM — see [[resolution-guide]] and [[hardware-requirements]]
5. **Train and iterate**: save intermediate checkpoints, test generalization — see [[overfitting-and-troubleshooting]]

## LoRA Types

| Type | Purpose | Dataset | Key Consideration |
|------|---------|---------|-------------------|
| **Identity/Character** | Consistent person or character | 12-20 images/clips of subject | Use trigger tokens, regularization images |
| **Style** | Visual aesthetic (anime, cinematic, etc.) | 30-50 images/clips | Higher rank (32), higher LR |
| **Motion** | Camera movement or action type | 15-30 video clips (mandatory) | Cannot learn motion from still images |
| **I2V** | Image-to-video generation | Video clips + reference frames | Requires I2V model variant |

## Key Concepts

- [[lora]] — What LoRA is and how it works in video diffusion
- [[dataset-preparation]] — Preparing training data
- [[captioning]] — Writing effective captions with trigger tokens
- [[training-parameters]] — Comprehensive parameter reference
- [[dual-noise-architecture]] — Wan 2.2's unique MoE system
- [[hardware-requirements]] — GPU and VRAM requirements
- [[resolution-guide]] — Choosing the right training resolution
- [[overfitting-and-troubleshooting]] — Diagnosing and fixing common issues

## Tools

- [[ai-toolkit]] — Ostris's YAML-based training toolkit (recommended for beginners)
- [[musubi-tuner]] — Kohya-ss's flexible training scripts (most community support)
- [[diffusion-pipe]] — DeepSpeed-based trainer (advanced users)
- [[comfyui]] — Node-based inference UI for testing trained LoRAs

## Models

- [[wan-2-2]] — Current generation (MoE dual-expert, A14B)
- [[wan-2-1]] — Previous generation (single model, 1.3B/14B)

## Synthesis & Comparisons

- [[training-tools-comparison]] — Side-by-side comparison of training tools
- [[wan21-vs-wan22]] — What changed between model versions for LoRA training
