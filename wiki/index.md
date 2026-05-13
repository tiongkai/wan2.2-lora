---
title: Wiki Index
type: index
created: 2026-05-12
updated: 2026-05-12
---

# Wiki Index

Content catalog for the Wan 2.2 LoRA Training Wiki. Read this first on every query.

## Entry Point

- [Overview](overview.md) — Start here. Quick-start path, LoRA types, navigation guide.

## Concepts

- [LoRA](concepts/lora.md) — What LoRA is, how it works in video diffusion, key parameters (rank, alpha, LR), LoRA vs full fine-tuning.
- [Dataset Preparation](concepts/dataset-preparation.md) — File formats, images vs video, dataset size by type, video clip requirements, config examples.
- [Captioning](concepts/captioning.md) — Caption structure, trigger tokens, length guidelines, caption dropout, auto-captioning tools.
- [Training Parameters](concepts/training-parameters.md) — Comprehensive reference: settings by LoRA type, tool-specific configs (musubi-tuner, AI Toolkit, diffusion-pipe), parameter deep dives, source contradictions.
- [Dual-Noise Architecture](concepts/dual-noise-architecture.md) — Wan 2.2's MoE system: high-noise vs low-noise experts, tool-specific approaches, model files needed.
- [Hardware Requirements](concepts/hardware-requirements.md) — VRAM by training type, GPU comparison table, training time estimates, VRAM optimization techniques, cloud options.
- [Resolution Guide](concepts/resolution-guide.md) — Resolution tiers, the golden rule (never exceed source), resolution-dataset size relationship, frames and resolution VRAM.
- [Multi-GPU Training](concepts/multi-gpu-training.md) — 2×24GB strategies: parallel dual-noise (musubi-tuner) vs pipeline parallelism (diffusion-pipe). DDP caveats.
- [Overfitting and Troubleshooting](concepts/overfitting-and-troubleshooting.md) — Detection tells, 6 common issues with fixes, Wan 2.2-specific behavior, checkpoint management, loss monitoring.

## Entities

- [Wan 2.2](entities/wan-2-2.md) — Current model. MoE dual-expert, A14B variants (T2V, I2V, S2V, T2I), model files, overfitting behavior.
- [Wan 2.1](entities/wan-2-1.md) — Predecessor model. Single DiT, 1.3B/14B variants, simpler training workflow.
- [Musubi-Tuner](entities/musubi-tuner.md) — kohya-ss training tool. TOML config, most community support, separate dual-noise runs, FP8/block-swap.
- [AI Toolkit](entities/ai-toolkit.md) — Ostris training tool. YAML config, simplest Wan 2.2 workflow, unified dual-noise, quantization.
- [Diffusion-Pipe](entities/diffusion-pipe.md) — DeepSpeed training tool. Advanced users, multi-GPU, TOML config.
- [ComfyUI](entities/comfyui.md) — Node-based inference UI. Not for training; used to test/generate with trained LoRAs.

## Source Summaries

- [WaveSpeed Training Settings](sources/wavespeed-training-settings.md) — Conservative identity/style settings, overfitting detection, regularization images. (Jan 2026)
- [Civitai Local Training Guide](sources/civitai-local-training-guide.md) — RTX 3090 training with diffusion-pipe, 256×256, 10-20 hours. (Sep 2025)
- [Apatero Best Practices](sources/apatero-best-practices.md) — Video clips mandatory for motion, resolution-dataset sizing. (Nov 2025)
- [Apatero Person LoRA Pro](sources/apatero-person-lora-pro.md) — Person LoRA workflow, caption structure, consumer hardware timelines. (Dec 2025)
- [StableDiffusion Tutorials](sources/stablediffusion-tutorials-wan22.md) — AI Toolkit guide, dual-transformer explanation. (Dec 2025)
- [AMD ROCm Fine-Tuning](sources/amd-rocm-finetuning.md) — Wan 2.2 on AMD GPUs, architecture deep-dive. (2025)
- [Ostris Wan 2.2 I2V Tutorial](sources/ostris-wan22-i2v-tutorial.md) — Authoritative AI Toolkit video tutorial, orbital shot demo. (Aug 2025)
- [Musubi Community Discussions](sources/musubi-community-discussions.md) — Community-validated settings, dual-noise workflows, known issues. (2025)
- [ArXiv Cinematic Fine-Tuning](sources/arxiv-cinematic-finetuning.md) — Academic paper: two-stage pipeline, style-motion decoupling. (2025)
- [YouTube Tutorials Collection](sources/youtube-tutorials-collection.md) — Catalog of 12+ video tutorials across tools and versions.

## Synthesis

- [Training Tools Comparison](synthesis/training-tools-comparison.md) — Side-by-side: musubi-tuner vs AI Toolkit vs diffusion-pipe. Features, VRAM, workflow, recommendations.
- [Wan 2.1 vs Wan 2.2](synthesis/wan21-vs-wan22.md) — Architecture, parameters, workflow, and migration guide.
- [Plan Critique: Revised Configs](synthesis/plan-critique-revised-configs.md) — Corrected AI Toolkit configs for violence detection LoRA plan. Fixes: dual-noise, resolution, rank, LR, captioning, regularization.
