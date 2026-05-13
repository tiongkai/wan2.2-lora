---
title: Multi-GPU Training
type: concept
tags: [multi-gpu, distributed, pipeline-parallelism, dual-noise, hardware]
created: 2026-05-13
updated: 2026-05-13
---

# Multi-GPU Training

With 2×24GB GPUs, there are two strategies for Wan 2.2 LoRA training. They solve different problems: one saves time, the other unlocks higher quality settings.

## Strategy 1: Parallel Dual-Noise (Recommended)

Since [[wan-2-2]]'s [[dual-noise-architecture]] requires training both high-noise and low-noise experts, run them simultaneously on separate GPUs:

```bash
# GPU 0: high-noise expert (layout/composition)
CUDA_VISIBLE_DEVICES=0 accelerate launch wan_train_network.py \
  --dit wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --min_timestep 900 --max_timestep 1000 \
  --learning_rate 2e-4 ...

# GPU 1: low-noise expert (details/refinement)
CUDA_VISIBLE_DEVICES=1 accelerate launch wan_train_network.py \
  --dit wan2.2_t2v_low_noise_14B_fp16.safetensors \
  --min_timestep 0 --max_timestep 900 \
  --learning_rate 2e-5 ...
```

**Tool**: [[musubi-tuner]] (trains experts separately by design)

**Pros**:
- No distributed training complexity — each GPU runs independently
- Halves wall-clock time (both experts train simultaneously)
- Each GPU stays within 24GB using standard single-GPU optimizations (FP8, gradient checkpointing)
- Proven, reliable — no multi-GPU bugs

**Cons**:
- Resolution/frame count still limited to single-GPU capacity (512×768, 33 frames)
- Produces two LoRA files per category (both loaded at inference)

## Strategy 2: Pipeline Parallelism (Higher Resolution)

Split the model layers across both GPUs using [[diffusion-pipe]]'s DeepSpeed pipeline parallelism. Effective VRAM: ~48GB.

```toml
# diffusion-pipe config
pipeline_stages = 2
activation_checkpointing = true
blocks_to_swap = 16
transformer_dtype = "float8"
```

```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
deepspeed --num_gpus=2 train.py --deepspeed --config config.toml
```

Optional manual layer partitioning:
```toml
partition = "manual"
partition_split = [18]    # layers 0-17 → GPU 0, rest → GPU 1
```

**Tool**: [[diffusion-pipe]]

**Pros**:
- ~48GB effective VRAM — enables 720×1280 resolution and 57-81 frames
- A100-like capacity from consumer hardware
- Single training run per expert

**Cons**:
- More complex setup (DeepSpeed, NCCL config)
- Some users still hit OOM on 2×24GB for I2V 14B — FP8 + block swapping still needed
- RTX 4000 series requires `NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"` environment variables

## Avoid: Musubi-Tuner DDP

[[musubi-tuner]] supports multi-GPU via Accelerate DDP (`accelerate config` → "Multi-GPU"), but this mode is buggy:
- `--blocks_to_swap` conflicts with DDP
- Users report OOM or only GPU 0 being used while GPU 1 sits idle
- Not recommended for production training

## Recommendation Matrix

| Scenario | Strategy | Tool |
|----------|----------|------|
| Default (2×24GB, get training done fast) | Parallel dual-noise | [[musubi-tuner]] |
| Need higher resolution (720p+) | Pipeline parallelism | [[diffusion-pipe]] |
| Single GPU available | Unified training | [[ai-toolkit]] |
| Single GPU, want control | Sequential dual-noise | [[musubi-tuner]] |

## Related

- [[dual-noise-architecture]] — Why two experts need training
- [[hardware-requirements]] — VRAM requirements per configuration
- [[musubi-tuner]] — Parallel dual-noise tool
- [[diffusion-pipe]] — Pipeline parallelism tool
- [[training-tools-comparison]] — Full tool comparison
