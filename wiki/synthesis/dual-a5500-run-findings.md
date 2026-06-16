---
title: End-to-End Run Findings — 2× RTX A5500 (24GB)
type: synthesis
tags: [hardware, vram, training, generation, comfyui, musubi-tuner, a5500]
created: 2026-06-06
updated: 2026-06-06
---

# End-to-End Run Findings — 2× RTX A5500 (24GB)

Operational findings from the **first full fighting-LoRA run** on the project rig: **2× NVIDIA RTX A5500 (24 GB each), 128 GB system RAM**. Covers what actually worked (and what OOM'd) for both training (with [[musubi-tuner]]) and generation (with [[comfyui]]) on 24 GB cards. These are first-hand results from this hardware — they refine the generic guidance in [[hardware-requirements]] and [[multi-gpu-training]].

## TL;DR

| Stage | What works on 24 GB | Cost |
|---|---|---|
| Train | `blocks_to_swap 30` + `expandable_segments` (NOT the default 20) | ~25 h for 3000 steps, dual-noise in parallel |
| Generate | `--lowvram --disable-cuda-malloc` + text encoder on CPU | ~350 s per 2 s clip |
| Models | **Stock Wan2.2 + trained LoRA** (no uncensored Remix encoder) | — |
| Clip length | 33 frames @ 16 fps = **~2 s** (by design for 24 GB) | 5 s needs 81 frames → much more VRAM/time |

## Training (musubi-tuner, dual-noise)

High-noise expert → GPU 0, low-noise expert → GPU 1, run in parallel (see [[multi-gpu-training]]).

- **The default `blocks_to_swap 20` OOMs at the first training step** — each expert peaks ~270 MiB over the 24 GB ceiling. VRAM peaks ~21 GB during steps (more than the ~14 GB seen between steps). Note GPU 0 also carries ~1.2 GB of desktop usage, tightening its margin.
- **Fix (zero quality impact):** `--blocks_to_swap 30` (more of the 14B model offloaded to CPU RAM) + `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reclaims ~1.6 GB reserved-but-unallocated fragmentation). Lowering swap below ~28 risks OOM again given the ~21 GB peaks.
- **Throughput:** ~29–30 s/step → **~25 h for 3000 steps** (both experts in parallel). Config: 512×768, 33 frames, rank/alpha 32, fp16 + fp8_base, adamw8bit, gradient checkpointing.
- **Loss behavior (matches [[overfitting-and-troubleshooting]]):** the **high-noise expert** (timesteps 900–1000, learns motion/composition) dropped steadily 0.30 → ~0.09 across the run and was the meaningful convergence signal. The **low-noise expert** (timesteps 0–900, refinement) stayed ~flat at 0.09 the whole time — flatness is normal there, not a problem. Checkpoints every 300 steps; the late checkpoints (1800–2400) sit in the recommended stop range, but loss was still descending at 3000.

## Generation (ComfyUI, stock Wan2.2 + LoRA)

A 14B T2V + LoRA does **not** fit on one 24 GB card by default — it OOMs during **LoRA patching** (`ERROR lora ... Allocation on device` → `Got an OOM`). Root cause is memory pressure: the **umt5-xxl text encoder (~11 GB)** stays resident while the **14B UNet (~14 GB)** loads → >24 GB.

**Three fixes, all required, now baked into `scripts/run_comfyui.sh` and the workflow JSON:**
1. `--lowvram` — stream the 14B UNet from the 128 GB system RAM.
2. `--disable-cuda-malloc` — use the native caching allocator so `expandable_segments` takes effect (ComfyUI defaults to `cudaMallocAsync`, which ignored it). Fixes the patching-time OOM.
3. **Text encoder on CPU** — `CLIPLoader` node `device: cpu` frees ~11 GB of VRAM. The umt5 encode runs once per prompt; CPU is fine.

With all three, one generation fits and is **stable at ~350 s/clip**. If it still OOMs, escalate to `--novram`.

**Workflow shape (validated against ComfyUI `/object_info`):** dual `UNETLoader` (fp8_e4m3fn) → `LoraLoaderModelOnly` per expert @ strength 0.75 → `ModelSamplingSD3` shift 8.0 → two `KSamplerAdvanced` relay (high-noise steps 0–10 with leftover noise, low-noise steps 10–20), cfg 5, euler/simple → `VAEDecode` → native `CreateVideo` (16 fps) → `SaveVideo` (mp4/h264). `comfy-video-helper-suite` was NOT used (it needs cv2/imageio-ffmpeg); native nodes write mp4 fine.

**Decision — stock models, not uncensored:** generate with stock Wan2.2 weights + the trained LoRA. The LoRA carries the target content, so the Wan2.2-Remix NSFW encoder from [[uncensored-wan22-models]] is **not used** here. Standard `umt5_xxl_fp16.safetensors` is the text encoder.

## Clip length: why 2 seconds, not 5

The generated clips are **33 frames @ 16 fps ≈ 2.06 s**, which surprises people expecting Wan2.2's demo-typical 5 s.

- **This is by design for 24 GB.** The plan's hardware profiles set **33 frames** for the 24 GB tier and reserve **57–81 frames (≈5 s)** for the 80 GB tier. See [[resolution-guide]] — frame count drives VRAM as hard as resolution.
- **The LoRA was trained at 33 frames** (`target_frames=[33]`), so generating at 33 frames matches the training distribution.
- **5-second clips DO work on 24 GB (verified 2026-06-06):** set `EmptyHunyuanLatentVideo.length = 81` (81/16 ≈ 5.06 s). An 81-frame clip generated successfully under the same `--lowvram` setup — no OOM, output h264 512×768 81f. It is slower than 33 frames (more latent/activation memory) but fits. Caveat: the LoRA was trained on 33-frame clips, so judge whether longer fight motion stays coherent or drifts/loops before standardizing on 81 frames.

## See also
- [[hardware-requirements]] · [[multi-gpu-training]] · [[resolution-guide]] · [[dual-noise-architecture]] · [[uncensored-wan22-models]] · [[comfyui]] · [[musubi-tuner]]
