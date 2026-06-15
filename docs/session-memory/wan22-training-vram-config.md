---
name: wan22-training-vram-config
description: VRAM settings required to train Wan2.2 14B dual-noise LoRAs on the 2x RTX A5500 (24GB) rig without OOM
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

Training Wan2.2 14B dual-noise LoRAs (high-noise→GPU 0, low-noise→GPU 1, in parallel) on the wan2.2-lora project's rig: **2× RTX A5500, 24 GB each**. GPU 0 also carries ~1.2 GB of desktop usage (Xorg, gnome-shell, Firefox) at baseline, so its effective margin is smaller than GPU 1's.

**Why:** The original `train_all.sh` config (`blocks_to_swap 20`, `fp8_base`, fp16, 512×768×33, rank 32) OOMs at the *first training step* — each expert peaks just ~270 MiB over the 24 GB ceiling. VRAM peaks ~21 GB during steps, not the ~14 GB seen between steps.

**How to apply:** Use `scripts/train_category.sh CATEGORY [STEPS]` (single-category launcher; skips re-caching if cache exists). It sets the two fixes that resolved the OOM with zero quality impact (resolution/frames/rank/steps unchanged):
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reclaims ~1.6 GB reserved-but-unallocated fragmentation
- `--blocks_to_swap 30` (was 20) — offloads more of the 14B model to CPU RAM

Trade-off: heavy block-swap → ~29 s/step → ~24 h for 3000 steps (both experts parallel). Lowering blocks_to_swap below ~28 risks OOM again given the ~21 GB peaks. `train_all.sh` still has the old swap=20 and will OOM — prefer `train_category.sh`, or backport these two changes before a full multi-category run. See [[wan22-project-status]].
