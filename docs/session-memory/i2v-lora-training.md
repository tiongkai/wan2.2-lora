---
name: i2v-lora-training
description: "How to train a Wan2.2 I2V LoRA in the wan2.2-lora project (models, script, the --i2v gotcha)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

Training a dedicated **I2V** fighting LoRA on Wan2.2 I2V-A14B (started 2026-06-06). Chosen over reusing the T2V LoRA because the I2V model has `in_dim=36` (extra conditioning-image channels) vs T2V's 16 — a T2V LoRA won't transfer cleanly.

**Models:** `models/wan2.2-i2v/split_files/diffusion_models/wan2.2_i2v_{high,low}_noise_14B_fp16.safetensors` (27 GB each, from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`). **Wan2.2 I2V needs NO CLIP-vision model** (unlike Wan2.1) — reuses the same T5 + VAE as T2V. (Download whole-repo by accident pulls 100GB+; always pass specific file paths to `hf download`.)

**Script:** `scripts/train_i2v_category.sh CATEGORY [STEPS]` + `configs/fighting_i2v_dataset.toml` (same 31 clips, separate `cache_i2v/` dir — I2V latents store the first frame so they're not interchangeable with the t2v cache). Same VRAM-safe flags as [[wan22-training-vram-config]] (`blocks_to_swap 30` + `expandable_segments`); ~28 s/step, ~23 h for 3000 steps; outputs → `loras/fighting_i2v/fighting_i2v_lora_r32_{high,low}.safetensors`.

**GOTCHA:** the `--i2v` flag goes ONLY on `wan_cache_latents.py` (caching). The training script `wan_train_network.py` REJECTS `--i2v` (exit 2, "unrecognized arguments") — it infers I2V mode from `--task i2v-A14B`. Caching needs both `--i2v` and `--task` is implicit via config.

**I2V generation (built 2026-06-06, validated against /object_info):** `scripts/comfyui_wan22_i2v_workflow.json` (WanImageToVideo node; clip_vision omitted — Wan2.2 doesn't need it; samplers wired to WanImageToVideo's conditioning+latent outputs, shift 5.0, cfg 3.5) + `scripts/generate_i2v.py` (uploads a start frame via /upload/image, `--image`, `--frames`, `--strength`). I2V experts exposed to ComfyUI via a second block in `ComfyUI/extra_model_paths.yaml`. Just needs the LoRA finals + a start image at runtime.

**Disk watch:** prune old checkpoints before training more categories (was tight at 98%, reclaimed to 93%/127 GB free after deleting gemma VLM + old t2v checkpoints). See [[comfyui-generation-setup]].
