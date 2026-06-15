---
name: comfyui-generation-setup
description: How ComfyUI is installed and wired for generating Wan2.2 synthetic clips in the wan2.2-lora project
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

ComfyUI for the wan2.2-lora project's synthetic-clip generation (set up 2026-06-04).

**Location:** `/home/lenovo5/TiongKai/Greenfield/ComfyUI` (sibling to wan2.2-lora and musubi-tuner), own venv at `ComfyUI/.venv` with torch 2.6.0+cu124 (matches training rig). Separate from the training venv.

**Model wiring:** `ComfyUI/extra_model_paths.yaml` points at the existing weights in `wan2.2-lora/models/...split_files/` and LoRAs in `wan2.2-lora/loras/` — nothing copied. Uses STOCK Wan2.2 + LoRA, no Remix encoder (see [[wan22-generation-stock-models]]).

**Workflow:** `wan2.2-lora/scripts/comfyui_wan22_t2v_workflow.json` (API format) — dual-expert (high+low noise) Wan2.2 T2V, dual LoRA (LoraLoaderModelOnly per expert @ strength 0.75), CLIPLoader type=wan with umt5_xxl, two KSamplerAdvanced relay (high steps 0-10 add_noise=enable/leftover, low steps 10-20), ModelSamplingSD3 shift=8.0, 512x768x33, EmptyHunyuanLatentVideo, then native CreateVideo(16fps)->SaveVideo(mp4/h264). VideoHelperSuite was NOT used (missing cv2/imageio-ffmpeg); native nodes write mp4 fine. Validated against /object_info.

**VRAM (critical):** a 14B T2V + LoRA does NOT fit on one 24GB card by default — it OOMs during LoRA patching ("Allocation on device"). Three fixes, all now baked in: (1) `run_comfyui.sh` launches with `--lowvram --disable-cuda-malloc`, (2) the workflow runs the umt5 text encoder on CPU (`CLIPLoader device=cpu`, frees ~11GB). With these, one generation fits and takes ~350s/clip (slow but stable). If it still OOMs, escalate to `--novram`.

**To run generation:**
1. `bash wan2.2-lora/scripts/run_comfyui.sh` — starts ComfyUI on GPU 1 (desktop-free) with the lowvram flags, output dir = `generated/clips/`, port 8188. Kill old instances by port first: `for p in $(fuser 8188/tcp); do kill -9 $p; done` (the process cmdline is `main.py`, so `pkill -f ComfyUI/main.py` does NOT match it).
2. `wan2.2-lora/.venv/bin/python wan2.2-lora/scripts/generate.py --category fighting --count N` — batch client; patches the template (node IDs in generate.py PATCH_FIELDS match the JSON), saves to `generated/clips/fighting/`, writes manifest CSV. generate.py's wait_for_completion checks real run status (ok/error/timeout), not just history presence.

**First verified clip:** 2026-06-06 — h264 512x768 33f @16fps, ~2s, from the step-3000 LoRA @ strength 0.75, 20 steps (10 high-noise + 10 low-noise), cfg 5, shift 8.

Trigger word is `fght99`; prompt library at `scripts/prompts/<category>.txt`. Generation needs the GPU free — can't run while training occupies both cards. See [[wan22-training-vram-config]].
