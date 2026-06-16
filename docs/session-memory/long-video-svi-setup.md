---
name: long-video-svi-setup
description: "SVI 2.0 Pro long-video setup in ComfyUI (WanVideoWrapper, LoRAs, the fp16 gotcha)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

Long-form (15s+) video for the wan2.2-lora project uses **SVI 2.0 Pro** (Stable Video Infinity) on **base Wan2.2 I2V** — no fighting LoRA (see [[base-model-over-lora]]). Set up 2026-06-16.

**Installed:** Kijai `ComfyUI-WanVideoWrapper` at `ComfyUI/custom_nodes/` (deps via `uv pip install` — the ComfyUI venv has no `pip`; uv-created). Key nodes: `WanVideoSVIProEmbeds` (anchor_samples=start latent, prev_samples=continuation, motion_latent_count=1 for Wan2.2), `WanVideoModelLoader`, dual `WanVideoSampler`, `WanVideoLoraSelect` (chain via prev_lora), `WanVideoBlockSwap`, `WanVideoContextOptions`.

**LoRAs** (in `loras/svi/`, from `Kijai/WanVideo_comfy`): `SVI_v2_PRO_Wan2.2-I2V-A14B_{HIGH,LOW}_lora_rank_128_fp16.safetensors` (1.2GB each) + `Wan_2_2_I2V_A14B_{HIGH,LOW}_lightx2v_4step_lora_260412_rank_64_fp16.safetensors` (602MB each). Per-expert scales: HIGH = SVI 1.0 + LightX2V ~0.5-0.6; LOW = SVI 1.0 + LightX2V 1.0.

**Workflow:** `scripts/comfyui_svi_i2v_workflow.json` (API format, validated vs /object_info). Dual experts, fp8_e4m3fn quant, blocks_to_swap 30, LightX2V 4-step distill → only ~6 sampler steps total (HIGH 0-3, LOW 3-end), cfg 1, shift 8, 480x832, native CreateVideo→SaveVideo (svi/svi_test). The SVI resize node is `WanVideoImageResizeToClosest` (crop_to_new) — `ImageResizeKJv2` is NOT installed.

**GOTCHA:** `WanVideoModelLoader.base_precision` must be **`fp16`**, NOT `fp16_fast` — fp16_fast needs torch 2.7+ nightly (rig has torch 2.6.0); it errors "torch.backends.cuda.matmul.allow_fp16_accumulation is not available".

**Long-form:** the workflow is one 81-frame window. To extend: chain SVIProEmbeds windows (each window's prev_samples = previous LOW-sampler samples) or attach WanVideoContextOptions (context_frames=81, overlap=16) for sliding-window context in one larger num_frames. See [[long-video-generation-wan22]] wiki page and [[comfyui-generation-setup]].
