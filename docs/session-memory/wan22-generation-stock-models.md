---
name: wan22-generation-stock-models
description: "Decision to generate violence clips with stock Wan2.2 models + trained LoRA, NOT the uncensored Remix NSFW encoder"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

For the wan2.2-lora project's synthetic-clip generation, the user wants to use **stock/normal Wan2.2 base models + the trained category LoRA** to produce the desired (violence) content. They explicitly do **not** want to use the uncensored Wan2.2 models (Wan2.2-Remix NSFW text encoder `nsfw_wan_umt5-xxl_bf16.safetensors`).

**Why:** The LoRA is trained specifically to generate the target content, so the base model + LoRA is expected to suffice — no uncensored text encoder needed.

**How to apply:** Use the standard `models/wan2.2-t2v/split_files/text_encoders/umt5_xxl_fp16.safetensors` text encoder in the ComfyUI generation workflow. Do NOT download or wire the Remix NSFW encoder. This overrides the recommendation in `plan.md` (Experiment 5) and `wiki/synthesis/uncensored-wan22-models.md` — those suggest the Remix encoder, but the user has decided against it. See [[wan22-training-vram-config]].
