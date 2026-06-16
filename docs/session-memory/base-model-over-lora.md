---
name: base-model-over-lora
description: Decision to focus on base Wan2.2 + prompts and shelve the fighting LoRA (data-quality ceiling)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

As of 2026-06-16 the user decided to **focus on the base Wan2.2 model + good prompts** and **shelve the trained fighting LoRA**.

**Why:** the fighting LoRA produces visible fighting motion but with a lot of blur. Root cause is the **low-quality surveillance training footage** — a LoRA can't render cleaner than its training data, so it pulls outputs toward that grainy/soft look, making them look worse than the crisp base model. Confirmed not a usage bug: a fair re-test at the LoRA's trained 512×768 dims still showed the blur (the earlier 720×512 test was also unfair — trained portrait, generated landscape). Captions were fine and loss converged, so the training run wasn't broken — it's a data ceiling.

**How to apply:** Generate with **stock Wan2.2 + scene-grounded, motion-descriptive prompts** (Qwen-caption the start image first — see workflow in [[comfyui-generation-setup]]). For long-form, use **SVI** (base Wan2.2 + the SVI continuity LoRA, NOT the fighting LoRA). If a category LoRA is ever revisited, retrain on **higher-quality footage** first. See [[wan22-generation-stock-models]] [[i2v-lora-training]] [[long-video-svi-setup]].
