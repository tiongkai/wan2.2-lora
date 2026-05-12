---
title: ComfyUI
type: entity
tags: [tool, inference, comfyui, ui]
created: 2026-05-12
updated: 2026-05-12
---

# ComfyUI

ComfyUI is a node-based visual UI for running diffusion model inference. It is **not** a training tool — it's used to generate videos with trained LoRAs.

## Role in the Workflow

```
Train LoRA → Load in ComfyUI → Generate video with LoRA applied
```

## Loading Wan LoRAs

LoRAs trained with any tool ([[musubi-tuner]], [[ai-toolkit]], [[diffusion-pipe]]) can be loaded in ComfyUI using LoRA loader nodes.

Alternatively, LoRAs can be loaded via the `diffusers` Python library:

```python
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B")
pipe.load_lora_weights("path/to/lora.safetensors")
```

## Key Features for Wan

- Supports Wan 2.1 and 2.2 T2V/I2V pipelines
- CausVid LoRA for extreme speed generation
- Self-Forcing workflow for smooth video
- Node-based interface allows complex workflows

## Related Tutorials

- [WAN 2.1 Self Forcing w/ LoRA](https://www.youtube.com/watch?v=TyiEHj8TtTE) — Low VRAM video generation
- [Wan 2.1 T2V & I2V with CausVid LoRA](https://www.youtube.com/watch?v=XNcn845UXdw)

## Related

- [[wan-2-2]] — The model
- [[ai-toolkit]] — Training tool
- [[musubi-tuner]] — Training tool
