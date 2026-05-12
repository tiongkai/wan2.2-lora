---
title: "Apatero Blog: WAN 2.2 Person LoRA Pro Method"
type: source
tags: [apatero, person-lora, identity, captioning]
source_url: https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025
source_date: 2025-12
created: 2026-05-12
updated: 2026-05-12
---

# Apatero Blog — WAN 2.2 Person LoRA Pro Method

## Key Takeaways

- A6000 (96GB VRAM): ~24 hours for a solid person LoRA
- Consumer hardware (24GB): 2-3 days but **absolutely works**
- 10-30 high-quality images or short clips recommended
- Defines a clear caption structure: `[trigger token], [description of person], [description of scene/pose]`

## Caption Structure

Every image/clip captioned as:
```
[trigger token], [description of person], [description of scene/pose]
```

This three-part structure ensures the model learns identity separately from context.

## Hardware Reality Check

The guide is honest about consumer hardware — it works but takes 2-3 days. This is important for setting realistic expectations for users without datacenter GPUs.

## Cross-References

- [[captioning]] — Caption structure adopted from this source
- [[hardware-requirements]] — Training time estimates
- [[training-parameters]] — Person LoRA specific settings
