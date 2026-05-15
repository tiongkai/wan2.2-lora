---
title: "Uncensored Wan 2.2 Models & LoRAs"
type: synthesis
tags: [wan-2-2, uncensored, nsfw, remix, lora, generation, safety-filter]
created: 2026-05-15
updated: 2026-05-15
---

# Uncensored Wan 2.2 Models & LoRAs

Catalog of community-created uncensored Wan 2.2 model variants and LoRAs relevant to generating sensitive content (violence, fighting) for security detection datasets.

## Key Finding: Wan 2.2's Safety Filtering

The standard [[wan-2-2]] model has **minimal built-in safety filtering** in the diffusion model itself. Content restrictions are primarily in the **T5 text encoder** (`umt5-xxl`), which softens or ignores violent/adult prompts. The community has solved this by replacing the text encoder.

## Wan2.2-Remix NSFW — Uncensored Base Model

**Not a LoRA** — a full replacement for the base model with an unrestricted text encoder.

| Detail | Value |
|--------|-------|
| HuggingFace | [FX-FeiHou/wan2.2-Remix](https://huggingface.co/FX-FeiHou/wan2.2-Remix), [garrychan/wan2.2-Remix-NSFW](https://huggingface.co/garrychan/wan2.2-Remix-NSFW) |
| Type | Full model replacement (high-noise + low-noise transformers + custom text encoder) |
| Variants | T2V, I2V, v2.0 and v3 (final) |
| Key innovation | Custom text encoder `nsfw_wan_umt5-xxl_bf16.safetensors` with content restrictions removed |
| ComfyUI | Standard workflows, drop-in replacement |
| VRAM | Same as base Wan 2.2 (~24GB with FP8) |

### T2V Files Needed

```
Wan2.2_Remix_NSFW_t2v_14b_high_lighting_v2.0.safetensors
Wan2.2_Remix_NSFW_t2v_14b_low_lighting_v2.0.safetensors
nsfw_wan_umt5-xxl_bf16.safetensors   # custom text encoder — the key piece
```

### I2V Files Needed

```
Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors
Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors
nsfw_wan_umt5-xxl_bf16.safetensors
```

### How It Works

The standard Wan 2.2 T5 text encoder has been trained to suppress or soften certain prompts. The Remix NSFW variant replaces this with an unrestricted text encoder that faithfully encodes all prompts, including violent and adult content. The diffusion models themselves are enhanced with motion LoRA blending for improved realism.

### Implication for Violence-Detection Pipeline

When generating synthetic clips in Task 7, prompts like `"fght99, two men fighting in a parking lot"` may be softened by the standard text encoder, producing mild or ambiguous output. The Remix NSFW text encoder would pass the full prompt through faithfully, producing more accurate violence depictions. Custom LoRAs trained on the standard model stack on top of the Remix base.

## Existing Fighting LoRA: LuisaP VIDEOGAME FIGHTING

| Detail | Value |
|--------|-------|
| CivitAI | [LuisaP WAN 2.2 VIDEOGAME FIGHTING](https://civitai.com/models/2093652/luisap-wan-22-videogame-fighting) |
| Type | I2V LoRA, **high-noise only** |
| Trigger | `FIGHTSCENE` |
| Style | Videogame-style fighting (not surveillance realism) |
| Prompt example | "FIGHTSCENE, two character fights each other, high action, violence, both characters punch on the face, on belly, on legs" |

**Validates the approach** — someone has successfully trained a fighting LoRA on Wan 2.2. However, it targets a videogame aesthetic, not surveillance-camera footage. Our pipeline targets the latter.

## Phr00t AllInOne — Merged Checkpoint

| Detail | Value |
|--------|-------|
| HuggingFace | [Phr00t/WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) |
| Type | All-in-one merged checkpoint (multiple LoRAs baked in) |
| Versions | Up to Mega v12 (latest uses bf16 Fun VACE base) |
| Speed | 4-step generation with CFG=1 |
| Focus | General NSFW adult content, not violence specifically |

Uses `wan2.2-i2v-rapid-aio-v10-nsfw.safetensors` and similar files. "Jack of all trades, master of none" approach to NSFW generation.

## General NSFW I2V LoRA

| Detail | Value |
|--------|-------|
| HuggingFace | [lopi999/Wan2.2-I2V_General-NSFW-LoRA](https://huggingface.co/lopi999/Wan2.2-I2V_General-NSFW-LoRA) |
| Type | I2V LoRA |
| Focus | General uncensored I2V content |

## Recommendation for Our Pipeline

| Stage | Standard Model | With Remix NSFW |
|-------|---------------|-----------------|
| **Training** (musubi-tuner) | Use standard Wan 2.2 weights — LoRA learns the concept regardless of text encoder | Same — LoRA training doesn't depend on text encoder freedom |
| **Generation** (ComfyUI) | Prompts may be softened by standard T5 | Use Remix NSFW text encoder for faithful prompt encoding |
| **Inference quality** | Violence may look mild/ambiguous | Violence rendered as prompted |

**Recommended approach:** Train LoRAs on standard Wan 2.2 weights (proven, documented), but generate synthetic clips using the Remix NSFW text encoder for accurate violence depiction.

## Related

- [[wan-2-2]] — Base model details
- [[captioning-vlms]] — VLM safety filters for captioning (separate problem from generation)
- [[comfyui]] — Inference/generation setup
