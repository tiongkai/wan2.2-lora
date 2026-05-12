---
title: Captioning Best Practices
type: concept
tags: [captioning, trigger-tokens, dataset, text-labels]
created: 2026-05-12
updated: 2026-05-12
---

# Captioning Best Practices

Every training image or video clip needs a matching `.txt` caption file. Caption quality directly impacts LoRA quality — vague or poetic captions produce vague LoRAs.

## Core Principles

1. **Descriptive, not poetic** — factual descriptions outperform creative writing
2. **Include a trigger token** — a unique identifier the model associates with your concept
3. **Describe what you see** — for video, include motion and temporal progression
4. **Be consistent** — use the same structure across all captions

## Trigger Tokens

A trigger token is a unique, non-English word placed at the start of every caption during training. At inference time, including the trigger in your prompt activates the LoRA's learned concept.

**Good trigger tokens**: `s33s`, `zxq-person`, `tok_character`, `sks_style`

**Bad trigger tokens**: real English words (will conflict with the model's existing knowledge)

## Caption Structure

```
[trigger token], [subject description], [scene/pose/action description]
```

### Examples

**Good** (identity LoRA):
```
s33s, a woman with short brown hair and glasses, sitting at a desk in warm office lighting, medium close-up shot
```

**Good** (style LoRA):
```
sks_style, a cityscape at sunset, bold saturated colors with visible brushstrokes, wide establishing shot
```

**Good** (motion/video LoRA):
```
orb_motion, the camera slowly orbits around a ceramic vase on a wooden table, revealing the room behind it, smooth continuous motion
```

**Bad**:
```
A moody candid portrait near a rainy window with ethereal light dancing across her face
```
(Too poetic, no trigger token, not descriptive of actual visual content)

## Caption Length

| Format | Recommended Length |
|--------|--------------------|
| Images | 20-40 words |
| Video clips | 50+ words |

Longer captions for video clips help the model understand temporal progression and motion context. Some community pipelines report good results with ≥50-word captions for video.

## Caption Dropout

Setting a caption dropout rate (~5% / `0.05`) during training randomly drops captions for some training steps. This helps the model learn to generate the concept even without the exact trigger prompt — improving generalization. Both [[ai-toolkit]] and [[musubi-tuner]] support this.

## Auto-Captioning Tools

For large datasets, manual captioning is impractical. Common auto-captioning approaches:

- Use a VLM (vision-language model) to generate initial captions, then manually review and add trigger tokens
- Florence-2, LLaVA, or similar models for batch captioning
- Some training tools have built-in captioning integration

**Always review auto-generated captions** — VLMs can hallucinate details or miss important visual elements.

## Common Mistakes

- Forgetting to add the trigger token to captions
- Using a real English word as trigger (conflicts with base model knowledge)
- Writing captions that describe mood/feeling rather than visual content
- Inconsistent caption structure across the dataset
- Captions that are too short for video clips (missing motion description)
- Not reviewing auto-generated captions for accuracy

## Related

- [[dataset-preparation]] — Overall dataset setup
- [[training-parameters]] — Where caption_dropout_rate is configured
- [[overfitting-and-troubleshooting]] — Caption quality issues causing training problems
