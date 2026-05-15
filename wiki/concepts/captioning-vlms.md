---
title: VLM Models for Auto-Captioning
type: concept
tags: [captioning, vlm, ollama, qwen, abliterated, uncensored]
created: 2026-05-15
updated: 2026-05-15
---

# VLM Models for Auto-Captioning

When training violence-detection or other sensitive-content LoRAs, mainstream VLMs refuse to describe the footage. This page catalogs vision-language models tested for captioning, with refusal rates and quality data.

## The Problem

Standard VLMs (Qwen2.5-VL, GPT-4V, etc.) are safety-tuned to refuse describing violence, weapons, or assault. For a violence-detection LoRA dataset, this means 70-80% of clips get fallback captions instead of real descriptions. Abliterated models solve this by surgically removing the refusal mechanism.

## Tested Models

### Qwen2.5-VL 7B (standard)

| Metric | Value |
|--------|-------|
| Ollama tag | `qwen2.5vl:7b` |
| Size | 6.0GB (Q4_K_M) |
| Caption quality | Excellent — 80 words avg, temporal descriptions |
| Refusal rate on violence | ~74% (tested on 31 fighting clips) |
| Response time | ~133s per 5-frame caption |
| Trigger compliance | 100% |

**Verdict:** Best caption quality, but unusable for sensitive content without abliteration.

### Qwen2.5-VL 7B Abliterated (huihui-ai) — RECOMMENDED

| Metric | Value |
|--------|-------|
| Ollama tag | `huihui_ai/qwen2.5-vl-abliterated:7b` |
| HuggingFace | [huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated) |
| Size | 6.0GB (Q4_K_M) |
| Caption quality | Same as standard Qwen (identical architecture, only refusal direction removed) |
| Refusal rate | ~0% |
| Method | Abliteration on text layers only; vision encoder untouched |

Drop-in replacement. Same Ollama API, same prompt format, same quality. Only the model name changes.

Also available in larger sizes:
- `huihui_ai/qwen2.5-vl-abliterated:32b` (21GB) — better quality, fits in 24GB VRAM
- `huihui_ai/qwen2.5-vl-abliterated:3b` (3.2GB) — edge/mobile

### Qwen2.5-VL 7B Abliterated Caption-it (prithivMLmods)

| Metric | Value |
|--------|-------|
| HuggingFace | [prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it](https://huggingface.co/prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it) |
| GGUF | [prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF) |
| Size | ~6GB |
| Specialty | Fine-tuned specifically for uncensored image captioning |

Fine-tuned on top of abliteration for detailed captioning. May produce more descriptive captions than the base abliterated model. Untested in our pipeline.

### Gemma 4 E4B OBLITERATED — NOT RECOMMENDED

| Metric | Value |
|--------|-------|
| HuggingFace | [OBLITERATUS/gemma-4-E4B-it-OBLITERATED](https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED) |
| Size | ~5GB (Q4_K_M) / ~17GB (bf16) |
| Caption quality | Very poor — 21 words avg, hallucinated URLs, no temporal descriptions |
| Refusal rate | 0% |
| Response time | ~12s (transformers) |

Despite 0% refusal rate, the aggressive abliteration on 4B params destroyed instruction-following. Produces hallucinated timestamps, URLs, and nonsensical output regardless of prompting. Not viable for captioning.

### Llama 3.2 11B Vision Abliterated

| Metric | Value |
|--------|-------|
| HuggingFace | [huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated](https://huggingface.co/huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated) |
| Size | ~7GB quantized |
| Caption quality | Untested |
| Refusal rate | ~0% (claimed) |

Alternative architecture. 11B params should produce good captions. Untested in our pipeline.

## What is Abliteration?

Abliteration removes a model's refusal mechanism without retraining. It works by:
1. Identifying the "refusal direction" in the model's residual stream
2. Projecting it out so the model can't represent refusal
3. Only modifying text layers — vision encoder stays intact

Key tool: [remove-refusals-with-transformers](https://huggingface.co/blog/mlabonne/abliteration) by mlabonne.

Community naming conventions for abliterated models: `*-abliterated`, `*-heretic`, `*-uncensored`.

As of 2026, the [Heretic AI](https://aithinkerlab.com/heretic-ai-abliteration-benchmarks-2026/) tool automates this with Optuna optimization, achieving <3% refusal rate with minimal capability damage.

## Recommendation for Violence-Detection Pipeline

Use `huihui_ai/qwen2.5-vl-abliterated:7b` as the default captioning model. It's:
- Identical quality to the standard model (same weights except refusal direction)
- Available directly on Ollama (one-line pull)
- Drop-in replacement (same API, same prompt format)
- 0% refusal on violent content

```python
# In scripts/caption.py
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen2.5-vl-abliterated:7b")
```

## Related

- [[captioning]] — Caption structure, trigger tokens, prompt guidelines
- [[dataset-preparation]] — Overall dataset setup
