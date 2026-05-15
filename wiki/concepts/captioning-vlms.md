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

### Qwen2.5-VL 7B Abliterated (huihui-ai)

| Metric | Value |
|--------|-------|
| Ollama tag | `huihui_ai/qwen2.5-vl-abliterated:7b` |
| HuggingFace | [huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated) |
| Size | 6.0GB (Q4_K_M) |
| Refusal rate on violence | ~16% (5/31 fallbacks on fighting clips) |
| Caption quality | Good but hallucination-prone — sometimes describes mundane office activity on fighting clips |
| Method | Abliteration on text layers only; vision encoder untouched |

Massive improvement in refusal rate (84% success vs 26% standard). However, 7B is too small to reliably interpret low-quality surveillance footage — produces hallucinated descriptions on some clips.

Also available in larger sizes:
- `huihui_ai/qwen2.5-vl-abliterated:32b` (21GB) — "virtually eliminates hallucinations," fits in 24GB VRAM
- `huihui_ai/qwen2.5-vl-abliterated:3b` (3.2GB) — edge/mobile

### Qwen3-VL 8B Abliterated (huihui-ai) — RECOMMENDED

| Metric | Value |
|--------|-------|
| Ollama tag | `huihui_ai/qwen3-vl-abliterated:8b` |
| HuggingFace | [huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated) |
| Size | 6.1GB (Q4_K_M) |
| Caption quality | Untested — expected better than Qwen2.5-VL 7B due to architecture improvements |
| Refusal rate | ~0% (claimed) |
| Method | Abliteration on text layers only |

Qwen3-VL is a generation ahead of Qwen2.5-VL with key upgrades:
- **DeepStack integration** — multi-level ViT features for tighter vision-language alignment
- **Enhanced interleaved-MRoPE** — stronger spatial-temporal modeling for video frames
- **Text-based time alignment** — explicit timestamp alignment for temporal grounding
- **256K context** natively, expandable to 1M tokens

Same VRAM footprint as the 2.5-VL 7B. Also available as thinking variant (`Huihui-Qwen3-VL-8B-Thinking-abliterated`) for chain-of-thought reasoning.

Other Qwen3-VL abliterated sizes:
- `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` — smaller, HF only
- `huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated` — MoE (30B total, 3B active), HF only

### Qwen2.5-VL 32B Abliterated — BEST QUALITY (if VRAM allows)

| Metric | Value |
|--------|-------|
| Ollama tag | `huihui_ai/qwen2.5-vl-abliterated:32b` |
| Size | 21GB |
| Caption quality | "Virtually eliminates hallucinations, catches fine details" — community-validated |
| Refusal rate | ~0% |
| VRAM | Fits in 24GB |

The jump from 7B → 32B is substantial for captioning. Community testing confirms: "30B at 16-bit is basically spot on all the time. No hallucinations, great detection of even minor details." If VRAM allows, this is the most reliable option.

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

## Qwen Model Generation Timeline

| Model | Release | Key VLM Sizes |
|-------|---------|---------------|
| Qwen2.5-VL | Mid 2025 | 3B, 7B, 32B, 72B |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Oct 2025 | 2B, 4B, 8B, 32B, 30B-A3B MoE, 235B-A22B MoE |
| [Qwen3.5](https://github.com/QwenLM/Qwen3.6) | Feb 2026 | Native multimodal, up to 397B-A17B |
| [Qwen3.6](https://github.com/QwenLM/Qwen3.6) | Apr 2026 | 27B, 35B-A3B — unified vision-language with early fusion |

All Qwen3+ VL models have abliterated variants by huihui-ai within days of release.

## Recommendation for Violence-Detection Pipeline

**Primary:** `huihui_ai/qwen3-vl-abliterated:8b` — best quality-per-VRAM with latest architecture improvements and 0% refusals. Same VRAM as Qwen2.5-VL 7B.

**If hallucination issues persist:** Upgrade to `huihui_ai/qwen2.5-vl-abliterated:32b` (21GB, fits in 24GB) for near-zero hallucinations.

**Fallback:** `huihui_ai/qwen2.5-vl-abliterated:7b` — proven working, 84% success rate on fighting clips.

```python
# In scripts/caption.py
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-vl-abliterated:8b")
```

## Related

- [[captioning]] — Caption structure, trigger tokens, prompt guidelines
- [[dataset-preparation]] — Overall dataset setup
