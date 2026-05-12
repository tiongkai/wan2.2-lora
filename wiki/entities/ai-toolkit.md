---
title: AI Toolkit (Ostris)
type: entity
tags: [tool, training, ai-toolkit, ostris]
created: 2026-05-12
updated: 2026-05-12
---

# AI Toolkit

AI Toolkit is a training toolkit by **Ostris** for fine-tuning diffusion models. It provides the simplest workflow for Wan 2.2 LoRA training, especially for [[dual-noise-architecture]] handling.

## Key Info

- **Repository**: https://github.com/ostris/ai-toolkit
- **Author**: Ostris (ostris.com)
- **Config format**: YAML
- **Tagline**: "The ultimate training toolkit for finetuning diffusion models"

## Supported Models

- Wan 2.1 (T2V, I2V, 1.3B, 14B)
- Wan 2.2 (T2V, I2V, 5B, 14B)
- FLUX, Stable Diffusion, and others

## Key Advantage: Unified Dual-Noise Training

Unlike [[musubi-tuner]], AI Toolkit handles both Wan 2.2 experts in a single training run:

```yaml
model_kwargs:
  train_high_noise: true
  train_low_noise: true
```

This produces a single LoRA file that works for both experts — no separate training runs needed.

## Installation

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt
```

## Example Config (Wan 2.2 I2V 14B)

```yaml
config:
  name: wan22-i2v-lora
  process:
    - type: sd_trainer
      training_folder: output
      device: cuda:0
      network:
        type: lora
        linear: 64
        linear_alpha: 64
      save:
        dtype: float16
        save_every: 500
      datasets:
        - folder_path: /path/to/dataset
          num_frames: 81
          resolution: [512, 768]
          caption_dropout_rate: 0.05
      train:
        batch_size: 1
        steps: 3000
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer:
          type: adamw
          lr: 0.0002
        dtype: bf16
      model:
        name_or_path: Wan2.2-I2V-A14B-Diffusers-bf16
        arch: wan22_14b_i2v
        quantize: true
        qtype: qfloat8
      model_kwargs:
        train_high_noise: true
        train_low_noise: true
      sample:
        sampler: flowmatch
        num_frames: 81
        fps: 16
```

## Architecture Flags

| `arch` value | Model |
|-------------|-------|
| `wan22_14b_t2v` | Wan 2.2 T2V 14B |
| `wan22_14b_i2v` | Wan 2.2 I2V 14B |
| `wan22_5b_i2v` | Wan 2.2 5B I2V |

## Quantization

AI Toolkit supports quantization to reduce VRAM:
- **FP8**: `quantize: true, qtype: qfloat8`
- **4-bit ARA**: even lower VRAM, community-reported I2V on 16GB

## Dataset Requirements

- Uniform format: entirely images OR entirely videos
- PNG for images, MP4 for videos
- TXT captions with matching filenames
- Identical aspect ratios across media files
- `caption_dropout_rate: 0.05` recommended for generalization

## YouTube Tutorials by Ostris

- [Train Wan 2.2 I2V 14B LoRA](https://www.youtube.com/watch?v=2d6A_l8c_x8) — orbital shot demo
- [Train Wan 2.2 5B I2V LoRA](https://www.youtube.com/watch?v=9ATaQdin1sA)
- [Train Wan 2.1 I2V LoRA](https://www.youtube.com/watch?v=_swwOO95FrY) — dolly zoom demo
- [Wan 2.1 Character LoRA (images only)](https://www.youtube.com/watch?v=oJdT5dzrNEY)

## Web UI

Available via **RunComfy** integration for cloud-based training with H100/H200 GPUs.
- T2V guide: https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-t2v-14b-lora-training
- I2V guide: https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-14b-lora-training

## Strengths

- Simplest Wan 2.2 workflow (single run for both experts)
- YAML config (more readable than CLI args)
- Built-in quantization
- Good YouTube tutorial coverage
- Web UI option via RunComfy

## Limitations

- Fewer community-tested configurations than [[musubi-tuner]]
- Less granular control over timestep ranges
- Newer project, smaller community

## Related

- [[musubi-tuner]] — Alternative with more community support
- [[diffusion-pipe]] — Alternative using DeepSpeed
- [[training-parameters]] — Full parameter reference
- [[training-tools-comparison]] — Side-by-side comparison
