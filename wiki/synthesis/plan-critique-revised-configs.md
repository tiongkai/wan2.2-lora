---
title: "Revised Training Configs — Violence Detection LoRA Plan"
type: synthesis
tags: [plan, critique, revised-configs, violence-detection, training]
created: 2026-05-12
updated: 2026-05-12
---

# Revised Training Configs for Violence Detection LoRA Plan

This page contains corrected AI Toolkit training configs addressing issues found in the original plan at `/Users/htx/Desktop/Projects/wan2.2-lora-1try/plan.md`.

## Changes Summary

| Issue | Original | Revised | Reason |
|-------|----------|---------|--------|
| Dual-noise flags | Missing | `train_high_noise: true, train_low_noise: true` | **Required** for Wan 2.2 — without this, LoRA is broken |
| Architecture | `arch: wan, wan_type: t2v` | `arch: wan22_14b_t2v` | Correct AI Toolkit arch identifier |
| Resolution | `[720, 1280]` | `[512, 768]` (24GB) | 1280×720 OOMs on 24GB with video frames |
| num_frames | 16 | 33 | 16 frames = <1s of video, insufficient for motion |
| Rank | 16 | 32 | Motion LoRAs need higher capacity than identity LoRAs |
| Learning rate | 5e-5 | 1e-4 | 5e-5 is for identity; motion needs higher LR |
| Caption dropout | Missing | 0.05 | Improves generalization |
| Quantization | `quantize: true, low_vram: true` | `quantize: true, qtype: qfloat8` | Correct AI Toolkit syntax |

## A100 Scaling Notes

When moving to A100 (80GB), change these values:

```yaml
# A100 overrides:
resolution: [720, 1280]   # full 720p
num_frames: 57             # or 81 for longer motion learning
batch_size: 2              # double batch for faster convergence
# Remove quantize/qtype if you have headroom
```

## Revised Template Config (Fighting)

```yaml
# configs/fighting_lora.yaml
job: extension
config:
  name: fighting_lora_r32
  process:
    - type: sd_trainer
      training_folder: "../loras/fighting"
      device: cuda:0
      trigger_word: "fght99"
      network:
        type: lora
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 300
        max_step_saves_to_keep: 5
      datasets:
        - folder_path: "../datasets/processed/fighting"
          caption_ext: txt
          resolution: [512, 768]
          num_frames: 33
          fps: 24
          caption_dropout_rate: 0.05
      train:
        batch_size: 1
        steps: 3000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine
        lr_warmup_steps: 150
        max_grad_norm: 1.0
        noise_scheduler: flowmatch
        log_every: 100
        log_with: tensorboard
        log_dir: "../logs/fighting"
      model:
        name_or_path: "Wan-AI/Wan2.2-T2V-A14B"
        arch: wan22_14b_t2v
        quantize: true
        qtype: qfloat8
      model_kwargs:
        train_high_noise: true
        train_low_noise: true
      sample:
        sampler: flowmatch
        sample_every: 300
        width: 768
        height: 512
        num_frames: 33
        guidance_scale: 4.0
        sample_steps: 30
        seed: 42
        walk_seed: true
        neg: "blurry, low quality, watermark, distorted"
        prompts:
          - "fght99, two men fighting in a parking lot, security camera angle, night lighting"
          - "fght99, physical altercation between two people on a sidewalk, overcast daylight"
```

## Per-Category Differences

Only the fields that differ from the template:

### Vandalism

```yaml
name: vandalism_lora_r32
trigger_word: "vndl77"
training_folder: "../loras/vandalism"
folder_path: "../datasets/processed/vandalism"
steps: 2500        # vandalism has simpler motion patterns
log_dir: "../logs/vandalism"
prompts:
  - "vndl77, person spray painting graffiti on a wall, street level view, night"
  - "vndl77, person smashing a shop window with a bat, security footage angle"
```

### Stabbing

```yaml
name: stabbing_lora_r32
trigger_word: "stbb44"
training_folder: "../loras/stabbing"
folder_path: "../datasets/processed/stabbing"
steps: 3000        # complex motion — arm extension, weapon draw
log_dir: "../logs/stabbing"
prompts:
  - "stbb44, armed person confronting another in an alley, low light, security camera"
  - "stbb44, knife drawn in a close-range confrontation, overhead CCTV angle"
```

### Shooting

```yaml
name: shooting_lora_r32
trigger_word: "shtn22"
training_folder: "../loras/shooting"
folder_path: "../datasets/processed/shooting"
steps: 3000        # complex motion — draw, aim, discharge
log_dir: "../logs/shooting"
prompts:
  - "shtn22, person drawing a handgun in a parking garage, overhead security camera"
  - "shtn22, armed confrontation on a street corner, wide angle CCTV footage"
```

## Revised Captioning Approach

Replace single-keyframe captioning with multi-frame temporal captioning:

```python
def extract_multi_keyframes(clip: Path, n_frames: int = 5, out_dir: Path = None) -> list[Path]:
    """Extract frames at 10%, 25%, 50%, 75%, 90% of clip duration."""
    out_dir = out_dir or clip.parent
    duration = probe_duration(clip)
    positions = [duration * p for p in [0.10, 0.25, 0.50, 0.75, 0.90]]
    frames = []
    for i, pos in enumerate(positions):
        frame_path = out_dir / f"{clip.stem}_kf{i}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(pos), "-i", str(clip),
            "-frames:v", "1", "-q:v", "2", str(frame_path)
        ], check=True, capture_output=True)
        frames.append(frame_path)
    return frames
```

Then send all 5 frames to Qwen2.5-VL with a prompt like:

```
These 5 frames are sampled sequentially from a {duration}s security camera video.
Describe the complete action sequence: what happens from start to end.
Start with the trigger word "{trigger}".
Include: subjects, their actions and movements, environment, lighting, camera angle.
Describe motion and temporal progression, not just static appearance.
Keep under 75 words. Use plain descriptive English.
```

## Regularization Images

Add 3-5 generic surveillance clips per category (no violence, just normal activity) to prevent Wan 2.2 from overriding the surveillance aesthetic with its default cinematic look. Place in the same processed directory — they get captioned WITHOUT the trigger word so the model associates the trigger specifically with the violent action, not the camera angle.

## ComfyUI Generation Script Fix

The original `generate.py` references `CLIPTextEncode` nodes. Wan 2.2 uses a **T5-based text encoder**, not CLIP. The workflow JSON should be exported from a working ComfyUI Wan 2.2 T2V workflow and loaded as-is, with only the prompt/seed/LoRA fields modified programmatically. Don't hardcode node class names.

```python
def build_workflow(prompt: str, seed: int, category: str, cfg: float = 4.0) -> dict:
    """Load a saved workflow JSON and patch variable fields."""
    template = json.loads(WORKFLOW_TEMPLATE_PATH.read_text())
    # Patch prompt, seed, LoRA path into the correct node IDs
    # Node IDs come from YOUR exported workflow — don't guess them
    ...
```

## Cross-References

- [[training-parameters]] — Source for revised values
- [[dual-noise-architecture]] — Why train_high_noise/train_low_noise is critical
- [[hardware-requirements]] — 24GB vs A100 tradeoffs
- [[resolution-guide]] — Why 512×768 for 24GB
- [[captioning]] — Multi-frame captioning rationale
- [[overfitting-and-troubleshooting]] — Regularization image guidance
- [[ai-toolkit]] — Correct config syntax
