---
title: Dataset Preparation
type: concept
tags: [dataset, preparation, video, images, training-data]
created: 2026-05-12
updated: 2026-05-12
---

# Dataset Preparation

The quality of your LoRA is bounded by the quality of your dataset. Wan 2.2 LoRA training accepts either video clips or still images, but the format must be uniform — **entirely images OR entirely videos**, no mixing.

## File Format

| Type | Media | Caption | Example |
|------|-------|---------|---------|
| Image LoRA | `.jpg` / `.png` | Same-name `.txt` | `img_001.png` + `img_001.txt` |
| Video LoRA | `.mp4` | Same-name `.txt` | `clip_001.mp4` + `clip_001.txt` |

Every media file must have a matching caption file. See [[captioning]] for how to write effective captions.

## Images vs Video: When to Use Each

| Use Case | Images OK? | Video Required? |
|----------|-----------|----------------|
| Character/identity | Yes (12-20 images) | Optional but better |
| Style/aesthetic | Yes (30-50 images) | Optional |
| Motion (camera, action) | **No** | **Yes, mandatory** |
| I2V behavior | No | Yes |

**Critical**: you cannot train effective motion LoRAs on still images hoping the model figures out motion. Video clips showing actual movement are mandatory for motion learning. [apatero-best-practices]

## Dataset Size

| LoRA Type | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Identity | 10-12 | 15-20 | Diverse poses, angles, lighting |
| Style | 20-30 | 30-50 | Variety of subjects in the target style |
| Motion | 12-15 | 20-30 | Consistent motion type across clips |
| 1024px training | 35+ clips | 50+ | Fewer clips = severe overfitting at high res |

## Video Clip Requirements

- **Length**: 4-8 seconds per clip
- **FPS**: 24fps minimum, consistent across all clips
- **Resolution**: 720p+ source material
- **Quality**: minimal compression artifacts, good lighting, visible detail
- **Motion**: smooth and continuous — no jump cuts, scene changes, or abrupt transitions
- **Consistency**: same resolution and FPS across all clips in the dataset

### Preprocessing Checklist

1. Trim clips to relevant portions (remove dead time, transitions)
2. Stabilize shaky footage if necessary
3. Ensure consistent resolution across clips
4. Remove clips with visible compression artifacts
5. Verify smooth frame rates (no dropped frames)
6. Remove clips with multiple scene changes

## Resolution Matching

**Never train at a resolution higher than your source video.** Training at 1024px from 720p source material teaches the model to reproduce compression artifacts and upscaling patterns. See [[resolution-guide]].

## Dataset Config Examples

### Musubi-tuner TOML (video + image mixed config)

```toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
image_directory = "train/image"
cache_directory = "train/cache_image"
resolution = [1024, 1024]
num_repeats = 1

[[datasets]]
video_directory = "train/video"
cache_directory = "train/cache_video"
frame_extraction = "uniform"
source_fps = 16.0
target_frames = [57]
max_frames = 57
enable_bucket = true
bucket_no_upscale = false
resolution = [384, 384]
```

### AI Toolkit YAML (datasets section)

```yaml
datasets:
  - folder_path: /path/to/dataset
    num_frames: 81
    resolution: [512, 768]
    caption_dropout_rate: 0.05
```

## Pre-Caching

Both [[musubi-tuner]] and [[ai-toolkit]] support pre-caching latents and text encoder outputs before training. This saves VRAM during training by avoiding redundant encoding passes. With [[musubi-tuner]], Wan 2.2 does **not** require a CLIP model (unlike Wan 2.1).

## Common Mistakes

- Mixing images and videos in the same dataset
- Using clips with jump cuts or scene transitions
- Training at higher resolution than source material
- Too few clips for the target resolution (< 15 at 1024px)
- Inconsistent FPS or resolution across clips
- Mixed camera + subject motion diluting what the LoRA learns

## Related

- [[captioning]] — How to write captions
- [[resolution-guide]] — Choosing training resolution
- [[training-parameters]] — Setting up the training run
- [[overfitting-and-troubleshooting]] — When dataset problems cause training issues
