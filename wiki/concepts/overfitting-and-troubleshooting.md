---
title: Overfitting and Troubleshooting
type: concept
tags: [overfitting, troubleshooting, artifacts, debugging, training-issues]
created: 2026-05-12
updated: 2026-05-12
---

# Overfitting and Troubleshooting

Wan 2.2 LoRA training has specific failure modes. The trickiest part: **WAN hides overfit behind pretty samples** — outputs look good but lack generalization. [wavespeed-training-settings]

## Overfitting Detection

### Key Tells

| Symptom | What It Means |
|---------|---------------|
| **Prompt inertia** | Changing the prompt barely changes output — everything drifts to same composition | 
| **Skin plasticity** | Pores vanish uniformly, especially cheeks/foreheads, even with gritty lighting |
| **Color clustering** | Dataset color bias amplified — if training data leans warm, everything goes warm |
| **Background homogenization** | All outputs default to shallow depth-of-field and soft bokeh |
| **Exact reproduction** | Generated outputs only reproduce training examples |
| **Prompt-locked** | LoRA only works with prompts matching training captions |

### Quantitative Tells

- Training loss drops very low but validation quality is poor
- Loss curve looks good but generalization testing fails
- Earlier checkpoints produce more diverse outputs than later ones

## Common Issues and Fixes

### 1. Overfitting

**Causes**: too many steps, too small dataset, too high learning rate, too high rank

**Fixes**:
- Stop earlier (try 2000 steps instead of 4000)
- Add more varied training data
- Lower learning rate by 30-50%
- Reduce rank (32 → 16)
- Add **regularization images** (10-20% of training batches) — gives the model permission to maintain base model variety
- Use caption dropout (0.05) for generalization

### 2. Artifacts

**Causes**: overfitting, excessive rank, training data quality, upscaling artifacts in source

**Fixes**:
- Reduce learning rate
- Decrease training steps by 30%
- Verify training data doesn't contain artifacts
- Test earlier checkpoints
- **Never train at resolution higher than source** — 1024px from 720p source teaches compression artifacts

### 3. Motion Artifacts

**Causes**: shaky footage, jump cuts, or scene changes in training clips

**Fixes**:
- Stabilize source footage
- Remove clips with abrupt transitions
- Ensure smooth, continuous motion in training data
- Avoid mixing camera motion with subject motion (dilutes learning)

### 4. Loss Spikes / Training Instability

**Causes**: learning rate too high, corrupted training data

**Fixes**:
- Reduce LR to 0.00015
- Check dataset for corrupted or malformed clips
- Verify consistent FPS and resolution across clips

### 5. LoRA Has No Effect

**Causes**: wrong dual-noise setup, undertrained, wrong LoRA loading

**Fixes**:
- Ensure both high-noise and low-noise LoRAs are trained (see [[dual-noise-architecture]])
- Verify LoRA is loaded correctly in inference tool
- Increase training steps
- Check that trigger token matches between training captions and inference prompts

### 6. Style and Subject Blending

**Cause**: trying to teach style AND subject simultaneously

**Fix**: WAN 2.2 overfits fast when learning two concepts at once. Train separate LoRAs for style and subject, then combine at inference.

## Wan 2.2-Specific Behavior

Wan 2.2 "behaves like a very opinionated SDXL checkpoint" — it pushes toward:
- Crisp portraits
- Smooth gradients
- Cinematic lighting
- Shallow depth of field

Training against these tendencies requires explicit countermeasures:
- **Regularization images** at 10-20% of batches throughout training (not just start)
- Diverse backgrounds and lighting in training data
- Varied camera angles and compositions

## Checkpoint Management

**Always save intermediate checkpoints** (every 500-1000 steps). The best checkpoint is often not the final one.

Testing protocol:
1. Generate with each checkpoint using training-similar prompts
2. Generate with each checkpoint using novel prompts (different from training captions)
3. A good checkpoint produces recognizable concepts in novel contexts
4. An overfit checkpoint only works with training-like prompts

## Loss Monitoring

| Pattern | Meaning | Action |
|---------|---------|--------|
| Steady decrease | Normal training | Continue |
| Plateau | Learning rate may be too low or model has converged | Consider stopping or adjusting LR |
| Upward trend | Training is degrading | Stop, use earlier checkpoint |
| Erratic/spiky | LR too high or bad data | Reduce LR, audit dataset |
| Very low plateau | Likely overfit | Use earlier checkpoint, add data |

## Related

- [[training-parameters]] — Adjusting parameters to fix issues
- [[dataset-preparation]] — Data quality requirements
- [[resolution-guide]] — Resolution-related overfitting
- [[dual-noise-architecture]] — Wan 2.2-specific training setup
