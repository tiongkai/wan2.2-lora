# Synthetic Data Pipeline

This is the operational path for turning Wan2.2 generations into detector
training data.

## 1. Generate

Start ComfyUI:

```bash
bash scripts/run_comfyui.sh
```

Generate base I2V clips:

```bash
.venv/bin/python scripts/generate_i2v.py --no-lora --category fighting \
  --image sample_832x480.png \
  --prompts-file scripts/prompts/fighting_prison.txt \
  --count 5 --frames 81 --width 832 --height 480
```

Each generation script writes a manifest next to its outputs. New manifests use a
canonical schema with actual `output_path`, method, prompt, seed, workflow hash,
model stack, and LoRA metadata.

## 2. Validate

Validate all generated manifests:

```bash
.venv/bin/python scripts/validate_dataset.py
```

The validator writes `generated/annotations/validation_report.csv` with duration,
resolution, frame count, blur score, static score, duplicate warning, and
pass/fail status.

## 3. Export

Dry-run export:

```bash
.venv/bin/python scripts/annotate.py --dry-run
```

Write annotations:

```bash
.venv/bin/python scripts/annotate.py
```

Export can be filtered by method:

```bash
.venv/bin/python scripts/annotate.py --methods i2v_base,svi
```

## Register Existing Outputs

For videos generated directly in ComfyUI, such as SVI experiments, register the
final mp4 before validation/export:

```bash
.venv/bin/python scripts/register_generated.py \
  --output generated/clips/svi/example.mp4 \
  --category fighting \
  --method svi \
  --prompt "fght99, two people fight in a prison phone-bank area" \
  --frames 321 --width 832 --height 480 \
  --workflow-template scripts/comfyui_svi_nodistill_long_workflow.json
```

## Notes

- Wan training frame counts must be `4n+1`.
- A 5.0 second source clip has 80 effective frames at Wan's 16 fps, so the largest
  valid training length is 77 frames, not 81.
- 81 frames is still useful for roughly 5 second generation.
- Machine-specific defaults live in `configs/pipeline.yaml`; command-line flags
  should be treated as run-specific overrides.
