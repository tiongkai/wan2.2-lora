# Quickstart — Wan 2.2 Synthetic Clip Generation

How to generate synthetic video clips with this platform (stock Wan 2.2 + ComfyUI). For full
project state and gotchas see [HANDOFF.md](../HANDOFF.md); for results see
[experiments-log.md](experiments-log.md).

> **TL;DR workflow:** start ComfyUI → Qwen-caption your start image for scene context →
> write a scene-grounded, single-action, motion-descriptive prompt → run a generator → clip
> lands in `generated/clips/`.

All commands run from the repo root: `cd /home/lenovo5/TiongKai/Greenfield/wan2.2-lora`.
Python scripts are a package — invoke with **`-m scripts.<name>`** (or `PYTHONPATH=. python scripts/<name>.py`).

---

## 0. Prerequisites (one-time)
- ComfyUI installed at `../ComfyUI` with its own venv; models wired via `../ComfyUI/extra_model_paths.yaml` (no copying). Stock Wan 2.2 T2V + I2V experts, `umt5_xxl` text encoder, `wan_2.1_vae`.
- Project venv at `.venv` (Python 3.11, torch 2.6 cu124) for the client scripts.
- **Ollama** running with the captioner pulled: `ollama pull huihui_ai/qwen2.5-vl-abliterated:7b`.
- ffmpeg at the path in the scripts (for frame extraction / concat).
- (New machine? Follow the checklist in HANDOFF.md.)

## 1. Start ComfyUI
```bash
bash scripts/run_comfyui.sh           # GPU 1 by default; lowvram flags baked in
# override: GPU=0 PORT=8189 bash scripts/run_comfyui.sh
```
Outputs are written under `generated/clips/`. Leave it running; it serves the HTTP API on :8188.
(Stop it later: `for p in $(fuser 8188/tcp); do kill -9 $p; done` — `pkill -f ComfyUI/main.py` does NOT match.)

## 2. Annotate the start image with Qwen (get scene context)
Image-to-video conditions on a start frame, so prompts work best when **grounded in what's
actually in that frame**. Use Qwen to extract that context:
```bash
.venv/bin/python scripts/caption_image.py carpark-singapore.jpg
```
Example output → *"open-air carpark; silver Honda parked centre-front, yellow taxi left, black
SUV right; red-and-yellow checkered facade behind; daytime, eye-level camera."*

Then turn that into a prompt following the rules we learned work best:
- **Scene-grounded** — reuse the caption's concrete details (objects, colours, camera, lighting).
- **One clear action** — a single focused event animates more coherently than several at once.
- **Motion-descriptive (kinematic)** — describe the *movement*, not an abstract verb.
  - ✅ "the silver car explodes in a fireball, flames and black smoke bursting outward, debris flying"
  - ❌ "car explosion"

**Or automate it with Qwen** — `enhance_prompt.py` looks at the start image and expands a short
idea into a grounded prompt following the rules above (collapses caption→craft into one call):
```bash
.venv/bin/python scripts/enhance_prompt.py carpark-singapore.jpg "the yellow taxi explodes"
# -> "yellow taxi explodes in mid-air, debris scattering across concrete barrier and grassy area,
#     multi-level parking structure behind, daylight surveillance view"
```
Or do it inline during generation with `--auto-prompt "<idea>"` (see §3) — it enhances per start
frame and prints the prompt it used. The 7B model is modest, so eyeball/edit for important runs.

## 3. Generate

### Image-to-video, single clip (most common)
```bash
.venv/bin/python -m scripts.generate_i2v \
  --no-lora --category fighting \
  --image carpark_832x480.png \
  --prompt "the silver car explodes in a huge fireball, ... daytime, eye-level camera" \
  --frames 81 --width 832 --height 480 --count 1 --port 8188
```
- `--no-lora` = stock Wan 2.2 (recommended; the trained LoRAs add blur — see HANDOFF).
- `--frames`: 33 ≈ 2 s, 81 ≈ 5 s (16 fps). `--count N` for N variations; `--image-dir DIR` cycles frames; `--prompts-file FILE` cycles prompts; **`--auto-prompt "<idea>"`** lets Qwen write the prompt from the start image.
- Output → `generated/clips/<category>_i2v_base/` + a manifest CSV (prompt/seed/start-frame per clip).

### Text-to-video
```bash
.venv/bin/python -m scripts.generate --category fighting --count 4
```
Output → `generated/clips/<category>/`.

### Long-form (10 s+): extend-and-stitch  ← best motion quality
Chains N×5 s base-model windows (each segment's last frame → next segment's start), then concats.
```bash
.venv/bin/python -m scripts.extend_video \
  --no-lora --image carpark_832x480.png --segments 2 --frames 81 \
  --width 832 --height 480 --prompt "<scene-grounded prompt>" \
  --port 8188 --out generated/clips/long/my_clip_10s.mp4
```
- `--segments 2` → ~10 s, `4` → ~20 s. Per-segment clips land in `generated/clips/long_chain/`.
- Trade-off: livelier motion, but a visible seam every 5 s.

### Long-form: SVI (seamless, drift-corrected)  ← best continuity
For very long clips where stitching would drift. Uses the SVI workflow JSONs + WanVideoWrapper.
Edit the prompt in `scripts/comfyui_svi_nodistill_long_workflow.json` (node 16) and queue it:
```bash
.venv/bin/python -c "import urllib.request,json,uuid; wf=json.load(open('scripts/comfyui_svi_nodistill_long_workflow.json')); urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8188/prompt',data=json.dumps({'prompt':wf,'client_id':str(uuid.uuid4())}).encode(),headers={'Content-Type':'application/json'}))"
```
- `nodistill` = full-quality (slow); the distilled variant is faster but softer. Output → `generated/clips/svi/`.
- Trade-off: seamless + consistent, but weaker scene progression than extend-and-stitch.

## 4. Resolution — match the model's native buckets
Wan 2.2 I2V is trained at **832×480 / 480×832 (480p)** and **1280×720 / 720×1280 (720p)**.
Generate at the bucket matching your image's orientation, and **pre-crop the start image to that
exact size** (no stretch) — e.g. `ffmpeg -i img.jpg -vf "crop=W:H:X:Y,scale=832:480" out.png`.
Mismatched orientation or off-bucket sizes cause artifacts.

## 5. Where things go
| Output | Location |
|---|---|
| I2V base clips | `generated/clips/<category>_i2v_base/` |
| I2V LoRA clips | `generated/clips/<category>_i2v/` |
| T2V clips | `generated/clips/<category>/` |
| Long-form (final) | `generated/clips/long/` and `generated/clips/svi/` |
| Per-segment sources | `generated/clips/long_chain/` |
| Manifests (prompt/seed/etc.) | `*_manifest.csv` beside the clips |

## 6. Common gotchas
- **Stock model > trained LoRA** for quality (LoRA blur = low-quality training data ceiling).
- **Native buckets only** (see §4); pre-crop, don't stretch.
- **SVI:** `base_precision=fp16` (not `fp16_fast`, needs torch 2.7); native bucket; the in-workflow resize node can force off-bucket dims.
- **Clip length:** a 5.0 s source = 80 frames @16 fps, so 81-frame *training* is impossible (use 77); generation at 81 is fine.
- **`wait_for_completion`** returns `(status, entry)`; presence in ComfyUI history ≠ success (it logs failed/OOM runs too).
