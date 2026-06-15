# Project Handoff / Session State

Portable record of this project's state, decisions, and hard-won gotchas so work can
continue on another machine. Distilled memory notes live in [`docs/session-memory/`](docs/session-memory/).
Last updated: 2026-06-15.

## Goal

End-to-end pipeline that trains category-specific LoRAs on **Wan 2.2** and generates
**labeled synthetic video** for training violence / threat / self-harm **detection**
models (fighting, vandalism, stabbing, shooting, self-injury). Synthetic data only;
purpose is detection (flag incidents for staff), not real harm. See `plan.md`.

## Hardware (machine-specific — adjust on a new box)

- **2× NVIDIA RTX A5500, 24 GB each**, 128 GB system RAM.
- GPU 0 also runs the desktop (~1.3 GB baseline) → slightly less headroom than GPU 1.
- Repos live under `/home/lenovo5/TiongKai/Greenfield/`: `wan2.2-lora/` (this repo),
  `musubi-tuner/` (training), `ComfyUI/` (generation). All three are siblings.
- Absolute paths are baked into `scripts/*.sh` and `ComfyUI/extra_model_paths.yaml` —
  update them when relocating.

## Current state (2026-06-15)

| Artifact | State |
|---|---|
| **T2V fighting LoRA** (`loras/fighting/`) | ✅ trained, 3000 steps, 512×768/33f. Works. |
| **I2V fighting LoRA** (`loras/fighting_i2v/`) | ✅ trained, 3000 steps, 33f. Sharp to ~2 s. |
| **I2V 77-frame LoRA** (`loras/fighting_i2v_77f/`) | ❌ NOT trained. OOMs at 480×832 even at max block-swap. Needs **416×720** re-cache (see below). |
| **100× 5 s fighting clips** (`generated/clips/fighting_i2v/`) | ✅ generated with the 33f LoRA. Blurry past ~2 s (LoRA trained at 33f; see Finding #2). |
| **Base-model baselines** (`generated/clips/fighting_i2v_base/`) | ✅ stock Wan2.2 I2V, no LoRA, from `sample.png`. Hold 5 s motion well. |
| **Self-injury clips** (`generated/clips/self_injury_i2v_base/`) | 🔄 base-model, headbutt-wall, generating at time of writing. |
| Other categories (vandalism/stabbing/shooting) | ❌ no training data downloaded yet. |

`sample.png` (repo root) = a real out-of-trainset still: a **prison phone-bank** scene
(maroon jumpsuits, blue payphones, numbered booths, elevated surveillance angle).

## Key findings / gotchas (the non-obvious stuff)

1. **Training VRAM (24 GB):** the default `blocks_to_swap 20` OOMs at step 1. Use
   `blocks_to_swap 30` + `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   ~29 s/step, ~25 h for 3000 steps dual-noise. VRAM peaks ~21 GB during steps.
2. **Clip length is capped by fps, not just VRAM.** Musubi resamples training video to
   Wan's native **16 fps**. A **5.0 s clip = exactly 80 frames @ 16 fps**, so
   `target_frames=81` (needs 5.06 s) silently drops **every** clip → "No training items."
   Valid Wan lengths are `4n+1`; the largest that fits 80 is **77** (4.81 s). Train long
   I2V at **77 frames**, not 81. (33 worked only because 33 < 80.)
3. **`--i2v` is caching-only.** `wan_cache_latents.py` takes `--i2v`; `wan_train_network.py`
   REJECTS it (exit 2) — training infers I2V from `--task i2v-A14B`.
4. **Wan 2.2 I2V needs NO CLIP-vision model** (unlike 2.1) — reuses the same T5 + VAE.
5. **Generation VRAM:** a 14B T2V/I2V + LoRA does NOT fit on 24 GB by default (OOMs during
   LoRA patching). Required: ComfyUI `--lowvram --disable-cuda-malloc` **and** the text
   encoder on CPU (`CLIPLoader device=cpu`, frees ~11 GB). ~15 s/clip at 33f, **~15 min at 81f**.
   77f training OOMs even at `blocks_to_swap 39` (max) — bottleneck is activation memory,
   so the only lever left is **lower resolution** (use **416×720**, needs a latent re-cache).
6. **Stock models, NOT uncensored.** Generate with stock Wan2.2 + the trained LoRA. Do
   NOT use the Wan2.2-Remix NSFW encoder (the LoRA carries the content). Use stock
   `umt5_xxl_fp16.safetensors`.
7. **Killing ComfyUI:** its cmdline is `main.py` (run from the ComfyUI cwd), so
   `pkill -f ComfyUI/main.py` does NOT match. Kill by port:
   `for p in $(fuser 8188/tcp); do kill -9 $p; done`.
8. **Captioning:** Ollama + `huihui_ai/qwen2.5-vl-abliterated:7b` (already pulled). Standard
   Qwen refuses ~74% of violent clips; abliterated ~84% success. See `plan.md` Experiment Log.

## Scripts & how to run

- `scripts/train_category.sh CATEGORY [STEPS]` — T2V LoRA (dual-noise, swap 30).
- `scripts/train_i2v_category.sh CATEGORY [STEPS]` — I2V LoRA, 33f.
- `scripts/train_i2v_77f.sh [STEPS]` — I2V LoRA at 77f/480×832 (**currently OOMs**; drop
  res to 416×720 in `configs/fighting_i2v_77f_dataset.toml` + the script, re-cache, retry).
- `scripts/run_comfyui.sh` — starts ComfyUI on GPU 1 (override `GPU=`/`PORT=`), lowvram flags
  baked in, output → `generated/clips/`. For dual-GPU batches run two instances (8188+8189).
- `scripts/generate.py` — T2V batch (ComfyUI API client).
- `scripts/generate_i2v.py` — I2V batch. Flags: `--image` / `--image-dir` (cycles frames),
  `--prompts-file`, `--no-lora` (stock baseline), `--frames` (33→2 s, 77→4.8 s, 81→5 s),
  `--width`/`--height`, `--shard`/`--num-shards` (dual-GPU split), `--port`. Output filenames
  `sample_NNNN`, per-shard manifest CSVs. `wait_for_completion` checks real run status.
- ComfyUI workflow templates (validated against `/object_info`):
  `scripts/comfyui_wan22_t2v_workflow.json`, `..._i2v_workflow.json`,
  `..._i2v_base_workflow.json` (no LoRA).

Example — 5 base-model clips from a still, scene-aligned prompts:
```bash
bash scripts/run_comfyui.sh                       # GPU 1
.venv/bin/python scripts/generate_i2v.py --no-lora --category fighting \
  --image sample.png --prompts-file scripts/prompts/fighting_prison.txt \
  --count 5 --frames 81 --width 720 --height 512 --port 8188
```

## Open / next steps

- **77-frame I2V LoRA**: re-cache + train at **416×720** (fits 24 GB; quick latent re-cache).
- **Base vs LoRA A/B**: same prompts+seeds through the 33f LoRA vs `--no-lora`.
- **Other categories**: download datasets (UCF-Crime, AIRT Lab — see `plan.md`), then train.
- **Dataset packaging**: `scripts/annotate.py` + `scripts/validate_dataset.py` over generated clips.

## Portability checklist for a new machine

1. Clone the three sibling repos (`wan2.2-lora`, `musubi-tuner`, `ComfyUI`).
2. Recreate venvs: `wan2.2-lora/.venv` (training/clients) and `ComfyUI/.venv`
   (torch 2.6.0+cu124 on this rig). musubi installed via `uv`.
3. Fix absolute paths in `scripts/*.sh` and `ComfyUI/extra_model_paths.yaml`.
4. Download model weights (NOT in git — see `.gitignore`): Wan2.2 T2V + I2V experts, VAE,
   umt5 T5 → `models/...` (Comfy-Org/Wan_2.2_ComfyUI_Repackaged; pass SPECIFIC file paths
   to `hf download`, never the whole repo — it's 100 GB+).
5. Pull the captioner: `ollama pull huihui_ai/qwen2.5-vl-abliterated:7b`.
6. The trained LoRAs (`loras/`) are also gitignored — copy them over separately or retrain.
