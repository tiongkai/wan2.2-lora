# Experiments Log

Chronological log of training & generation experiments, results, and where data is stored.
Rig: 2× RTX A5500 (24 GB). Last updated 2026-06-16. See also [HANDOFF.md](../HANDOFF.md),
[wiki/synthesis/long-video-generation-wan22.md](../wiki/synthesis/long-video-generation-wan22.md).

## Headline conclusions
- **Use the base Wan2.2 model, not the trained LoRAs.** The fighting LoRA shows real motion but
  adds blur — a ceiling from the low-quality surveillance training footage, not a broken run.
- **Long-form options (confirmed at 20 s):** base-model extend-and-stitch has **livelier motion &
  scene changes** but **visible seams** every 5 s; SVI is **seamless & drift-corrected** but has
  **weaker scene progression** (it anchors to hold consistency) — and needs full steps (no
  LightX2V distillation) to avoid softness. Pick per use-case; for these scenes the base extend's
  motion read better, SVI's continuity read better.
- **Clip length cap:** a 5.0 s source clip = 80 frames at Wan's native 16 fps, so 81-frame
  training is impossible; 77 is the max valid (4n+1) length.

## Training experiments

| LoRA | Config | Result | Stored |
|---|---|---|---|
| **T2V fighting** | 3000 steps, 512×768, 33f, rank32 | Works for T2V | `loras/fighting/` (final + step2700/3000) |
| **I2V fighting** | 3000 steps, 480×832→33f, rank32 | Visible fighting motion but **blurry** (data-quality ceiling). **Shelved.** | `loras/fighting_i2v/` (final + 20 ckpts) |
| **I2V 77f** | 77f/480×832 attempt | **Failed — OOM** on 24 GB even at blocks_to_swap 39 (activation memory bound). Not trained. | — (none) |
| I2V 81f | 81f attempt | **Failed** — 5 s clips only yield 80 frames @16fps; caching produced 0 items. | — |
| shooting / stabbing / vandalism | — | Not started (no datasets downloaded) | `loras/{cat}/` empty |
| **SVI Pro + LightX2V** | downloaded (not trained) | Used for SVI long-form | `loras/svi/` (4 LoRAs) |

Caption quality was checked and is GOOD (detailed fight descriptions); loss converged
(I2V high-noise 0.069→0.049). So the LoRA blur is data quality, not training error.

## Generation experiments (short clips)

| Run | Method / settings | Result | Stored |
|---|---|---|---|
| T2V fighting | T2V LoRA, 2 s & 5 s | Works | `generated/clips/fighting/` (2) |
| **100-clip batch** | I2V 33f LoRA, 5 s (81f), 720×512, dual-GPU, 31 start frames | All 100 ok, but **blurry past 2 s** (LoRA trained at 33f) | `generated/clips/fighting_i2v/` (~105 incl. smoke/diag) |
| Base baselines | Base model (no LoRA), 5 clips, generic prompts | Base clean; generic prompts mismatched scene | `generated/clips/fighting_i2v_base/` `sample_000*_00001` |
| Base + Qwen prompts | Base model, 5 clips, prison-aligned prompts (Qwen-captioned) | Better — prompts match the scene | `generated/clips/fighting_i2v_base/` `sample_000*_00002` |
| Self-injury (short) | Base model, 5 clips, brief prompts | — | `generated/clips/self_injury_i2v_base/` `*_00001` |
| Self-injury (motion) | Base model, 5 clips, kinematic prompts | Better motion description | `generated/clips/self_injury_i2v_base/` `*_00002` |
| Diagnostic 33f vs 81f | sample.png, same seed, LoRA | Confirmed blur begins exactly past 2 s (33-frame training edge) | `/tmp/diag/` (A_33frames_2s, B_81frames_5s) |

## Long-form experiments (~20 s)

| Clip | Method | Resolution | Result | Stored |
|---|---|---|---|---|
| **prison_brawl_20s** | Base model extend-and-stitch, 4× 5 s I2V windows chained + concat | 720×512 | **Best motion sync** (user's benchmark); visible seams every 5 s | `generated/clips/long/prison_brawl_20s.mp4` |
| fighting_lora_20s | 2 s fighting-LoRA chain, 10× 33f segments | 720×512 (stretched — unfair) | LoRA blur; bad (wrong aspect) | `generated/clips/long/fighting_lora_20s.mp4` |
| fighting_lora_512x768_20s | 2 s fighting-LoRA chain, 10× 33f | 512×768 (trained dims, fair) | Still blurry → confirms data-quality ceiling | `generated/clips/long/fighting_lora_512x768_20s.mp4` |
| (per-segment sources) | individual chain segments (runs) | — | inputs to the concats above | `generated/clips/long_chain/` (by mtime) |
| **prison_brawl_svi_20s** | SVI no-distill (full 20-step) + context-options sliding window, 321f | 832×480 | Seamless 20 s; **scene changes weaker** than base brawl (user) | `generated/clips/svi/prison_brawl_svi_20s_00001_.mp4` |
| **self_injury_base_20s** | Base extend-and-stitch, 4× 5 s, headbutt-wall prompt | 832×480 | Livelier; self-injury visible (1st attempt failed on a `--no-lora` flag bug, since fixed) | `generated/clips/long/self_injury_base_20s.mp4` |
| **self_injury_svi_20s** | SVI no-distill + context options, 321f, headbutt-wall prompt | 832×480 | Seamless; self-injury visible | `generated/clips/svi/self_injury_svi_20s_00001_.mp4` |

Long-form workflows: `scripts/comfyui_svi_i2v_long_workflow.json` (distilled), `comfyui_svi_nodistill_long_workflow.json` (full-quality), `comfyui_svi_selfinjury_long_workflow.json`. Prompts are scene-grounded (Qwen caption), single-action, kinematic. `extend_video.py` gained `--lora`/`--no-lora`/`--category`/`--strength`.

## SVI 2.0 Pro experiments (all 5 s validation windows so far)

Base Wan2.2 + SVI Pro + LightX2V LoRAs via WanVideoWrapper. Same prompt+seed(43) each;
only resolution/prompt changed. **Long-form SVI run still pending.**

| Clip | Resolution | Note | Stored |
|---|---|---|---|
| svi_test_00001 | 480×832 portrait | **Artifacts** — landscape source force-cropped to portrait | `generated/clips/svi/` |
| svi_test_00002 | 832×464 | landscape, off native bucket (resize node forced 464) | `generated/clips/svi/` |
| svi_test_00003 | 832×464 | exact 832×480 input but resize node still gave 464 | `generated/clips/svi/` |
| svi_test_00004 | **832×480** ✅ native bucket | resize node bypassed; correct resolution | `generated/clips/svi/` |
| svi_test_00005 | 832×480 | one-clear-action detailed prompt (match prison-brawl style) | DONE `generated/clips/svi/` |

SVI gotchas found: `base_precision` must be `fp16` not `fp16_fast` (torch 2.6); the
`WanVideoImageResizeToClosest` node forces off-bucket dims (bypassed it); native I2V buckets
are 832×480 / 480×832 / 1280×720 / 720×1280. Workflow: `scripts/comfyui_svi_i2v_workflow.json`.

## Data storage map
- **Trained LoRAs:** `loras/<category>/` (gitignored — large)
- **Generated clips:** `generated/clips/<category>[_base|_i2v]/`, long-form in `generated/clips/long/`,
  per-segment in `generated/clips/long_chain/`, SVI in `generated/clips/svi/` (gitignored)
- **Models:** `models/wan2.2-t2v/`, `models/wan2.2-i2v/` (gitignored)
- **Start image:** `sample.png` (prison phone-bank, low-res 464×324) + `sample_832x480.png`, `sample_portrait.png`
- **Manifests** (prompt/seed/start-frame per clip): `*_manifest*.csv` in each `generated/clips/*` folder
- **Faststart copies sent to user:** `/tmp/{diag,base,prison,si,motion}/` (ephemeral, not in repo)
- **Scripts:** `scripts/generate.py` (T2V), `generate_i2v.py` (I2V, `--no-lora`/`--image-dir`/`--prompts-file`),
  `extend_video.py` (long-form stitch, `--lora`), `comfyui_*_workflow.json` (workflows)
