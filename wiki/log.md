# Wiki Log

## [2026-06-15] synthesis | Long-video generation with Wan 2.2 (deep research)

Added `wiki/synthesis/long-video-generation-wan22.md` from a deep-research pass (22 sources, 13 claims surviving 3-vote adversarial verification). Key verified findings: (1) **SVI / Stable Video Infinity** is the leading long-video method — "Error-Recycling Fine-Tuning" trains the DiT to correct its own drift, beating naive extend-and-stitch; SVI 2.0 LoRAs exist for Wan 2.1 and 2.2 (Wan 2.2 = 1 motion frame; Kijai fp16 LoRAs on HF). (2) Consistency via ~80-frame first-frame padding (dynamics↔consistency tradeoff). (3) **MoE sigma boundary 0.90 I2V / 0.875 T2V**, auto-computed by `ComfyUI-WanMoEScheduler` (defaults 4+4 steps, shift precision 0.01). (4) Native FLF2V node = `WanFirstLastFrameToVideo`. (5) LightX2V scale per expert (high 0.5–0.6 / low 1.0 + SVI 1.0). Unverified-but-sourced leads: `ComfyUI-Wan-SVI2Pro-FLF` stitch pack, Qwen `InSubject`/`InScene`/`next-scene` keyframe LoRAs. Refuted: "infinite at no cost", a specific 1025-frame sliding-window claim. NOTE: the deep-research workflow's final synthesis agent failed (transient API ConnectionRefused + a structured-output bug, since fixed to return plain markdown); this page was synthesized by hand from the verified claims. Updated index.md.

## [2026-06-06] synthesis | End-to-end run findings on 2× RTX A5500 (24GB)

Added `wiki/synthesis/dual-a5500-run-findings.md` from the first full fighting-LoRA run on the project rig (first-hand, not web-sourced). Key findings: (1) Training — default `blocks_to_swap 20` OOMs at step 1; fix is `blocks_to_swap 30` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; ~25 h for 3000 steps dual-noise; high-noise loss 0.30→0.09 (meaningful), low-noise flat ~0.09 (normal). (2) Generation — 14B T2V + LoRA OOMs during LoRA patching on 24 GB; fix requires all of `--lowvram` + `--disable-cuda-malloc` + text encoder on CPU (`CLIPLoader device=cpu`); ~350 s/clip. (3) Decision: stock Wan2.2 + LoRA, NOT the Remix NSFW encoder. (4) Clips are 33 frames @16fps ≈ 2 s by design for 24 GB; 5 s needs 81 frames (much more VRAM/time, LoRA trained at 33). Updated index.md. Validated ComfyUI workflow against `/object_info`; first verified clip produced (h264 512×768 33f).

## [2026-05-12] ingest | Initial wiki creation

Built the initial wiki from web research covering 15+ sources. Ingested:

- 10 source summaries (WaveSpeed, Civitai, Apatero x3, StableDiffusionTutorials, AMD ROCm, Ostris YouTube, musubi-tuner discussions, ArXiv paper, YouTube collection)
- 8 concept pages (LoRA, dataset prep, captioning, training params, dual-noise architecture, hardware, resolution, overfitting/troubleshooting)
- 6 entity pages (Wan 2.2, Wan 2.1, musubi-tuner, AI Toolkit, diffusion-pipe, ComfyUI)
- 2 synthesis pages (training tools comparison, Wan 2.1 vs 2.2)
- 1 overview page

Sources searched: Google web search (multiple queries), YouTube search. Key domains: wavespeed.ai, civitai.com, apatero.com, stablediffusiontutorials.com, rocm.blogs.amd.com, github.com (kohya-ss/musubi-tuner, ostris/ai-toolkit), medium.com, huggingface.co, arxiv.org, youtube.com.

User-requested source: https://www.youtube.com/watch?v=2d6A_l8c_x8 (Ostris Wan 2.2 I2V 14B LoRA training)

## [2026-05-13] concept | Multi-GPU training strategies for 2×24GB

Added wiki/concepts/multi-gpu-training.md. Two strategies: parallel dual-noise (train high/low experts simultaneously on separate GPUs via musubi-tuner) and pipeline parallelism (split model layers across GPUs via diffusion-pipe/DeepSpeed for ~48GB effective VRAM). Updated plan.md with parallel dual-noise train_all.sh script and diffusion-pipe pipeline config notes.

## [2026-05-12] synthesis | Plan critique — violence detection LoRA configs

Reviewed `/Users/htx/Desktop/Projects/wan2.2-lora-1try/plan.md` against the wiki. Found 12 issues (2 critical, 7 significant, 3 minor). Produced revised configs in `wiki/synthesis/plan-critique-revised-configs.md`. Critical fixes: missing dual-noise flags and insufficient num_frames. Target hardware: 24GB (RTX 4090/3090) with A100 scaling notes.
