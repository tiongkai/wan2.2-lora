# Wiki Log

## [2026-05-12] ingest | Initial wiki creation

Built the initial wiki from web research covering 15+ sources. Ingested:

- 10 source summaries (WaveSpeed, Civitai, Apatero x3, StableDiffusionTutorials, AMD ROCm, Ostris YouTube, musubi-tuner discussions, ArXiv paper, YouTube collection)
- 8 concept pages (LoRA, dataset prep, captioning, training params, dual-noise architecture, hardware, resolution, overfitting/troubleshooting)
- 6 entity pages (Wan 2.2, Wan 2.1, musubi-tuner, AI Toolkit, diffusion-pipe, ComfyUI)
- 2 synthesis pages (training tools comparison, Wan 2.1 vs 2.2)
- 1 overview page

Sources searched: Google web search (multiple queries), YouTube search. Key domains: wavespeed.ai, civitai.com, apatero.com, stablediffusiontutorials.com, rocm.blogs.amd.com, github.com (kohya-ss/musubi-tuner, ostris/ai-toolkit), medium.com, huggingface.co, arxiv.org, youtube.com.

User-requested source: https://www.youtube.com/watch?v=2d6A_l8c_x8 (Ostris Wan 2.2 I2V 14B LoRA training)

## [2026-05-12] synthesis | Plan critique — violence detection LoRA configs

Reviewed `/Users/htx/Desktop/Projects/wan2.2-lora-1try/plan.md` against the wiki. Found 12 issues (2 critical, 7 significant, 3 minor). Produced revised configs in `wiki/synthesis/plan-critique-revised-configs.md`. Critical fixes: missing dual-noise flags and insufficient num_frames. Target hardware: 24GB (RTX 4090/3090) with A100 scaling notes.
