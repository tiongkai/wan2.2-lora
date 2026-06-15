---
title: Long-Video Generation with Wan 2.2 (SVI, MoE sigmas, FLF2V stitching, keyframes)
type: synthesis
tags: [long-video, svi, stable-video-infinity, flf2v, sigmas, moe, keyframes, qwen, comfyui]
created: 2026-06-15
updated: 2026-06-15
---

# Long-Video Generation with Wan 2.2

How to push past the ~5 s single-clip limit toward 15–60 s+ video. Synthesized from a
deep-research pass (22 sources, 13 claims that survived 3-vote adversarial verification).
Confidence is marked per finding; **unverified** = sourced but verification didn't confirm,
**refuted** = failed verification. Time-sensitive — landscape as of **mid-June 2026**.

## TL;DR

- The current best long-video method is **SVI (Stable Video Infinity)** — an autoregressive
  approach that fine-tunes the model to **correct its own drift**, so quality doesn't collapse
  over long durations the way naive extend-and-stitch does. Use the **SVI 2.0 LoRAs** in
  [[comfyui]] (Kijai [WanVideoWrapper] or native).
- Wan 2.2 is **MoE** (high-noise + low-noise experts). Getting the **sigma boundary / shift**
  right is what the Reddit "just calculate your sigmas" advice is about — and there's a node
  (**ComfyUI-WanMoEScheduler**) that auto-computes it.
- The **extend → extract-last-frame → FLF2V → stitch** chain works but is the *lower*-quality
  fallback; SVI's error-recycling is specifically designed to beat it.

## 1. SVI — Stable Video Infinity (confidence: high)

- **What it is:** a method to generate **unlimited-duration** video with high temporal
  coherence and *controllable scene transitions* (not just looping motion). [arxiv-2510.09212]
  [svi-github]
- **How it works — "Error-Recycling Fine-Tuning":** it feeds the Diffusion Transformer's own
  self-generated errors back as supervision, so the model learns to **identify and correct its
  own drift** during autoregressive generation. [svi-github] [arxiv-2510.09212]
- **Why it beats extend-and-stitch:** naive autoregressive generation has a train/test mismatch
  — trained on *clean* frames but at inference it conditions on its own *error-prone* outputs.
  That gap is the root cause of long-video quality decay; SVI closes it via closed-loop
  feedback. [svi-github] [arxiv-2510.09212]
- **SVI 2.0 for Wan:** trained **separately for Wan 2.1 and Wan 2.2**. The **Wan 2.2 version
  uses 1 motion frame only** (more motion frames *reduced* dynamics); Wan 2.1 supports 5 or 1.
  Kijai published fp16 LoRA versions: `Kijai/WanVideo_comfy/.../LoRAs/Stable-Video-Infinity/v2.0`
  on HuggingFace. [kijai-1718]
- **Cross-clip consistency = first-frame padding.** SVI pads ~80 frames of the first frame as
  anchors ("ID memory") to hold identity across clips. **Removing padding → more dynamics but
  significantly less consistency** — a direct dynamics↔consistency tradeoff. The Kijai impl
  uses 5 start images + padding-with-reference but a single mask. (Caveat: padding *too* many
  frames can confuse the LightX2V distill LoRA.) [kijai-1718]

### Native ComfyUI vs Kijai WanVideoWrapper
Both implement SVI. Kijai's [WanVideoWrapper] is where the SVI 2.0 LoRAs and the active
tuning discussion live [kijai-1718]; the native path uses standard ComfyUI Wan nodes. The
research did **not** produce a verified head-to-head quality claim — treat "native vs Kijai"
as a wash and pick by which workflow you already run. (unverified)

## 2. Sigmas / scheduler for Wan 2.2 MoE (confidence: high)

- Wan 2.2 14B is **MoE**: a **high-noise expert** and a **low-noise expert** in separate files
  (`wan2.2_*_high_noise_14B`, `wan2.2_*_low_noise_14B`), split by denoising timestep. [comfy-wan22]
- **The boundary matters.** Recommended **sigma boundary: 0.90 for I2V, 0.875 for T2V**
  (0.875 is the default starting point). This is where sampling hands off from the high-noise
  to the low-noise expert. [wanmoescheduler]
- **You don't have to hand-calc it:** **`ComfyUI-WanMoEScheduler`** [wanmoescheduler] auto-computes
  the **`shift`** so the high/low stages align to the target sigma boundary — this is exactly the
  "sigmas are easy, just calculate them" advice from Reddit, automated.
- **WanMoEScheduler defaults:** 4 high-noise steps + 4 low-noise steps, boundary 0.875, shift
  search precision 0.01, denoise 1.0. [wanmoescheduler]
- Note our **own trained LoRAs** already use a boundary consistent with this: high-noise expert
  trained on timesteps 900–1000, low-noise on 0–900 (boundary ≈ 0.90). See [[dual-noise-architecture]].
- The **official native docs give no numeric** steps/CFG/sigma/shift/fps — only a `length`
  param for frame count; numbers come from the community. [comfy-wan22] (medium — one source)

## 3. Extend-and-stitch chain: FLF2V (confidence: mixed)

- **Native FLF2V node = `WanFirstLastFrameToVideo`**, using the same model locations as the I2V
  workflow. [comfy-wan22] [comfy-flf] This is the node to chain segments: take a clip → extract
  its last frame → feed as the first frame (+ a target last frame) of the next segment.
- **Community node pack for the full chain (UNVERIFIED — sourced but not confirmed):**
  `ComfyUI-Wan-SVI2Pro-FLF` [svi2pro-flf] reportedly combines SVI2 motion continuity with Wan 2.2
  FLF control, using an anchor/prev/end latent structure (anchor = current segment, prev = motion
  tail from prior segment, end = locked target), a `WanCutLastSlot` node (Wan latents have
  **temporal stride 4** — one latent slot = 4 frames), and an "Image Batch Extend With Overlap"
  node with an overlap slider to smooth seams. Suggested **81–101 frames per segment**. Treat as
  a promising lead to test, not gospel — verification didn't confirm these specifics.
- **Seam/drift pitfalls:** the whole reason SVI exists is that stitched segments accumulate
  color/motion drift. If you stitch manually, expect to fight drift at the joins; overlap-blend
  the boundary frames and keep an anchor frame for identity.

> **REFUTED — do not believe:** that SVI scales "to infinite duration with **no additional
> inference cost**" (0-3 against) and the specific "1025 frames via 81-frame sliding window with
> 16-frame overlap" wrapper claim (0-3). Longer = more compute; budget accordingly. [kijai-wrapper]

## 4. Keyframe generation for consistency (confidence: unverified — leads only)

The plan of "generate consistent keyframes, then drive FLF2V between them" is sound, and the
research surfaced the actual tools behind the Reddit "insubject lora" mention — but none produced
a *verified* claim, so these are **leads to try**, not confirmed best practice:
- **`peteromallet/Qwen-Image-Edit-InSubject`** [insubject] — the literal "insubject lora": a
  Qwen-Image-Edit LoRA for keeping the **same subject** across generated frames.
- **`peteromallet/Qwen-Image-Edit-InScene`** [inscene] — same-scene consistency.
- **`lovis93/next-scene-qwen-image-lora-2509`** [nextscene] — a "next scene" Qwen-Image LoRA for
  coherent scene progression.
- Workflow idea: Qwen-Image(-Edit) + one of these LoRAs to mint consistent keyframes →
  `WanFirstLastFrameToVideo` to animate between consecutive keyframes. [qwen-wan-coherence]

## 5. Best practices, VRAM, limits

- **Prefer SVI over manual stitching** when you want one continuous long shot — it's purpose-built
  against the drift that kills stitched clips. Use manual FLF2V stitching when you need explicit
  control of each segment's end state (e.g. scripted keyframes). (high / synthesis)
- **Tune the LightX2V distill-LoRA scale per expert** (high) — it strongly affects SVI output:
  default 1.0 gives worse dynamics/text-following and makes the reference frame reappear.
  Recommended: **high-noise: LightX2V ≈ 0.5–0.6 + SVI 1.0; low-noise: LightX2V 1.0 + SVI 1.0.**
  [kijai-1718]
- **On 24 GB (our 2× A5500):** the research didn't yield a verified VRAM number for SVI, but our
  measured constraints carry over — see [[dual-a5500-run-findings]]: 14B I2V at 81 frames needs
  ComfyUI `--lowvram` and runs ~15 min/clip; activation memory (frame count) is the binding limit,
  not weights. Expect SVI long runs to be slow on 24 GB. (medium — extrapolated)
- **Realistic limit:** "infinite" is marketing; quality and compute degrade with length. SVI
  pushes the usable ceiling well past single-clip but is not free. (high)

## Caveats
- Verification was partial — several plausible, well-sourced claims (the SVI2Pro-FLF node graph,
  the Qwen keyframe LoRAs) landed **unverified** because the verify phase erred on them, not
  because they're wrong. Test them directly before trusting.
- Fast-moving area (SVI 2.0 / "2.0 Pro" released late 2025–early 2026). Re-check repos before relying.

## Sources
Primary: [svi-github] https://github.com/vita-epfl/Stable-Video-Infinity ·
[arxiv-2510.09212] https://arxiv.org/pdf/2510.09212 ·
[kijai-1718] https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1718 ·
[kijai-wrapper] https://github.com/kijai/ComfyUI-WanVideoWrapper ·
[comfy-wan22] https://docs.comfy.org/tutorials/video/wan/wan2_2 ·
[comfy-flf] https://docs.comfy.org/tutorials/video/wan/wan-flf ·
[wanmoescheduler] https://github.com/cmeka/ComfyUI-WanMoEScheduler ·
[svi2pro-flf] https://github.com/Well-Made/ComfyUI-Wan-SVI2Pro-FLF ·
[insubject] https://huggingface.co/peteromallet/Qwen-Image-Edit-InSubject ·
[inscene] https://huggingface.co/peteromallet/Qwen-Image-Edit-InScene ·
[nextscene] https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509 ·
[qwen-wan-coherence] https://www.runcomfy.com/comfyui-workflows/create-coherent-scenes-qwen-image-edit-wan-2-2-in-comfyui-cinematic-coherence-workflow

## See also
[[dual-noise-architecture]] · [[dual-a5500-run-findings]] · [[comfyui]] · [[resolution-guide]] · [[wan-2-2]]
