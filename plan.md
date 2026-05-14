# Wan2.2 Synthetic Violence Detection Dataset Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that trains category-specific LoRAs on Wan2.2 and generates labeled synthetic video datasets for violence-detection model training (fighting, vandalism, stabbing, shooting, and related threat scenarios).

**Architecture:** Each threat category gets its own LoRA trained on curated real-world reference clips; a batch generation pipeline then uses those LoRAs with varied scene prompts to produce thousands of synthetic labeled clips; a post-processing stage packages the output into ML-ready dataset formats (YOLO, classification CSV, COCO-style JSON).

**Tech Stack:** Python 3.11, musubi-tuner (kohya-ss), ffmpeg-python, scenedetect, Ollama + Qwen2.5-VL 7B multi-frame (captioning), TensorBoard + W&B (monitoring), ComfyUI (inference/generation), pandas, PyYAML, tqdm, pytest

**Hardware Profiles:**
- **24GB (RTX 3090/4090/5090):** 512×768, 33 frames, batch 1, FP8 quantization — primary target
- **80GB (A100/H100):** 720×1280, 57–81 frames, batch 2, optional quantization — future scaling

---

## Project Structure

```
wan2.2-lora/
├── configs/                      # per-category training YAML configs (ai-toolkit)
│   ├── fighting_lora.yaml
│   ├── vandalism_lora.yaml
│   ├── stabbing_lora.yaml
│   └── shooting_lora.yaml
├── datasets/                     # raw and processed training data
│   ├── raw/                      # downloaded source clips, per category
│   │   ├── fighting/
│   │   ├── vandalism/
│   │   ├── stabbing/
│   │   └── shooting/
│   └── processed/                # trimmed, resized, captioned — training-ready
│       ├── fighting/
│       ├── vandalism/
│       ├── stabbing/
│       └── shooting/
├── loras/                        # trained LoRA output weights
│   ├── fighting/
│   ├── vandalism/
│   ├── stabbing/
│   └── shooting/
├── generated/                    # synthetic videos produced for the final dataset
│   ├── clips/                    # raw generated .mp4 files
│   └── annotations/              # label files (CSV, JSON, YOLO .txt)
├── scripts/
│   ├── preprocess.py             # clip trimming, resize, scene-cut rejection
│   ├── caption.py                # VLM-based auto-captioning + trigger injection
│   ├── train_all.sh              # orchestrates ai-toolkit across all categories
│   ├── generate.py               # batch generation via ComfyUI API
│   ├── annotate.py               # builds annotation files from generation metadata
│   └── validate_dataset.py       # quality checks on generated clips
├── tests/
│   ├── test_preprocess.py
│   ├── test_caption.py
│   ├── test_annotate.py
│   └── test_validate_dataset.py
├── requirements.txt
└── plan.md                       # symlink / copy of this file for root visibility
```

---

## Category Definitions & Trigger Words

| Category | Trigger Word | Description for Training |
|---|---|---|
| Fighting | `fght99` | Two or more persons engaged in physical altercation: punching, kicking, grappling |
| Vandalism | `vndl77` | Person deliberately damaging property: spray paint, breaking glass, scratching |
| Stabbing | `stbb44` | Armed assault with bladed weapon: knife drawn, thrusting motion, close-range contact |
| Shooting | `shtn22` | Firearm-related: handgun or rifle draw, aiming posture, discharge flash |

---

## Task 1: Environment Setup & Dependency Install

**Files:**
- Create: `requirements.txt`
- Create: `scripts/setup.sh`

- [ ] **Step 1: Write requirements.txt**

```text
ffmpeg-python==0.2.0
scenedetect[opencv]==0.6.4
pyyaml==6.0.2
pandas==2.2.3
tqdm==4.67.1
requests==2.32.3
pytest==8.3.5
wandb==0.19.10
tensorboard==2.19.0
huggingface-hub==0.30.2
pillow==11.2.1
opencv-python-headless==4.11.0.86
```

- [ ] **Step 2: Install dependencies**

```bash
cd /Users/htx/Desktop/Projects/wan2.2-lora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 3: Clone musubi-tuner alongside this project**

```bash
cd /home/lenovo5/TiongKai/Greenfield
git clone https://github.com/kohya-ss/musubi-tuner
cd musubi-tuner
uv pip install -r requirements.txt
```

- [ ] **Step 4: Download Wan2.2 T2V model weights**

musubi-tuner needs the native safetensors files (not diffusers format). Download from the original Wan 2.2 release:

```bash
mkdir -p models/wan2.2-t2v
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
  --local-dir models/wan2.2-t2v \
  --ignore-patterns "*.bin"
```

Expected files musubi-tuner needs:
- `models/wan2.2-t2v/wan2.2_t2v_high_noise_14B_fp16.safetensors` — high-noise transformer
- `models/wan2.2-t2v/wan2.2_t2v_low_noise_14B_fp16.safetensors` — low-noise transformer
- `models/wan2.2-t2v/wan_2.1_vae.safetensors` — VAE (shared with Wan 2.1)
- `models/wan2.2-t2v/models_t5_umt5-xxl-enc-bf16.pth` — T5 text encoder

> **Note:** Exact filenames depend on the HuggingFace repo layout — verify paths after download and update `scripts/train_all.sh` accordingly.

- [ ] **Step 5: Create directory scaffold**

```bash
cd /Users/htx/Desktop/Projects/wan2.2-lora
mkdir -p datasets/raw/{fighting,vandalism,stabbing,shooting}
mkdir -p datasets/processed/{fighting,vandalism,stabbing,shooting}
mkdir -p loras/{fighting,vandalism,stabbing,shooting}
mkdir -p generated/{clips,annotations}
mkdir -p configs logs
```

- [ ] **Step 6: Commit scaffold**

```bash
git init
git add requirements.txt
git commit -m "feat: initial project scaffold and requirements"
```

---

## Task 2: Dataset Acquisition

**Files:** None — operational task, no code.

Collect 15–30 clips per category into `datasets/raw/<category>/`. Prefer clips already at surveillance camera angle, 720p+, 2–5 seconds, no hard cuts. The sources below are ranked by quality for LoRA training.

### Primary Sources (Use These First)

#### [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/) — All 4 categories
The gold standard: **1,900 real CCTV surveillance videos** across 13 anomaly classes. Directly covers Fighting, Shooting, Assault, Vandalism, Robbery. Already filmed from surveillance angles at low resolution — exactly the aesthetic the LoRA should learn.

- Direct download (official): `https://webpages.uncc.edu/cchen62/dataset.html`
- [Kaggle mirror](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) — 8-class subset, easiest to download
- [HuggingFace mirror](https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset) — frame-extracted version

#### [AIRT Lab Dataset](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) — Fighting, Stabbing, Shooting
**350 clips at 1080p 30fps**, actors simulating punches, kicks, stabbing, and gunshots. No copyright issues (free for research). Consistent controlled footage — ideal for LoRA training since every clip has clean action with no scene cuts. Best source for stabbing and shooting specifically.

#### [XD-Violence](https://roc-ng.github.io/XD-Violence/) — Fighting, Shooting
**4,754 videos, 217 hours** across Abuse, Fighting, Riot, Shooting, Explosion. Multi-scene variety: surveillance, handheld, dashcam. Good for adding environmental diversity to the dataset.

#### [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) — Fighting
2,000 real surveillance clips already trimmed to ≤5 seconds at 30fps — no preprocessing needed for duration. Email authors to request access.

### Supplementary Sources

| Category | Source | Notes |
|---|---|---|
| **Fighting** | Pexels search "brawl", "fight" | Free commercial license |
| **Fighting** | YouTube MMA/boxing, CC filter | Filter by Creative Commons license |
| **Vandalism** | Pexels search "graffiti", "vandalism" | Limited but some exist |
| **Vandalism** | News b-roll (BBC, Reuters) | Check license per clip |
| **Stabbing** | YouTube stage combat training videos | Abundant, royalty-free, weapon props |
| **Stabbing** | Blender renders with prop knives | Perfect consistency, zero rights issues |
| **Shooting** | [VSD dataset](https://www.interdigital.com/data_sets/violent-scenes-dataset) timestamps + legally obtained films | Annotations only; you supply films |
| **Shooting** | Blender/Unreal Engine renders | Best option for clean firearm motion |

### Per-Category Minimum Targets

| Category | Min Clips | Primary Source |
|---|---|---|
| Fighting | 20 | UCF-Crime + RWF-2000 |
| Vandalism | 15 | UCF-Crime + Pexels |
| Stabbing | 15 | AIRT Lab + stage combat YouTube |
| Shooting | 15 | UCF-Crime + AIRT Lab |

### Regularization Clips (Important for Wan 2.2)

Wan 2.2 has an opinionated aesthetic — it pushes toward crisp portraits, cinematic light, and shallow depth of field. Without regularization clips, the LoRA will override the surveillance-camera look you want.

**Add 3–5 generic surveillance clips per category** (normal activity, no violence):
- People walking through a parking lot (normal)
- Empty hallway from a corridor camera
- Street scene from an elevated CCTV angle
- Store interior with customers browsing

Place these in the same `datasets/processed/<category>/` directory. Caption them **without** the trigger word:
```
security camera footage, person walking through parking lot, overhead CCTV angle, night lighting
```

This teaches the model that the trigger word specifically means the violent action, not the camera angle or environment. The regularization clips prevent Wan 2.2 from baking in its default bokeh/portrait aesthetic.

### Download Steps

- [ ] **Step 1: Download UCF-Crime from Kaggle**

```bash
pip install kaggle
kaggle datasets download -d odins0n/ucf-crime-dataset -p datasets/raw/ucf_crime
unzip datasets/raw/ucf_crime/ucf-crime-dataset.zip -d datasets/raw/ucf_crime/
```

Expected: directories per class including `Fighting/`, `Shooting/`, `Vandalism/`, `Assault/`.

- [ ] **Step 2: Copy UCF-Crime clips into category folders**

```bash
cp datasets/raw/ucf_crime/Fighting/*.mp4    datasets/raw/fighting/
cp datasets/raw/ucf_crime/Shooting/*.mp4    datasets/raw/shooting/
cp datasets/raw/ucf_crime/Vandalism/*.mp4   datasets/raw/vandalism/
cp datasets/raw/ucf_crime/Assault/*.mp4     datasets/raw/stabbing/   # supplement until AIRT clips arrive
```

- [ ] **Step 3: Download AIRT Lab dataset**

```bash
# Clone the repo — clips are hosted via the GitHub release assets
git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos \
  datasets/raw/airt_lab
# Violent clips cover: punches, kicks, stabbing, gunshots
cp datasets/raw/airt_lab/violent/*.mp4  datasets/raw/fighting/
# Sort stabbing and shooting clips manually after inspecting filenames
```

- [ ] **Step 4: Verify clip counts**

```bash
for cat in fighting vandalism stabbing shooting; do
  count=$(ls datasets/raw/$cat/*.mp4 2>/dev/null | wc -l)
  echo "$cat: $count clips"
done
```

Expected: each category has ≥15 clips. If any fall short, supplement from the table above before proceeding.

- [ ] **Step 5: Commit manifest (no video files — add to .gitignore)**

```bash
echo "datasets/raw/**/*.mp4" >> .gitignore
echo "datasets/raw/**/*.mov" >> .gitignore
echo "datasets/processed/**/*.mp4" >> .gitignore
echo "generated/**/*.mp4" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore video files, datasets tracked by manifest only"
```

---

## Task 3: Video Preprocessing Script

**Files:**
- Create: `scripts/preprocess.py`
- Create: `tests/test_preprocess.py`

The preprocessor accepts a source directory of raw clips and outputs trimmed, resized, scene-cut-validated clips into the processed directory. It rejects clips with hard cuts, normalizes to 512×768 (portrait) or 768×512 (landscape) for 24GB training, and enforces 2–5 second duration at 24fps.

> **A100 override:** Change `width=768, height=512` to `width=1280, height=720` for full 720p training.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import pytest
from pathlib import Path
import tempfile, shutil, subprocess

# Helper: create a 6-second solid-color test video
def make_test_video(path: Path, duration: int = 6):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=blue:s=1280x720:r=24:d={duration}",
        "-c:v", "libx264", str(path)
    ], check=True, capture_output=True)

def test_trim_clips_to_max_five_seconds(tmp_path):
    from scripts.preprocess import trim_clip
    src = tmp_path / "input.mp4"
    dst = tmp_path / "output.mp4"
    make_test_video(src, duration=6)
    trim_clip(src, dst, max_seconds=5)
    # probe output duration
    import subprocess, json
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(dst)
    ], capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info["streams"][0]["duration"])
    assert duration <= 5.1  # 0.1s tolerance for keyframe rounding

def test_reject_short_clip(tmp_path):
    from scripts.preprocess import is_valid_clip
    src = tmp_path / "short.mp4"
    make_test_video(src, duration=1)
    assert is_valid_clip(src, min_seconds=2) is False

def test_accept_valid_clip(tmp_path):
    from scripts.preprocess import is_valid_clip
    src = tmp_path / "good.mp4"
    make_test_video(src, duration=3)
    assert is_valid_clip(src, min_seconds=2) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/htx/Desktop/Projects/wan2.2-lora
source .venv/bin/activate
pytest tests/test_preprocess.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.preprocess'`

- [ ] **Step 3: Implement preprocess.py**

```python
# scripts/preprocess.py
import subprocess, json, shutil
from pathlib import Path
from tqdm import tqdm


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True
    )
    streams = json.loads(result.stdout).get("streams", [])
    return float(streams[0]["duration"]) if streams else 0.0


def is_valid_clip(path: Path, min_seconds: float = 2.0) -> bool:
    try:
        return probe_duration(path) >= min_seconds
    except Exception:
        return False


def trim_clip(src: Path, dst: Path, max_seconds: float = 5.0, fps: int = 24,
              width: int = 768, height: int = 512):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(max_seconds),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
        "-c:v", "libx264", "-crf", "18", "-an",
        str(dst)
    ], check=True, capture_output=True)


def has_scene_cut(path: Path, threshold: float = 27.0) -> bool:
    """Return True if clip contains a hard scene cut (disqualifies it)."""
    result = subprocess.run([
        "python", "-m", "scenedetect",
        "-i", str(path),
        "detect-adaptive", f"--threshold={threshold}",
        "list-scenes"
    ], capture_output=True, text=True)
    lines = [l for l in result.stdout.splitlines() if "Scene" in l and "1 " not in l]
    return len(lines) > 0


def process_category(raw_dir: Path, out_dir: Path,
                     min_seconds: float = 2.0, max_seconds: float = 5.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.mov"))
    accepted, rejected = 0, 0
    for clip in tqdm(clips, desc=f"Processing {raw_dir.name}"):
        if not is_valid_clip(clip, min_seconds):
            rejected += 1
            continue
        if has_scene_cut(clip):
            rejected += 1
            continue
        dst = out_dir / clip.name
        trim_clip(clip, dst, max_seconds=max_seconds)
        accepted += 1
    print(f"{raw_dir.name}: {accepted} accepted, {rejected} rejected")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=["fighting", "vandalism", "stabbing", "shooting"])
    args = p.parse_args()
    base = Path(__file__).parent.parent
    process_category(
        raw_dir=base / "datasets" / "raw" / args.category,
        out_dir=base / "datasets" / "processed" / args.category,
    )
```

- [ ] **Step 4: Add `__init__.py` so tests import correctly**

```bash
touch scripts/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/preprocess.py scripts/__init__.py tests/test_preprocess.py
git commit -m "feat: clip preprocessing — trim, validate, scene-cut rejection"
```

---

## Task 4: Auto-Captioning Pipeline

**Files:**
- Create: `scripts/caption.py`
- Create: `tests/test_caption.py`

Captions each processed clip by extracting **5 keyframes** across the clip duration (at 10%, 25%, 50%, 75%, 90%), sending all frames to Qwen2.5-VL 7B via Ollama's multi-image API, and writing a `.txt` sidecar file with the trigger word prepended. Multi-frame captioning captures the **temporal action sequence** (approach → action → aftermath), which is critical for motion LoRAs. Fallback: GPT-5.4 for re-captioning if local VLM quality is insufficient.

- [ ] **Step 1: Install and pull Ollama model**

```bash
# Install Ollama if not present: https://ollama.com
ollama pull qwen2.5vl:7b
```

Expected: model downloads (~5GB).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_caption.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_multi_keyframes_creates_jpgs(tmp_path):
    import subprocess
    # Create a 3-second test video
    src = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=red:s=640x480:r=24:d=3",
        "-c:v", "libx264", str(src)
    ], check=True, capture_output=True)
    from scripts.caption import extract_multi_keyframes
    frames = extract_multi_keyframes(src, n_frames=5, out_dir=tmp_path)
    assert len(frames) == 5
    for f in frames:
        assert f.exists()
        assert f.suffix == ".jpg"


def test_build_caption_injects_trigger():
    from scripts.caption import build_caption
    raw = "Two people fighting on the street."
    result = build_caption(raw, trigger="fght99")
    assert result.startswith("fght99,")
    assert "Two people fighting" in result


def test_caption_file_written(tmp_path):
    from scripts.caption import write_caption_file
    clip = tmp_path / "test.mp4"
    clip.touch()
    write_caption_file(clip, "fght99, two people fighting in a parking lot")
    txt = tmp_path / "test.txt"
    assert txt.exists()
    assert txt.read_text().startswith("fght99,")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_caption.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.caption'`

- [ ] **Step 4: Implement caption.py**

```python
# scripts/caption.py
import subprocess, base64, json, textwrap
from pathlib import Path
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"

CAPTION_SYSTEM = textwrap.dedent("""
    You are a dataset annotation assistant for a computer vision security system.
    You will receive 5 sequential frames from a short security camera video clip.
    Describe the complete action sequence from start to end for training a video generation model.
    Include: subjects present, their movements and actions over time, environment, lighting, camera angle.
    Start with the provided trigger word followed by a comma.
    Describe motion and temporal progression, not just static appearance.
    Avoid filler phrases. Keep it under 75 words. Use plain descriptive English.
""").strip()

CATEGORY_CONTEXT = {
    "fighting":  "The clip shows a physical altercation or assault.",
    "vandalism": "The clip shows property damage or defacement.",
    "stabbing":  "The clip shows an armed blade attack.",
    "shooting":  "The clip shows a firearm being used or brandished.",
}


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True
    )
    streams = json.loads(result.stdout).get("streams", [])
    return float(streams[0]["duration"]) if streams else 0.0


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


def _image_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def query_vlm(frames: list[Path], trigger: str, category: str, clip_duration: float) -> str:
    prompt = (
        f"These 5 frames are sampled sequentially from a {clip_duration:.1f}s security camera video. "
        f"Trigger word: '{trigger}'. "
        f"Context: {CATEGORY_CONTEXT.get(category, '')} "
        "Describe the complete action sequence starting with the trigger word."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": CAPTION_SYSTEM,
        "images": [_image_to_b64(f) for f in frames],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def build_caption(raw_text: str, trigger: str) -> str:
    text = raw_text.strip()
    if not text.startswith(trigger):
        text = f"{trigger}, {text}"
    return text


def write_caption_file(clip: Path, caption: str):
    txt = clip.with_suffix(".txt")
    txt.write_text(caption)


def caption_directory(processed_dir: Path, trigger: str, category: str):
    clips = list(processed_dir.glob("*.mp4"))
    for clip in tqdm(clips, desc=f"Captioning {category}"):
        txt = clip.with_suffix(".txt")
        if txt.exists():
            continue  # skip already captioned
        frames = extract_multi_keyframes(clip, n_frames=5)
        try:
            duration = probe_duration(clip)
            raw = query_vlm(frames, trigger=trigger, category=category, clip_duration=duration)
            caption = build_caption(raw, trigger=trigger)
        except Exception as e:
            print(f"  WARN: VLM failed for {clip.name}: {e}")
            caption = f"{trigger}, security camera footage, threat event, outdoor environment"
        write_caption_file(clip, caption)
        for f in frames:
            f.unlink(missing_ok=True)  # clean up temp frames


TRIGGERS = {
    "fighting":  "fght99",
    "vandalism": "vndl77",
    "stabbing":  "stbb44",
    "shooting":  "shtn22",
}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=list(TRIGGERS.keys()))
    args = p.parse_args()
    base = Path(__file__).parent.parent
    caption_directory(
        processed_dir=base / "datasets" / "processed" / args.category,
        trigger=TRIGGERS[args.category],
        category=args.category,
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_caption.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/caption.py tests/test_caption.py
git commit -m "feat: VLM-based auto-captioning with trigger word injection"
```

---

## Task 5: Per-Category Training Configs

**Files:**
- Create: `configs/fighting_dataset.toml`
- Create: `configs/vandalism_dataset.toml`
- Create: `configs/stabbing_dataset.toml`
- Create: `configs/shooting_dataset.toml`
- Create: `scripts/train_all.sh`

musubi-tuner separates dataset config (TOML) from training arguments (CLI flags). Each category gets one shared TOML used by both experts. `train_all.sh` runs the high-noise and low-noise experts **in parallel** on GPU 0 and GPU 1 respectively, halving total wall-clock time. Pre-caching latents and text encoder outputs runs once per category before training starts.

> **A100 override:** Change `resolution = [512, 768]` → `[720, 1280]` and `target_frames = [33]` → `[57]` in each dataset TOML.

- [ ] **Step 1: Write fighting dataset config (template for all)**

```toml
# configs/fighting_dataset.toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "datasets/processed/fighting"
cache_directory = "datasets/processed/fighting/cache"
frame_extraction = "uniform"
source_fps = 24.0
target_frames = [33]
max_frames = 33
resolution = [512, 768]
caption_dropout_rate = 0.05
```

- [ ] **Step 2: Write vandalism dataset config**

```toml
# configs/vandalism_dataset.toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "datasets/processed/vandalism"
cache_directory = "datasets/processed/vandalism/cache"
frame_extraction = "uniform"
source_fps = 24.0
target_frames = [33]
max_frames = 33
resolution = [512, 768]
caption_dropout_rate = 0.05
```

- [ ] **Step 3: Write stabbing dataset config**

```toml
# configs/stabbing_dataset.toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "datasets/processed/stabbing"
cache_directory = "datasets/processed/stabbing/cache"
frame_extraction = "uniform"
source_fps = 24.0
target_frames = [33]
max_frames = 33
resolution = [512, 768]
caption_dropout_rate = 0.05
```

- [ ] **Step 4: Write shooting dataset config**

```toml
# configs/shooting_dataset.toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "datasets/processed/shooting"
cache_directory = "datasets/processed/shooting/cache"
frame_extraction = "uniform"
source_fps = 24.0
target_frames = [33]
max_frames = 33
resolution = [512, 768]
caption_dropout_rate = 0.05
```

- [ ] **Step 5: Write train_all.sh**

```bash
#!/usr/bin/env bash
# scripts/train_all.sh
# Trains all four category LoRAs using musubi-tuner.
# High-noise (GPU 0) and low-noise (GPU 1) experts run simultaneously per category.
# Run from project root: bash scripts/train_all.sh
set -e

MUSUBI="$(cd .. && pwd)/musubi-tuner"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Model paths — verify these match your download after Task 1 Step 4
DIT_HIGH="$BASE_DIR/models/wan2.2-t2v/wan2.2_t2v_high_noise_14B_fp16.safetensors"
DIT_LOW="$BASE_DIR/models/wan2.2-t2v/wan2.2_t2v_low_noise_14B_fp16.safetensors"
VAE="$BASE_DIR/models/wan2.2-t2v/wan_2.1_vae.safetensors"
T5="$BASE_DIR/models/wan2.2-t2v/models_t5_umt5-xxl-enc-bf16.pth"

COMMON_ARGS=(
  --task t2v-A14B
  --vae "$VAE"
  --sdpa --mixed_precision bf16 --fp8_base
  --optimizer_type adamw8bit
  --gradient_checkpointing
  --network_module networks.lora_wan --network_dim 32 --network_alpha 32
  --timestep_sampling shift --discrete_flow_shift 3.0
  --max_train_steps 3000 --save_every_n_steps 300 --seed 42
)

train_category() {
  local category=$1
  local steps=${2:-3000}
  local config="$BASE_DIR/configs/${category}_dataset.toml"
  local out_dir="$BASE_DIR/loras/$category"
  local name="${category}_lora_r32"

  echo "===== Pre-caching: $category ====="
  cd "$MUSUBI"
  uv run python cache_latents.py \
    --task t2v-A14B --dit "$DIT_HIGH" --vae "$VAE" \
    --dataset_config "$config"
  uv run python cache_text_encoder_outputs.py \
    --task t2v-A14B --t5xxl "$T5" \
    --dataset_config "$config"

  echo "===== Training $category — high-noise (GPU 0) + low-noise (GPU 1) ====="
  CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/wan_train_network.py \
    "${COMMON_ARGS[@]}" \
    --dit "$DIT_HIGH" \
    --dataset_config "$config" \
    --learning_rate 2e-4 \
    --min_timestep 900 --max_timestep 1000 \
    --max_train_steps "$steps" \
    --output_dir "$out_dir" --output_name "${name}_high" \
    --log_with tensorboard --logging_dir "$BASE_DIR/logs/$category" &

  CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
    --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/wan_train_network.py \
    "${COMMON_ARGS[@]}" \
    --dit "$DIT_LOW" \
    --dataset_config "$config" \
    --learning_rate 2e-5 \
    --min_timestep 0 --max_timestep 900 \
    --max_train_steps "$steps" \
    --output_dir "$out_dir" --output_name "${name}_low" \
    --log_with tensorboard --logging_dir "$BASE_DIR/logs/$category" &

  wait
  cd "$BASE_DIR"
  echo "===== Done: $category ====="
}

train_category fighting  3000
train_category vandalism 2500
train_category stabbing  3000
train_category shooting  3000

echo "All LoRAs trained."
```

- [ ] **Step 6: Make executable and commit**

```bash
chmod +x scripts/train_all.sh
git add configs/ scripts/train_all.sh
git commit -m "feat: per-category musubi-tuner dataset configs and parallel train_all script"
```

---

## Task 6: Training Execution & Monitoring

This task is operational — run training and validate outputs. No code to write.

- [ ] **Step 1: Start TensorBoard before training**

```bash
source .venv/bin/activate
tensorboard --logdir ./logs --port 6006 &
# Open http://localhost:6006 in browser
```

- [ ] **Step 2: (Optional) Configure W&B**

```bash
wandb login
# In each config, change: log_with = "wandb"
# W&B dashboard will show sample video previews at each checkpoint
```

- [ ] **Step 3: Start training (single category first)**

```bash
# From project root — trains fighting high+low noise in parallel
bash scripts/train_all.sh 2>&1 | tee logs/train_run.log &
# Or run just one category manually:
# CUDA_VISIBLE_DEVICES=0 uv run accelerate launch ... (see train_all.sh for full args)
```

- [ ] **Step 4: Checkpoint evaluation checklist (run at step 600, 1200, 1800, 2400)**

For each checkpoint, generate a test clip using ComfyUI or SwarmUI:
- Trigger word activates the correct behavior
- No training background "baked in" (try varied scene prompts)
- Temporal smoothness — no flicker between frames
- Identity consistent across the clip duration

Stop training at the best checkpoint. Common stopping point: 1800–2400 steps for 15–25 clip datasets.

- [ ] **Step 5: Verify LoRA output files exist**

```bash
ls loras/fighting/
# Expected (musubi-tuner dual-noise):
#   fighting_lora_r32_high.safetensors  — high-noise expert (layout/composition)
#   fighting_lora_r32_low.safetensors   — low-noise expert (details/refinement)
# Both files are required at inference time; load both in ComfyUI
```

- [ ] **Step 6: Repeat for remaining categories**

```bash
bash scripts/train_all.sh
```

Monitor loss curves per category in TensorBoard. Target loss: ~0.02–0.04 at convergence.

---

## Task 7: Batch Synthetic Video Generation

**Files:**
- Create: `scripts/generate.py`
- Create: `scripts/prompts/fighting.txt`
- Create: `scripts/prompts/vandalism.txt`
- Create: `scripts/prompts/stabbing.txt`
- Create: `scripts/prompts/shooting.txt`

Uses ComfyUI's HTTP API to batch-generate synthetic clips with each trained LoRA. Each prompt gets 5 generations with varied seeds to maximize diversity.

- [ ] **Step 1: Write generation prompts — fighting**

```text
# scripts/prompts/fighting.txt
# Format: one prompt per line; generator will vary seeds and LoRA weights
fght99, two men brawling outside a nightclub, surveillance camera angle, nighttime, wet pavement
fght99, group fight in a subway station, overhead CCTV perspective, fluorescent lighting
fght99, physical altercation in a convenience store, interior security camera, wide angle
fght99, two people wrestling on a sidewalk, street-level dash-cam perspective, daytime
fght99, aggressive confrontation in a parking garage, fisheye security lens, harsh shadows
fght99, brawl outside a sports venue, crowd in background, handheld camera, afternoon
fght99, two individuals fighting in an apartment hallway, corridor camera, close range
fght99, road rage physical fight, roadside camera angle, bright daylight
fght99, schoolyard fight, distant surveillance angle, overcast outdoor lighting
fght99, woman defending herself from attacker in stairwell, low-angle security camera
```

- [ ] **Step 2: Write generation prompts — vandalism**

```text
# scripts/prompts/vandalism.txt
vndl77, hooded figure spray painting graffiti on a brick wall, night, security camera
vndl77, person smashing car windshield with a crowbar, parking lot, CCTV overhead
vndl77, individual keying a parked car in a shopping center, daytime, wide angle
vndl77, person kicking and breaking a glass bus shelter, street camera, evening
vndl77, two teenagers tagging a train carriage, platform camera, artificial lighting
vndl77, person throwing a rock through a shop window, high street, daytime security cam
vndl77, hooded figure slashing car tyres, underground carpark, fisheye lens
vndl77, graffiti artist defacing a road sign, rural road camera, dusk
vndl77, person breaking a park bench, CCTV angle, morning daylight
vndl77, individual setting fire to a rubbish bin, alley security camera, night
```

- [ ] **Step 3: Write generation prompts — stabbing**

```text
# scripts/prompts/stabbing.txt
stbb44, armed attacker confronting victim in narrow alley, overhead CCTV, night
stbb44, knife-draw during argument outside a pub, street security camera, evening
stbb44, person brandishing blade near ATM, wide-angle bank exterior camera, night
stbb44, stabbing incident in underground car park, fisheye security camera, poor lighting
stbb44, armed robbery attempt in convenience store, interior camera, fluorescent light
stbb44, two people arguing escalates to knife threat, pavement, daytime CCTV
stbb44, attacker approaching from behind in an alley, elevated security angle, dusk
stbb44, blade drawn in altercation outside school gates, wide angle, afternoon
stbb44, confrontation on public transport platform, overhead CCTV, evening
stbb44, knife threat near park bench, ground-level camera, natural daylight
```

- [ ] **Step 4: Write generation prompts — shooting**

```text
# scripts/prompts/shooting.txt
shtn22, person drawing handgun in a convenience store robbery, interior camera, night
shtn22, armed individual in a parking lot, overhead security camera, artificial lighting
shtn22, gunman entering a building lobby, wide-angle entrance camera, day
shtn22, drive-by incident caught on traffic camera, street level, daytime
shtn22, armed robbery in a bank, interior CCTV, fluorescent lighting, wide angle
shtn22, armed individual fleeing scene, courtyard camera, evening light
shtn22, firearm drawn during road rage, dashboard camera angle, daylight
shtn22, shooting incident in an alley, elevated CCTV, night, poor visibility
shtn22, armed person on a rooftop, distant surveillance angle, afternoon
shtn22, firearm brandished at ATM, bank exterior camera, nighttime
```

- [ ] **Step 5: Implement generate.py**

```python
# scripts/generate.py
"""
Batch synthetic video generation via ComfyUI HTTP API.
Requires ComfyUI running at localhost:8188 with Wan2.2 T2V workflow loaded.

Usage:
  python scripts/generate.py --category fighting --count 50
"""
import json, random, time, requests, argparse
from pathlib import Path
import uuid

COMFYUI_URL = "http://localhost:8188"

BASE_LORA_PATHS = {
    "fighting":  "../../wan2.2-lora/loras/fighting/fighting_lora_r32",
    "vandalism": "../../wan2.2-lora/loras/vandalism/vandalism_lora_r32",
    "stabbing":  "../../wan2.2-lora/loras/stabbing/stabbing_lora_r32",
    "shooting":  "../../wan2.2-lora/loras/shooting/shooting_lora_r32",
}

GENERATION_DEFAULTS = {
    "steps": 30,
    "cfg": 4.0,
    "lora_strength": 0.75,
    "width": 768,           # match training resolution; A100: 1280
    "height": 512,          # match training resolution; A100: 720
    "num_frames": 33,       # match training frames; A100: 57
}


WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "comfyui_wan22_t2v_workflow.json"
# ^^^ Export a working Wan 2.2 T2V + LoRA workflow from ComfyUI as API JSON.
# IMPORTANT: Wan 2.2 uses a T5-based text encoder, NOT CLIP.
# The workflow must use T5 text encoding nodes, not CLIPTextEncode.
# Node IDs below are placeholders — update them to match YOUR workflow.

# Patch field paths: map semantic fields to (node_id, field_path) in your workflow JSON.
# Update these after exporting your workflow.
PATCH_FIELDS = {
    "prompt":       ("6", ["inputs", "text"]),
    "neg_prompt":   ("7", ["inputs", "text"]),
    "seed":         ("3", ["inputs", "seed"]),
    "lora_path":    ("4", ["inputs", "lora_path"]),
    "lora_strength":("4", ["inputs", "strength"]),
    "filename":     ("9", ["inputs", "filename_prefix"]),
}


def build_workflow(prompt: str, seed: int, category: str, cfg: float = 4.0) -> dict:
    """
    Load a saved ComfyUI workflow JSON and patch variable fields.
    Export your working Wan 2.2 T2V workflow from ComfyUI (Save as API format)
    and save it as comfyui_wan22_t2v_workflow.json in the scripts/ directory.
    """
    if not WORKFLOW_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Workflow template not found at {WORKFLOW_TEMPLATE_PATH}. "
            "Export a Wan 2.2 T2V + LoRA workflow from ComfyUI in API format."
        )
    workflow = json.loads(WORKFLOW_TEMPLATE_PATH.read_text())

    lora_base = BASE_LORA_PATHS[category]
    patches = {
        "prompt": prompt,
        "neg_prompt": "blurry, low quality, watermark, text, distorted",
        "seed": seed,
        "lora_path": f"{lora_base}.safetensors",
        "filename": f"synthetic_{category}",
    }
    for key, value in patches.items():
        if key in PATCH_FIELDS:
            node_id, field_path = PATCH_FIELDS[key]
            node = workflow[node_id]
            target = node
            for part in field_path[:-1]:
                target = target[part]
            target[field_path[-1]] = value

    return workflow


def queue_prompt(workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        history = resp.json()
        if prompt_id in history:
            return True
        time.sleep(2)
    return False


def load_prompts(category: str) -> list[str]:
    p = Path(__file__).parent / "prompts" / f"{category}.txt"
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return lines


def generate_batch(category: str, count: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(category)
    metadata_rows = []

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        seed = random.randint(0, 2**32 - 1)
        workflow = build_workflow(prompt, seed=seed, category=category)
        prompt_id = queue_prompt(workflow)
        success = wait_for_completion(prompt_id)
        status = "ok" if success else "timeout"
        metadata_rows.append({
            "id": i, "category": category, "prompt": prompt,
            "seed": seed, "prompt_id": prompt_id, "status": status
        })
        print(f"[{i+1}/{count}] {status} — seed {seed}")

    # Save generation manifest
    import pandas as pd
    df = pd.DataFrame(metadata_rows)
    df.to_csv(out_dir / f"{category}_manifest.csv", index=False)
    print(f"Manifest saved: {out_dir}/{category}_manifest.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=["fighting", "vandalism", "stabbing", "shooting"])
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()
    base = Path(__file__).parent.parent
    generate_batch(args.category, args.count,
                   out_dir=base / "generated" / "clips" / args.category)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/generate.py scripts/prompts/
git commit -m "feat: batch generation script with ComfyUI API and prompt library"
```

---

## Task 8: Dataset Annotation & Export

**Files:**
- Create: `scripts/annotate.py`
- Create: `tests/test_annotate.py`

Reads generation manifests and ComfyUI output filenames to produce ML-ready annotation files: a unified CSV, per-clip YOLO-style action labels, and a COCO-compatible JSON.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_annotate.py
import pytest, json
from pathlib import Path


SAMPLE_MANIFEST = [
    {"id": 0, "category": "fighting", "prompt": "fght99, two men fighting",
     "seed": 12345, "prompt_id": "abc-1", "status": "ok"},
    {"id": 1, "category": "fighting", "prompt": "fght99, brawl in subway",
     "seed": 67890, "prompt_id": "abc-2", "status": "ok"},
]

CLASS_MAP = {"fighting": 0, "vandalism": 1, "stabbing": 2, "shooting": 3}


def test_csv_has_required_columns(tmp_path):
    import pandas as pd
    from scripts.annotate import build_annotation_csv
    df = pd.DataFrame(SAMPLE_MANIFEST)
    result = build_annotation_csv(df, class_map=CLASS_MAP)
    for col in ["filename", "category", "class_id", "prompt", "seed"]:
        assert col in result.columns


def test_coco_json_structure(tmp_path):
    import pandas as pd
    from scripts.annotate import build_coco_json
    df = pd.DataFrame(SAMPLE_MANIFEST)
    coco = build_coco_json(df, class_map=CLASS_MAP, clips_dir=tmp_path)
    assert "categories" in coco
    assert "annotations" in coco
    assert "videos" in coco
    assert len(coco["categories"]) == 4


def test_class_ids_correct(tmp_path):
    import pandas as pd
    from scripts.annotate import build_annotation_csv
    df = pd.DataFrame(SAMPLE_MANIFEST)
    result = build_annotation_csv(df, class_map=CLASS_MAP)
    assert result.iloc[0]["class_id"] == 0  # fighting = 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_annotate.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.annotate'`

- [ ] **Step 3: Implement annotate.py**

```python
# scripts/annotate.py
import json
import pandas as pd
from pathlib import Path

CLASS_MAP = {"fighting": 0, "vandalism": 1, "stabbing": 2, "shooting": 3}


def build_annotation_csv(df: pd.DataFrame, class_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["class_id"] = df["category"].map(class_map)
    df["filename"] = df.apply(
        lambda r: f"synthetic_{r['category']}_{r['id']:05d}.mp4", axis=1
    )
    return df[["filename", "category", "class_id", "prompt", "seed", "status"]]


def build_coco_json(df: pd.DataFrame, class_map: dict, clips_dir: Path) -> dict:
    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    videos, annotations = [], []
    for _, row in df.iterrows():
        vid_id = int(row["id"])
        fname = f"synthetic_{row['category']}_{vid_id:05d}.mp4"
        videos.append({"id": vid_id, "file_name": fname,
                        "category": row["category"]})
        annotations.append({
            "id": vid_id, "video_id": vid_id,
            "category_id": class_map[row["category"]],
            "prompt": row["prompt"], "seed": int(row["seed"]),
        })
    return {"categories": categories, "videos": videos, "annotations": annotations}


def export_dataset(generated_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for category in CLASS_MAP:
        manifest = generated_dir / "clips" / category / f"{category}_manifest.csv"
        if manifest.exists():
            all_dfs.append(pd.read_csv(manifest))

    if not all_dfs:
        raise FileNotFoundError("No manifest CSVs found. Run generate.py first.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[combined["status"] == "ok"]

    # CSV
    csv_df = build_annotation_csv(combined, CLASS_MAP)
    csv_df.to_csv(out_dir / "annotations.csv", index=False)
    print(f"Saved annotations.csv ({len(csv_df)} clips)")

    # COCO JSON
    coco = build_coco_json(combined, CLASS_MAP, clips_dir=generated_dir / "clips")
    (out_dir / "annotations_coco.json").write_text(json.dumps(coco, indent=2))
    print(f"Saved annotations_coco.json")

    # Class distribution summary
    print("\nClass distribution:")
    print(csv_df["category"].value_counts().to_string())


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    export_dataset(
        generated_dir=base / "generated",
        out_dir=base / "generated" / "annotations"
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_annotate.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/annotate.py tests/test_annotate.py
git commit -m "feat: annotation export — CSV, COCO JSON, class distribution"
```

---

## Task 9: Dataset Quality Validation

**Files:**
- Create: `scripts/validate_dataset.py`
- Create: `tests/test_validate_dataset.py`

Checks every generated clip for: minimum duration, valid video stream, non-black frames, and presence in the annotation CSV.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_validate_dataset.py
import subprocess, pytest
from pathlib import Path


def make_video(path, duration=3, color="blue"):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=1280x720:r=24:d={duration}",
        "-c:v", "libx264", str(path)
    ], check=True, capture_output=True)


def make_black_video(path, duration=3):
    make_video(path, duration=duration, color="black")


def test_valid_clip_passes(tmp_path):
    from scripts.validate_dataset import validate_clip
    clip = tmp_path / "good.mp4"
    make_video(clip)
    result = validate_clip(clip)
    assert result["valid"] is True


def test_black_clip_fails(tmp_path):
    from scripts.validate_dataset import validate_clip
    clip = tmp_path / "black.mp4"
    make_black_video(clip)
    result = validate_clip(clip)
    assert result["valid"] is False
    assert "black" in result["reason"].lower()
```

- [ ] **Step 2: Implement validate_dataset.py**

```python
# scripts/validate_dataset.py
import subprocess, json, cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def is_mostly_black(path: Path, sample_frames: int = 5,
                    brightness_threshold: float = 8.0) -> bool:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return True
    indices = [int(total * i / sample_frames) for i in range(1, sample_frames + 1)]
    brightness_vals = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            brightness_vals.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
    cap.release()
    return np.mean(brightness_vals) < brightness_threshold if brightness_vals else True


def validate_clip(path: Path, min_duration: float = 1.5) -> dict:
    try:
        info = probe_video(path)
        streams = info.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            return {"path": str(path), "valid": False, "reason": "no video stream"}
        duration = float(info.get("format", {}).get("duration", 0))
        if duration < min_duration:
            return {"path": str(path), "valid": False,
                    "reason": f"too short: {duration:.1f}s"}
        if is_mostly_black(path):
            return {"path": str(path), "valid": False, "reason": "black frames"}
        return {"path": str(path), "valid": True, "reason": "ok"}
    except Exception as e:
        return {"path": str(path), "valid": False, "reason": str(e)}


def validate_generated_dataset(clips_dir: Path, annotations_csv: Path) -> pd.DataFrame:
    ann = pd.read_csv(annotations_csv)
    results = []
    for category in ["fighting", "vandalism", "stabbing", "shooting"]:
        cat_dir = clips_dir / category
        if not cat_dir.exists():
            continue
        for clip in tqdm(list(cat_dir.glob("*.mp4")), desc=f"Validating {category}"):
            results.append(validate_clip(clip))
    df = pd.DataFrame(results)
    total = len(df)
    valid = df["valid"].sum()
    print(f"\nValidation: {valid}/{total} clips passed ({100*valid/total:.1f}%)")
    print("\nFailure reasons:")
    print(df[~df["valid"]]["reason"].value_counts().to_string())
    return df


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    result_df = validate_generated_dataset(
        clips_dir=base / "generated" / "clips",
        annotations_csv=base / "generated" / "annotations" / "annotations.csv",
    )
    result_df.to_csv(base / "generated" / "annotations" / "validation_report.csv",
                     index=False)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_validate_dataset.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_dataset.py tests/test_validate_dataset.py
git commit -m "feat: generated dataset quality validation — duration, black-frame detection"
```

---

## Task 10: End-to-End Run & Dataset Packaging

This is the operational task that ties all scripts together into a full production run.

- [ ] **Step 1: Preprocess all categories**

```bash
source .venv/bin/activate
for cat in fighting vandalism stabbing shooting; do
  python scripts/preprocess.py --category $cat
done
```

Expected: `datasets/processed/<category>/` populated with trimmed, scene-cut-validated clips.

- [ ] **Step 2: Auto-caption all categories**

```bash
# Ensure Ollama is running: ollama serve
for cat in fighting vandalism stabbing shooting; do
  python scripts/caption.py --category $cat
done
```

Expected: each `.mp4` in processed directories has a matching `.txt` caption.

- [ ] **Step 3: Train all LoRAs**

```bash
bash scripts/train_all.sh
```

Monitor TensorBoard at http://localhost:6006. Stop each run at the best checkpoint (target loss ~0.03).

- [ ] **Step 4: Generate synthetic dataset**

```bash
# Ensure ComfyUI is running: python main.py --listen
for cat in fighting vandalism stabbing shooting; do
  python scripts/generate.py --category $cat --count 200
done
```

Expected: 800 total synthetic clips across 4 categories, manifest CSVs written.

- [ ] **Step 5: Build annotation files**

```bash
python scripts/annotate.py
```

Expected: `generated/annotations/annotations.csv` and `annotations_coco.json`.

- [ ] **Step 6: Validate**

```bash
python scripts/validate_dataset.py
```

Expected: >90% pass rate. Investigate failures from `validation_report.csv`.

- [ ] **Step 7: Package final dataset**

```bash
cd generated
tar -czf synthetic_violence_dataset_v1.tar.gz clips/ annotations/
echo "Dataset packaged: $(du -sh synthetic_violence_dataset_v1.tar.gz)"
```

- [ ] **Step 8: Final commit**

```bash
git add generated/annotations/
git commit -m "feat: v1 synthetic violence detection dataset — 800 clips, 4 categories"
```

---

## Monitoring Checklist (Per Training Run)

> **WARNING: WAN 2.2 hides overfit behind pretty samples.** Outputs can look good while the LoRA has actually lost generalization. Always test with novel prompts, not just training-similar ones.

| Checkpoint | What to Check | Action if Bad |
|---|---|---|
| Step 300 | Loss descending smoothly | If flat: check LR, verify captions have trigger word |
| Step 600 | Sample video shows correct action category | If wrong: verify dataset quality, check caption prompt |
| Step 1200 | Trigger word reliably activates behavior | If not: reduce LR 30%, extend steps by 500 |
| Step 1800 | **Generalization test**: generate with novel prompts (different scenes, lighting, camera angles not in training data). Output should show the correct action in the new context | If all outputs look like training data: overfitting. Use earlier checkpoint, add regularization clips |
| Step 1800 | No background from training data baked in | If yes: add varied-background clips to dataset |
| Step 2400+ | Loss plateau, action identity stable across diverse prompts | Save checkpoint, stop training |

### Overfit Detection (Wan 2.2 Specific)

| Symptom | What It Means |
|---|---|
| **Prompt inertia** | Changing the prompt barely changes output — it looks the same regardless |
| **Skin plasticity** | If people are in the scene, skin looks unnaturally smooth and plastic |
| **Color clustering** | All outputs share the same color temperature as training data |
| **Background homogenization** | Every output has the same shallow DoF / soft bokeh background (Wan 2.2 default) |

## Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| VLM captions wrong category | Generic scene, no obvious action | Manually review and edit `.txt` files |
| Loss spike at step 800 | Gradient explosion | Confirm `max_grad_norm: 1.0` in config |
| Generated videos are black | ComfyUI workflow node mismatch | Update `PATCH_FIELDS` node IDs in `generate.py` to match your exported workflow |
| LoRA ignores trigger word | Alpha ≠ rank in config | Ensure `linear_alpha: 32` matches `linear: 32` in YAML |
| LoRA has no effect at all | Only one expert trained, or wrong `--dit` path | Verify both `_high` and `_low` safetensors exist in `loras/<category>/` and are loaded in ComfyUI |
| Cinematic look instead of CCTV | Wan 2.2 overriding aesthetic | Add regularization clips (see Task 2), increase regularization ratio |
| Stabbing/shooting clips too scarce | Hard to find clean source material | Use Blender/movie prop footage, screen-record licensed films |
| Plasticky skin, loss spikes | Learning rate too high | Reduce LR from 1e-4 to 5e-5, monitor loss curve |
