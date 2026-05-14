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
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    return float(video_streams[0]["duration"]) if video_streams else 0.0


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
