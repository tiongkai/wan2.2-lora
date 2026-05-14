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
        "-c:v", "mpeg4", "-q:v", "3", "-an",
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
