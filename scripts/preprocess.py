import subprocess, json, re, sys
from pathlib import Path
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import probe_duration
from scripts.categories import enabled_categories


def effective_wan_frames(duration_seconds: float, wan_fps: int = 16) -> int:
    return int(duration_seconds * wan_fps)


def is_valid_wan_length(frames: int) -> bool:
    return frames > 0 and (frames - 1) % 4 == 0


def nearest_valid_wan_length(frames: int) -> int:
    if frames < 1:
        return 0
    return frames - ((frames - 1) % 4)


def validate_target_frames(duration_seconds: float, target_frames: int, wan_fps: int = 16) -> None:
    if not is_valid_wan_length(target_frames):
        raise ValueError(f"target_frames must be 4n+1 for Wan, got {target_frames}")
    available = effective_wan_frames(duration_seconds, wan_fps=wan_fps)
    if available < target_frames:
        nearest = nearest_valid_wan_length(available)
        raise ValueError(
            f"target_frames={target_frames} requires {target_frames / wan_fps:.2f}s at {wan_fps} fps, "
            f"but clip only provides {available} effective frames. Max valid target is {nearest}."
        )


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
        sys.executable, "-m", "scenedetect",
        "-i", str(path),
        "detect-adaptive", f"--threshold={threshold}",
        "list-scenes", "-n"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scenedetect failed on {path}: {result.stderr[:200]}")
    m = re.search(r"Detected (\d+) scenes", result.stdout)
    if not m:
        raise RuntimeError(f"Could not parse scenedetect output for {path}")
    return int(m.group(1)) > 1


def process_category(raw_dir: Path, out_dir: Path,
                     min_seconds: float = 2.0, max_seconds: float = 5.0,
                     target_frames: int | None = None, wan_fps: int = 16):
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.mov")) + list(raw_dir.glob("*.avi"))
    accepted, rejected = 0, 0
    report_rows = []
    for clip in tqdm(clips, desc=f"Processing {raw_dir.name}"):
        try:
            duration = probe_duration(clip)
        except Exception as exc:
            rejected += 1
            report_rows.append({"clip": clip.name, "accepted": False, "reason": str(exc)})
            continue
        if duration < min_seconds:
            rejected += 1
            report_rows.append({"clip": clip.name, "accepted": False, "reason": f"too short: {duration:.2f}s"})
            continue
        if target_frames is not None:
            try:
                validate_target_frames(min(duration, max_seconds), target_frames, wan_fps=wan_fps)
            except ValueError as exc:
                rejected += 1
                report_rows.append({"clip": clip.name, "accepted": False, "reason": str(exc)})
                continue
        if has_scene_cut(clip):
            rejected += 1
            report_rows.append({"clip": clip.name, "accepted": False, "reason": "scene cut"})
            continue
        dst = out_dir / (clip.stem + ".mp4")
        trim_clip(clip, dst, max_seconds=max_seconds)
        accepted += 1
        processed_duration = min(duration, max_seconds)
        report_rows.append({
            "clip": clip.name,
            "accepted": True,
            "reason": "ok",
            "original_duration": duration,
            "processed_duration": processed_duration,
            "wan_fps": wan_fps,
            "effective_wan_frames": effective_wan_frames(processed_duration, wan_fps=wan_fps),
            "max_valid_wan_frames": nearest_valid_wan_length(effective_wan_frames(processed_duration, wan_fps=wan_fps)),
            "target_frames": target_frames or "",
        })
    if report_rows:
        import pandas as pd
        pd.DataFrame(report_rows).to_csv(out_dir / "preprocess_report.csv", index=False)
    print(f"{raw_dir.name}: {accepted} accepted, {rejected} rejected")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=enabled_categories())
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Override raw clips directory (default: datasets/raw/<category>)")
    p.add_argument("--target-frames", type=int, default=None,
                   help="Wan target frame count to validate, e.g. 33 or 77")
    p.add_argument("--wan-fps", type=int, default=16)
    args = p.parse_args()
    base = Path(__file__).parent.parent
    raw_dir = args.raw_dir or (base / "datasets" / "raw" / args.category)
    process_category(
        raw_dir=raw_dir,
        out_dir=base / "datasets" / "processed" / args.category,
        target_frames=args.target_frames,
        wan_fps=args.wan_fps,
    )
