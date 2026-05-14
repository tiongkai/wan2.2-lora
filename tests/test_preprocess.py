import pytest
from pathlib import Path
import subprocess

# Helper: create a 6-second solid-color test video
def make_test_video(path: Path, duration: int = 6):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=blue:s=1280x720:r=24:d={duration}",
        "-c:v", "mpeg4", str(path)
    ], check=True, capture_output=True)

def test_trim_clips_to_max_five_seconds(tmp_path):
    from scripts.preprocess import trim_clip
    src = tmp_path / "input.mp4"
    dst = tmp_path / "output.mp4"
    make_test_video(src, duration=6)
    trim_clip(src, dst, max_seconds=5)
    import subprocess, json
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(dst)
    ], capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info["streams"][0]["duration"])
    assert 4.5 <= duration <= 5.1  # 0.1s tolerance for keyframe rounding

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

def test_single_scene_clip_passes_scene_cut_check(tmp_path):
    from scripts.preprocess import has_scene_cut
    src = tmp_path / "single.mp4"
    make_test_video(src, duration=3)
    assert has_scene_cut(src) is False

def test_scene_cut_does_not_pollute_cwd(tmp_path, monkeypatch):
    from scripts.preprocess import has_scene_cut
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "clip.mp4"
    make_test_video(src, duration=3)
    has_scene_cut(src)
    csv_files = list(tmp_path.glob("*.csv"))
    assert csv_files == [], f"Unexpected CSV files created: {csv_files}"
