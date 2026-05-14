import subprocess, pytest
from pathlib import Path


def make_video(path, duration=3, color="blue"):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=1280x720:r=24:d={duration}",
        "-c:v", "mpeg4", str(path)
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
