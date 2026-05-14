import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_multi_keyframes_creates_jpgs(tmp_path):
    import subprocess
    src = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=red:s=640x480:r=24:d=3",
        "-c:v", "mpeg4", str(src)
    ], check=True, capture_output=True)
    from scripts.caption import extract_multi_keyframes
    frames = extract_multi_keyframes(src, n_frames=5, out_dir=tmp_path)
    assert len(frames) == 5
    for f in frames:
        assert f.exists()
        assert f.suffix == ".jpg"


def test_extract_keyframes_respects_n_frames(tmp_path):
    import subprocess
    src = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=green:s=640x480:r=24:d=3",
        "-c:v", "mpeg4", str(src)
    ], check=True, capture_output=True)
    from scripts.caption import extract_multi_keyframes
    frames = extract_multi_keyframes(src, n_frames=3, out_dir=tmp_path)
    assert len(frames) == 3


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
