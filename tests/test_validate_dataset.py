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


def test_failure_results_keep_stable_qa_columns(tmp_path):
    from scripts.validate_dataset import QA_RESULT_COLUMNS, validate_clip

    black = tmp_path / "black.mp4"
    corrupt = tmp_path / "corrupt.mp4"
    missing = tmp_path / "missing.mp4"
    make_black_video(black)
    corrupt.write_bytes(b"not a video")

    for result in [validate_clip(black), validate_clip(missing), validate_clip(corrupt)]:
        assert set(QA_RESULT_COLUMNS).issubset(result)
        assert result["valid"] is False
        assert "blur_score" in result
        assert "static_score" in result
        assert "perceptual_hash" in result


def test_manifest_validation_all_failure_report_has_stable_columns(tmp_path):
    from scripts.manifest import build_manifest_row, write_manifest
    from scripts.validate_dataset import VALIDATION_REPORT_COLUMNS, validate_manifest_dataset

    generated = tmp_path / "generated"
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="i2v_base",
        prompt="fght99, missing",
        seed=1,
        frames=81,
        output_path=tmp_path / "missing.mp4",
        status="ok",
    )
    write_manifest([row], generated / "clips" / "fighting_i2v_base" / "manifest.csv")

    report = validate_manifest_dataset(generated)
    assert list(report.columns) == VALIDATION_REPORT_COLUMNS
    assert len(report) == 1
    assert not bool(report.iloc[0]["valid"])
    assert report.iloc[0]["reason"] == "missing file"
