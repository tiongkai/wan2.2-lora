import subprocess, json, cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.manifest import load_manifests, resolve_repo_path


QA_RESULT_COLUMNS = [
    "path",
    "valid",
    "reason",
    "duration",
    "width",
    "height",
    "frame_count",
    "black_frame_status",
    "blur_score",
    "static_score",
    "perceptual_hash",
    "duplicate_warning",
]
QA_CONTEXT_COLUMNS = ["manifest_path", "output_path", "category", "method", "id"]
VALIDATION_REPORT_COLUMNS = QA_RESULT_COLUMNS + QA_CONTEXT_COLUMNS
QA_DEFAULTS = {
    "path": "",
    "valid": False,
    "reason": "",
    "duration": None,
    "width": None,
    "height": None,
    "frame_count": None,
    "black_frame_status": None,
    "blur_score": None,
    "static_score": None,
    "perceptual_hash": "",
    "duplicate_warning": False,
}


def qa_result_row(path: Path | str, valid: bool, reason: str, **updates) -> dict:
    row = QA_DEFAULTS.copy()
    row.update({"path": str(path), "valid": bool(valid), "reason": reason})
    row.update(updates)
    return {column: row.get(column, "") for column in QA_RESULT_COLUMNS}


def _stream_metrics(stream: dict, duration: float) -> dict:
    return {
        "duration": duration,
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "frame_count": int(stream.get("nb_frames", 0) or 0),
    }


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
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


def _sample_frames(path: Path, sample_frames: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [min(total - 1, int(total * i / (sample_frames + 1))) for i in range(1, sample_frames + 1)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def blur_score(path: Path) -> float:
    frames = _sample_frames(path)
    if not frames:
        return 0.0
    vals = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return float(np.mean(vals))


def static_score(path: Path) -> float:
    frames = _sample_frames(path)
    if len(frames) < 2:
        return 1.0
    diffs = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(cv2.absdiff(prev, gray))))
        prev = gray
    # Higher means more static: 1.0 no change, 0.0 substantial change.
    return max(0.0, min(1.0, 1.0 - (float(np.mean(diffs)) / 30.0)))


def average_hash(path: Path) -> str:
    frames = _sample_frames(path, sample_frames=1)
    if not frames:
        return ""
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = float(np.mean(small))
    bits = small > avg
    return "".join("1" if b else "0" for b in bits.flatten())


def validate_clip(path: Path, min_duration: float = 1.5, quality_checks: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        return qa_result_row(path, False, "missing file")
    try:
        info = probe_video(path)
        streams = info.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            return qa_result_row(path, False, "no video stream")
        duration = float(info.get("format", {}).get("duration", 0))
        stream = video_streams[0]
        metrics = _stream_metrics(stream, duration)
        if duration < min_duration:
            return qa_result_row(path, False, f"too short: {duration:.1f}s", **metrics)
        black_frames = is_mostly_black(path)
        if black_frames:
            return qa_result_row(path, False, "black frames", **metrics, black_frame_status=True)
        blur = blur_score(path)
        static = static_score(path)
        reasons = []
        if quality_checks:
            if blur < 12.0:
                reasons.append("blurry")
            if static > 0.97:
                reasons.append("static")
        return qa_result_row(
            path,
            not reasons,
            "; ".join(reasons) if reasons else "ok",
            **metrics,
            black_frame_status=False,
            blur_score=blur,
            static_score=static,
            perceptual_hash=average_hash(path),
        )
    except Exception as e:
        return qa_result_row(path, False, str(e))


def validate_manifest_dataset(generated_dir: Path) -> pd.DataFrame:
    manifest = load_manifests(generated_dir)
    if manifest.empty:
        print("\nValidation: no manifests found")
        return pd.DataFrame(columns=VALIDATION_REPORT_COLUMNS)
    results = []
    ok = manifest[manifest["status"] == "ok"].copy()
    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="Validating manifest clips"):
        output_path = str(row.get("output_path") or "")
        if not output_path:
            result = qa_result_row("", False, "missing output_path")
        else:
            result = validate_clip(resolve_repo_path(output_path), quality_checks=True)
        result.update({
            "manifest_path": row.get("manifest_path", ""),
            "output_path": output_path,
            "category": row.get("category", ""),
            "method": row.get("method", ""),
            "id": row.get("id", ""),
        })
        results.append(result)
    df = pd.DataFrame(results)
    if df.empty:
        print("\nValidation: no clips found")
        return pd.DataFrame(columns=VALIDATION_REPORT_COLUMNS)
    df = df.reindex(columns=VALIDATION_REPORT_COLUMNS)
    dupes = df["perceptual_hash"].duplicated(keep=False) & df["perceptual_hash"].fillna("").ne("")
    df["duplicate_warning"] = dupes
    df.loc[dupes & df["valid"], "valid"] = False
    df.loc[dupes, "reason"] = df.loc[dupes, "reason"].where(
        df.loc[dupes, "reason"].ne("ok"), "duplicate"
    )
    total = len(df)
    valid = df["valid"].sum()
    print(f"\nValidation: {valid}/{total} clips passed ({100*valid/total:.1f}%)")
    print("\nFailure reasons:")
    print(df[~df["valid"]]["reason"].value_counts().to_string())
    return df


def write_validated_manifest(generated_dir: Path, validation_df: pd.DataFrame, out_path: Path) -> None:
    manifest = load_manifests(generated_dir)
    validation_df = validation_df.reindex(columns=VALIDATION_REPORT_COLUMNS)
    if manifest.empty or validation_df.empty:
        validation_df.to_csv(out_path, index=False)
        return
    qa = validation_df[["output_path", "valid", "reason"]].rename(
        columns={"valid": "qa_pass", "reason": "qa_reason"}
    )
    joined = manifest.merge(qa.drop_duplicates("output_path", keep="last"), how="left", on="output_path")
    joined.to_csv(out_path, index=False)


def validate_generated_dataset(clips_dir: Path, annotations_csv: Path | None = None) -> pd.DataFrame:
    """Legacy directory validator retained for tests and ad-hoc checks."""
    results = []
    for cat_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        for clip in tqdm(list(cat_dir.glob("*.mp4")), desc=f"Validating {cat_dir.name}"):
            results.append(validate_clip(clip))
    df = pd.DataFrame(results)
    if df.empty:
        print("\nValidation: no clips found")
        return pd.DataFrame(columns=QA_RESULT_COLUMNS)
    df = df.reindex(columns=QA_RESULT_COLUMNS)
    total = len(df)
    valid = df["valid"].sum()
    print(f"\nValidation: {valid}/{total} clips passed ({100*valid/total:.1f}%)")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-dirs", action="store_true", help="Validate generated/clips folders directly")
    args = p.parse_args()
    base = Path(__file__).parent.parent
    if args.legacy_dirs:
        result_df = validate_generated_dataset(clips_dir=base / "generated" / "clips")
    else:
        result_df = validate_manifest_dataset(generated_dir=base / "generated")
    (base / "generated" / "annotations").mkdir(parents=True, exist_ok=True)
    report_path = base / "generated" / "annotations" / "validation_report.csv"
    result_df.to_csv(report_path, index=False)
    if not args.legacy_dirs:
        write_validated_manifest(
            generated_dir=base / "generated",
            validation_df=result_df,
            out_path=base / "generated" / "annotations" / "manifest_validated.csv",
        )
