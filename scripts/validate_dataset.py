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
