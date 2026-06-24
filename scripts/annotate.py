import json
import pandas as pd
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from numbers import Integral, Real

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.categories import CLASS_MAP
from scripts.manifest import load_manifests, resolve_repo_path


LONG_METHODS = {"extend_chain", "extend_chain_lora", "svi"}
WINDOW_COLUMNS = ["source_video", "window_index", "window_start_frame", "window_end_frame"]


def normalize_seed(value):
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return int(number) if number.is_integer() else str(number)
    text = str(value).strip()
    if not text:
        return ""
    try:
        return int(text)
    except ValueError:
        return text


def _json_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def build_annotation_csv(df: pd.DataFrame, class_map: dict) -> pd.DataFrame:
    df = df.copy()
    if "class_id" not in df.columns:
        df["class_id"] = df["category"].map(class_map)
    if "output_path" in df.columns and df["output_path"].fillna("").astype(str).str.len().any():
        df["filename"] = df["output_path"]
    else:
        df["filename"] = ""
    cols = ["filename", "category", "class_id", "prompt", "seed", "status"]
    for optional in [
        "method", "frames", "fps", "width", "height", "start_frame",
        "source_video", "window_index", "window_start_frame", "window_end_frame",
    ]:
        if optional in df.columns:
            cols.append(optional)
    return df[cols]


def build_coco_json(df: pd.DataFrame, class_map: dict, clips_dir: Path) -> dict:
    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    videos, annotations = [], []
    for _, row in df.iterrows():
        vid_id = int(row["id"])
        fname = row.get("output_path") or ""
        video = {"id": vid_id, "file_name": fname,
                 "category": row["category"],
                 "method": row.get("method", "")}
        annotation = {
            "id": vid_id, "video_id": vid_id,
            "category_id": class_map[row["category"]],
            "prompt": row["prompt"], "seed": normalize_seed(row.get("seed", "")),
        }
        for column in WINDOW_COLUMNS:
            if column in df.columns:
                value = _json_value(row.get(column, ""))
                video[column] = value
                annotation[column] = value
        videos.append(video)
        annotations.append(annotation)
    return {"categories": categories, "videos": videos, "annotations": annotations}


def _read_qa_report(generated_dir: Path) -> pd.DataFrame:
    report = generated_dir / "annotations" / "validation_report.csv"
    if not report.exists():
        return pd.DataFrame()
    qa = pd.read_csv(report)
    if "output_path" not in qa.columns or "valid" not in qa.columns:
        return pd.DataFrame()
    if "reason" not in qa.columns:
        qa["reason"] = ""
    qa = qa[["output_path", "valid", "reason"]].copy()
    qa = qa.rename(columns={"valid": "qa_pass", "reason": "qa_reason"})
    return qa.drop_duplicates(subset=["output_path"], keep="last")


def _apply_long_mode(df: pd.DataFrame, long_mode: str, window_frames: int) -> pd.DataFrame:
    if long_mode == "whole":
        return df
    if df.empty:
        return df.copy()
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    rows = []
    for _, row in df.iterrows():
        method = str(row.get("method") or "")
        frames_value = row.get("frames")
        frames = int(float(frames_value)) if not pd.isna(frames_value) and str(frames_value).strip() else 0
        if method not in LONG_METHODS or frames <= window_frames:
            rows.append(row.to_dict())
            continue
        for start in range(0, frames, window_frames):
            end = min(frames, start + window_frames)
            out = row.to_dict()
            out["source_video"] = row.get("output_path", "")
            out["window_index"] = start // window_frames
            out["window_start_frame"] = start
            out["window_end_frame"] = end
            out["frames"] = end - start
            rows.append(out)
            if end == frames:
                break
    return pd.DataFrame(rows)


def _is_explicit_true(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _valid_export_rows(df: pd.DataFrame, qa_report_exists: bool = False, allow_unvalidated: bool = False) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if qa_report_exists:
        if "qa_pass" not in ok.columns:
            ok["qa_pass"] = pd.NA
        explicit_pass = ok["qa_pass"].map(_is_explicit_true)
        has_qa = ok["qa_pass"].notna()
        if allow_unvalidated:
            keep = explicit_pass | ~has_qa
        else:
            missing_count = int((~has_qa).sum())
            if missing_count:
                print(
                    f"Excluded {missing_count} unvalidated clips; "
                    "rerun validation or pass --allow-unvalidated to include them."
                )
            keep = explicit_pass
        ok = ok[keep].copy()
    missing_output = ok["output_path"].fillna("").astype(str) == ""
    if missing_output.any():
        examples = ok[missing_output].head(3)[["category", "id", "manifest_path"]].to_dict("records")
        raise FileNotFoundError(f"Manifest rows are missing output_path: {examples}")
    missing_files = [p for p in ok["output_path"] if not resolve_repo_path(p).exists()]
    if missing_files:
        raise FileNotFoundError(f"Manifest output files do not exist: {missing_files[:5]}")
    return ok


def export_dataset(
    generated_dir: Path,
    out_dir: Path,
    methods: set[str] | None = None,
    dry_run: bool = False,
    long_mode: str = "whole",
    window_frames: int = 81,
    allow_unvalidated: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = load_manifests(generated_dir)
    if combined.empty:
        raise FileNotFoundError("No manifest CSVs found. Run generation first.")
    qa = _read_qa_report(generated_dir)
    qa_report_exists = (generated_dir / "annotations" / "validation_report.csv").exists()
    if not qa.empty:
        combined = combined.merge(qa, how="left", on="output_path")
    elif qa_report_exists:
        combined["qa_pass"] = pd.NA
        combined["qa_reason"] = ""
    if methods:
        combined = combined[combined["method"].isin(methods)]
    combined = _valid_export_rows(combined, qa_report_exists=qa_report_exists, allow_unvalidated=allow_unvalidated)
    combined = _apply_long_mode(combined, long_mode=long_mode, window_frames=window_frames)
    combined["id"] = range(len(combined))  # globally unique IDs

    csv_df = build_annotation_csv(combined, CLASS_MAP)
    coco = build_coco_json(combined, CLASS_MAP, clips_dir=generated_dir / "clips")
    if dry_run:
        print(f"Dry run: {len(csv_df)} clips exportable")
        print(csv_df[["filename", "category", "class_id"]].head(20).to_string(index=False))
        return csv_df
    _write_csv_atomic(csv_df, out_dir / "annotations.csv")
    print(f"Saved annotations.csv ({len(csv_df)} clips)")

    _write_text_atomic(out_dir / "annotations_coco.json", json.dumps(coco, indent=2))
    print(f"Saved annotations_coco.json")

    print("\nClass distribution:")
    print(csv_df["category"].value_counts().to_string())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="", help="Comma-separated methods to export")
    p.add_argument("--dry-run", action="store_true", help="List exportable clips without writing annotations")
    p.add_argument("--long-mode", choices=["whole", "windows"], default="whole")
    p.add_argument("--window-frames", type=int, default=81)
    p.add_argument("--allow-unvalidated", action="store_true", help="Include clips missing QA rows when a QA report exists")
    args = p.parse_args()
    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None
    base = Path(__file__).parent.parent
    export_dataset(
        generated_dir=base / "generated",
        out_dir=base / "generated" / "annotations",
        methods=methods,
        dry_run=args.dry_run,
        long_mode=args.long_mode,
        window_frames=args.window_frames,
        allow_unvalidated=args.allow_unvalidated,
    )
