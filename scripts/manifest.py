from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.categories import CLASS_MAP


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FPS = 16
CANONICAL_COLUMNS = [
    "id",
    "category",
    "class_id",
    "method",
    "prompt",
    "negative_prompt",
    "seed",
    "start_frame",
    "frames",
    "fps",
    "width",
    "height",
    "workflow_template",
    "workflow_sha256",
    "model_stack",
    "lora_high",
    "lora_low",
    "lora_strength",
    "output_path",
    "prompt_id",
    "status",
    "error_reason",
    "created_at",
    "shard",
    "num_shards",
]
SUPPORTED_METHODS = {"t2v", "i2v_base", "i2v_lora", "extend_chain", "extend_chain_lora", "svi"}
REQUIRED_COLUMNS = {
    "id",
    "category",
    "class_id",
    "method",
    "prompt",
    "seed",
    "frames",
    "fps",
    "output_path",
    "status",
    "created_at",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_relative(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: "" for column in CANONICAL_COLUMNS}
    row.update(kwargs)
    category = str(row.get("category") or "")
    if not row.get("class_id") and category in CLASS_MAP:
        row["class_id"] = CLASS_MAP[category]
    if not row.get("fps"):
        row["fps"] = DEFAULT_FPS
    if not row.get("created_at"):
        row["created_at"] = now_iso()
    for key in ["start_frame", "workflow_template", "output_path"]:
        row[key] = repo_relative(row.get(key))
    return row


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df[CANONICAL_COLUMNS + [c for c in df.columns if c not in CANONICAL_COLUMNS]].to_csv(path, index=False)


def discover_manifests(generated_dir: Path) -> list[Path]:
    if not generated_dir.exists():
        return []
    return sorted(
        p for p in generated_dir.rglob("*manifest*.csv")
        if "annotations" not in p.parts and p.is_file()
    )


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["manifest_path"] = repo_relative(path)
    return coerce_manifest(df)


def load_manifests(generated_dir: Path) -> pd.DataFrame:
    frames = [read_manifest(path) for path in discover_manifests(generated_dir)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)


def validate_manifest_df(df: pd.DataFrame, require_files: bool = False) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"missing columns: {', '.join(sorted(missing))}")
        return errors
    for idx, row in df.iterrows():
        prefix = f"row {idx}"
        category = str(row.get("category") or "")
        if category not in CLASS_MAP:
            errors.append(f"{prefix}: unsupported category {category!r}")
        method = str(row.get("method") or "")
        if method not in SUPPORTED_METHODS:
            errors.append(f"{prefix}: unsupported method {method!r}")
        status = str(row.get("status") or "")
        if status == "ok":
            output_path = str(row.get("output_path") or "")
            if not output_path:
                errors.append(f"{prefix}: status ok but output_path is empty")
            elif require_files and not resolve_repo_path(output_path).exists():
                errors.append(f"{prefix}: output file does not exist: {output_path}")
    return errors


def coerce_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "class_id" not in df.columns and "category" in df.columns:
        df["class_id"] = df["category"].map(CLASS_MAP)
    if "method" not in df.columns:
        df["method"] = df.apply(_infer_method, axis=1)
    if "fps" not in df.columns:
        df["fps"] = DEFAULT_FPS
    if "output_path" not in df.columns:
        df["output_path"] = ""
    if "error_reason" not in df.columns:
        df["error_reason"] = ""
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _infer_method(row: pd.Series) -> str:
    manifest_path = str(row.get("manifest_path", ""))
    category = str(row.get("category", ""))
    if "_i2v_base" in manifest_path:
        return "i2v_base"
    if "_i2v" in manifest_path:
        return "i2v_lora"
    if "/svi/" in manifest_path:
        return "svi"
    if "/long" in manifest_path:
        return "extend_chain"
    if category:
        return "t2v"
    return "unknown"


def comfy_history_output_paths(history_entry: dict[str, Any], output_root: Path) -> list[str]:
    paths: list[str] = []
    outputs = history_entry.get("outputs", {}) if history_entry else {}
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                subfolder = item.get("subfolder") or ""
                filename = item["filename"]
                path = output_root / subfolder / filename
                paths.append(repo_relative(path))
    return paths


def latest_matching_output(output_root: Path, prefix: str, since: float) -> str:
    prefix_path = Path(prefix)
    search_root = output_root / prefix_path.parent if str(prefix_path.parent) != "." else output_root
    if not search_root.exists():
        return ""
    candidates = [
        p for p in search_root.glob(f"{prefix_path.name}*.mp4")
        if p.stat().st_mtime >= since
    ]
    return repo_relative(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else ""


def manifest_summary_json(row: dict[str, Any]) -> str:
    return json.dumps({k: row.get(k, "") for k in CANONICAL_COLUMNS}, sort_keys=True)
