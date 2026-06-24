from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG))
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def config_path(*keys: str, default: str = "") -> Path:
    value = config_value(*keys, default=default)
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def config_value(*keys: str, default: Any = None) -> Any:
    current: Any = load_pipeline_config()
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

