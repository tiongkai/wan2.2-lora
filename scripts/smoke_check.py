from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.categories import CATEGORIES
from scripts.generate import build_workflow as build_t2v_workflow
from scripts.generate_i2v import build_workflow as build_i2v_workflow
from scripts.pipeline_config import config_path, load_pipeline_config


def main() -> None:
    cfg = load_pipeline_config()
    required_sections = {"paths", "runtime", "wan", "generation", "categories"}
    missing = required_sections - set(cfg)
    if missing:
        raise SystemExit(f"Missing config sections: {', '.join(sorted(missing))}")

    prompt_root = config_path("paths", "prompts_dir")  # repo-anchored, not CWD-relative
    for name, category in CATEGORIES.items():
        if not category.enabled:
            continue
        prompt_file = prompt_root / category.prompt_file
        if not prompt_file.exists():
            raise SystemExit(f"Missing prompt file for {name}: {prompt_file}")

    build_t2v_workflow("fght99, smoke test", 1, "fighting")
    build_i2v_workflow(
        "fght99, smoke test",
        1,
        "fighting",
        "sample.png",
        cfg["wan"]["short_frames"],
        cfg["generation"]["default_lora_strength"],
        "smoke/test",
        width=cfg["generation"]["default_width"],
        height=cfg["generation"]["default_height"],
        no_lora=False,
    )
    build_i2v_workflow(
        "fght99, smoke test",
        1,
        "fighting",
        "sample.png",
        cfg["wan"]["five_second_generation_frames"],
        cfg["generation"]["default_lora_strength"],
        "smoke/test",
        width=cfg["generation"]["default_width"],
        height=cfg["generation"]["default_height"],
        no_lora=True,
    )
    print("Smoke check ok")


if __name__ == "__main__":
    main()
