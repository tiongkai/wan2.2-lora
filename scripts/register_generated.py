from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.categories import require_category
from scripts.manifest import build_manifest_row, sha256_file, write_manifest


def register_video(
    output: Path,
    category: str,
    method: str,
    prompt: str,
    seed: int | str = "",
    start_frame: Path | None = None,
    frames: int | str = "",
    fps: int = 16,
    width: int | str = "",
    height: int | str = "",
    workflow_template: Path | None = None,
    model_stack: str = "wan2.2-i2v-a14b",
) -> Path:
    require_category(category)
    if not output.exists():
        raise FileNotFoundError(output)
    workflow_hash = sha256_file(workflow_template) if workflow_template and workflow_template.exists() else ""
    row = build_manifest_row(
        id=0,
        category=category,
        method=method,
        prompt=prompt,
        seed=seed,
        start_frame=start_frame or "",
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        workflow_template=workflow_template or "",
        workflow_sha256=workflow_hash,
        model_stack=model_stack,
        output_path=output,
        status="ok",
    )
    manifest_path = output.parent / f"{output.stem}_manifest.csv"
    write_manifest([row], manifest_path)
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description="Register an existing generated video into the canonical manifest lifecycle.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--category", required=True)
    p.add_argument("--method", required=True, choices=["svi", "extend_chain", "i2v_base", "i2v_lora", "t2v"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", default="")
    p.add_argument("--start-frame", type=Path, default=None)
    p.add_argument("--frames", default="")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--width", default="")
    p.add_argument("--height", default="")
    p.add_argument("--workflow-template", type=Path, default=None)
    p.add_argument("--model-stack", default="wan2.2-i2v-a14b")
    args = p.parse_args()
    manifest = register_video(
        output=args.output,
        category=args.category,
        method=args.method,
        prompt=args.prompt,
        seed=args.seed,
        start_frame=args.start_frame,
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        workflow_template=args.workflow_template,
        model_stack=args.model_stack,
    )
    print(f"Manifest saved: {manifest}")


if __name__ == "__main__":
    main()
