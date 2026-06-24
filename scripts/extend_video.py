# scripts/extend_video.py
"""
Extend a single start image into a LONGER video by I2V-chaining base Wan2.2 segments:
gen segment -> extract its last frame -> use as the next segment's start -> repeat,
then concat with ffmpeg. This is the extend-and-stitch FALLBACK (drift accumulates);
SVI is the higher-quality method (see wiki/synthesis/long-video-generation-wan22.md).

Usage:
  python scripts/extend_video.py --image sample.png --segments 4 --frames 81 \
    --prompt "..." --port 8188 --out generated/clips/long/clip_20s.mp4
"""
import argparse, time, subprocess, importlib.util, uuid
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.categories import require_category
from scripts.manifest import (
    build_manifest_row,
    comfy_history_output_paths,
    repo_relative,
    resolve_repo_path,
    sha256_file,
    write_manifest,
)

# Reuse helpers from generate_i2v.py (upload/queue/wait/build_workflow).
_spec = importlib.util.spec_from_file_location("gi", str(Path(__file__).parent / "generate_i2v.py"))
gi = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gi)

FFMPEG = "/home/lenovo5/ffmpeg/ffmpeg-n6.1.1-31-gaa5e6017a5-linux64-lgpl-shared-6.1/bin/ffmpeg"
OUTPUT_DIR = Path(__file__).parent.parent / "generated" / "clips"


def _matches_segment_prefix(path: Path, output_root: Path, prefix: str) -> bool:
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError:
        return False
    return rel == f"{prefix}.mp4" or rel.startswith(f"{prefix}_")


def _latest_prefix_mp4(output_root: Path, prefix: str, since: float) -> Path | None:
    prefix_path = Path(prefix)
    search_root = output_root / prefix_path.parent if str(prefix_path.parent) != "." else output_root
    if not search_root.exists():
        return None
    candidates = [
        p for p in search_root.glob(f"{prefix_path.name}*.mp4")
        if p.stat().st_mtime >= since and _matches_segment_prefix(p, output_root, prefix)
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def resolve_segment_output(history_entry: dict, output_root: Path, prefix: str, since: float | None = None) -> Path | None:
    for output_path in comfy_history_output_paths(history_entry, output_root):
        path = resolve_repo_path(output_path)
        if path.suffix.lower() == ".mp4" and _matches_segment_prefix(path, output_root, prefix):
            return path
    if since is None:
        return None
    return _latest_prefix_mp4(output_root, prefix, since)


def extract_last_frame(video, out_png, w, h):
    subprocess.run([FFMPEG, "-y", "-sseof", "-0.15", "-i", str(video),
                    "-vf", f"scale={w}:{h}", "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--frames", type=int, default=81, help="per segment; 81@16fps=5.06s")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--prompt", required=True)
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--out", required=True)
    p.add_argument("--lora", action="store_true",
                   help="Apply the category I2V LoRA per segment (use --frames 33 to match its 2s training)")
    p.add_argument("--no-lora", action="store_true",
                   help="Explicit base model (no LoRA) — the default; accepted so the flag doesn't error")
    p.add_argument("--category", default="fighting")
    p.add_argument("--strength", type=float, default=0.75)
    args = p.parse_args()
    category_cfg = require_category(args.category)
    gi.COMFYUI_URL = f"http://localhost:{args.port}"

    start = Path(args.image)
    tmp = Path("/tmp/extend"); tmp.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    seg_files = []
    prompt_ids = []
    for i in range(args.segments):
        name = gi.upload_image(start)
        print(f"[seg {i+1}/{args.segments}] start={start.name} -> {name}", flush=True)
        t0 = time.time()
        prefix = f"long_chain/{run_id}/seg_{i:02d}"
        wf = gi.build_workflow(args.prompt, 1000 + i, args.category, name, args.frames, args.strength,
                               prefix, width=args.width, height=args.height,
                               no_lora=not args.lora)
        pid = gi.queue_prompt(wf)
        status, entry = gi.wait_for_completion(pid, timeout=2400)
        prompt_ids.append(pid)
        seg = resolve_segment_output(entry, OUTPUT_DIR, prefix, since=t0)
        print(f"[seg {i+1}] {status} -> {seg}", flush=True)
        if status != "ok" or not seg:
            print("ABORT: segment failed"); return
        seg_files.append(seg)
        nf = tmp / f"frame_{i:02d}.png"
        extract_last_frame(seg, nf, args.width, args.height)
        start = nf

    listf = tmp / "concat.txt"
    listf.write_text("".join(f"file '{s.resolve()}'\n" for s in seg_files))
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c", "copy", "-movflags", "+faststart", str(outp)],
                   check=True, capture_output=True)
    method = "extend_chain_lora" if args.lora else "extend_chain"
    workflow_template = gi.WORKFLOW_TEMPLATE_PATH if args.lora else gi.BASE_WORKFLOW_TEMPLATE_PATH
    lora_base = category_cfg.i2v_lora_base or ""
    row = build_manifest_row(
        id=0,
        category=args.category,
        method=method,
        prompt=args.prompt,
        negative_prompt=gi.NEG_PROMPT,
        seed=1000,
        start_frame=args.image,
        frames=args.frames * args.segments,
        width=args.width,
        height=args.height,
        workflow_template=workflow_template,
        workflow_sha256=sha256_file(workflow_template),
        model_stack="wan2.2-i2v-a14b",
        lora_high=f"{lora_base}_high.safetensors" if args.lora else "",
        lora_low=f"{lora_base}_low.safetensors" if args.lora else "",
        lora_strength=args.strength if args.lora else "",
        output_path=outp,
        status="ok",
        error_reason="",
    )
    manifest_path = outp.parent / f"{outp.stem}_manifest.csv"
    row["segment_count"] = args.segments
    row["segment_frames"] = args.frames
    row["segment_paths"] = ";".join(repo_relative(s) for s in seg_files)
    row["segment_prompt_ids"] = ";".join(prompt_ids)
    write_manifest([row], manifest_path)
    dur = args.segments * args.frames / 16.0
    print(f"DONE: {args.segments} segments -> {outp} (~{dur:.1f}s)", flush=True)
    print(f"Manifest saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
