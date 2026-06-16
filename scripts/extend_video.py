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
import argparse, time, subprocess, importlib.util
from pathlib import Path

# Reuse helpers from generate_i2v.py (upload/queue/wait/build_workflow).
_spec = importlib.util.spec_from_file_location("gi", str(Path(__file__).parent / "generate_i2v.py"))
gi = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gi)

FFMPEG = "/home/lenovo5/ffmpeg/ffmpeg-n6.1.1-31-gaa5e6017a5-linux64-lgpl-shared-6.1/bin/ffmpeg"
OUTPUT_DIR = Path(__file__).parent.parent / "generated" / "clips"


def newest_mp4(since):
    mp4s = [p for p in OUTPUT_DIR.rglob("*.mp4") if p.stat().st_mtime > since]
    return max(mp4s, key=lambda p: p.stat().st_mtime) if mp4s else None


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
    p.add_argument("--category", default="fighting")
    p.add_argument("--strength", type=float, default=0.75)
    args = p.parse_args()
    gi.COMFYUI_URL = f"http://localhost:{args.port}"

    start = Path(args.image)
    tmp = Path("/tmp/extend"); tmp.mkdir(exist_ok=True)
    seg_files = []
    for i in range(args.segments):
        name = gi.upload_image(start)
        print(f"[seg {i+1}/{args.segments}] start={start.name} -> {name}", flush=True)
        t0 = time.time()
        wf = gi.build_workflow(args.prompt, 1000 + i, args.category, name, args.frames, args.strength,
                               f"long_chain/seg_{i:02d}", width=args.width, height=args.height,
                               no_lora=not args.lora)
        pid = gi.queue_prompt(wf)
        status = gi.wait_for_completion(pid, timeout=2400)
        seg = newest_mp4(t0)
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
    dur = args.segments * args.frames / 16.0
    print(f"DONE: {args.segments} segments -> {outp} (~{dur:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
