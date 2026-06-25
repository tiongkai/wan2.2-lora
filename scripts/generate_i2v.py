# scripts/generate_i2v.py
"""
Image-to-video synthetic clip generation via ComfyUI HTTP API (Wan2.2 I2V + LoRA).
Requires ComfyUI running with the I2V models wired (see run_comfyui.sh).

Uploads a start frame, animates it with the category I2V LoRA, saves an mp4.

Usage:
  python scripts/generate_i2v.py --category fighting --image path/to/frame.jpg --count 4
  # optional: --prompt "fght99, ..."  --frames 81   --strength 0.75
"""
import json, random, time, requests, argparse
from pathlib import Path
import uuid
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.categories import CATEGORIES, enabled_categories, require_category
from scripts.manifest import (
    build_manifest_row,
    comfy_history_output_paths,
    latest_matching_output,
    repo_relative,
    sha256_file,
    write_manifest,
)
from scripts.workflow_utils import set_nested, validate_patch_fields
from scripts.enhance_prompt import enhance_prompt
from scripts.pipeline_config import config_value

COMFYUI_URL = f"http://{config_value('runtime', 'comfyui_host', default='localhost')}:{config_value('runtime', 'comfyui_port', default=8188)}"
OUTPUT_ROOT = Path(__file__).parent.parent / "generated" / "clips"
BASE_LORA_PATHS = {name: c.i2v_lora_base for name, c in CATEGORIES.items() if c.enabled and c.i2v_lora_base}

WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "comfyui_wan22_i2v_workflow.json"
LORA_STRENGTH = 0.75
NEG_PROMPT = "blurry, low quality, watermark, text, distorted, static, still image"

PATCH_FIELDS = {
    "prompt":        ("6",  ["inputs", "text"]),
    "neg_prompt":    ("7",  ["inputs", "text"]),
    "seed_high":     ("19", ["inputs", "noise_seed"]),
    "seed_low":      ("20", ["inputs", "noise_seed"]),
    "lora_high":     ("12", ["inputs", "lora_name"]),
    "lora_low":      ("13", ["inputs", "lora_name"]),
    "strength_high": ("12", ["inputs", "strength_model"]),
    "strength_low":  ("13", ["inputs", "strength_model"]),
    "image":         ("23", ["inputs", "image"]),
    "length":        ("18", ["inputs", "length"]),
    "width":         ("18", ["inputs", "width"]),
    "height":        ("18", ["inputs", "height"]),
    "filename":      ("9",  ["inputs", "filename_prefix"]),
}
EXPECTED_NODE_CLASSES = {
    "6": "CLIPTextEncode",
    "7": "CLIPTextEncode",
    "9": "SaveVideo",
    "12": "LoraLoaderModelOnly",
    "13": "LoraLoaderModelOnly",
    "18": "WanImageToVideo",
    "19": "KSamplerAdvanced",
    "20": "KSamplerAdvanced",
    "23": "LoadImage",
}


def upload_image(path: Path) -> str:
    """Upload a start frame to ComfyUI's input dir; returns the name to reference."""
    with open(path, "rb") as f:
        resp = requests.post(f"{COMFYUI_URL}/upload/image",
                             files={"image": (path.name, f, "application/octet-stream")},
                             data={"overwrite": "true"}, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    # ComfyUI returns {"name": ..., "subfolder": ..., "type": "input"}
    sub = info.get("subfolder", "")
    return f"{sub}/{info['name']}" if sub else info["name"]


BASE_WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "comfyui_wan22_i2v_base_workflow.json"


def build_workflow(prompt, seed, category, image_name, frames, strength, filename_prefix,
                   width=512, height=768, no_lora=False):
    template = BASE_WORKFLOW_TEMPLATE_PATH if no_lora else WORKFLOW_TEMPLATE_PATH
    if not template.exists():
        raise FileNotFoundError(f"Workflow template not found at {template}.")
    workflow = json.loads(template.read_text())
    patches = {
        "prompt": prompt,
        "neg_prompt": NEG_PROMPT,
        "seed_high": seed,
        "seed_low": seed,
        "image": image_name,
        "length": frames,
        "width": width,
        "height": height,
        "filename": filename_prefix,
    }
    if not no_lora:  # base workflow has no LoRA nodes to patch
        lora_base = BASE_LORA_PATHS[category]
        patches.update({
            "lora_high": f"{lora_base}_high.safetensors",
            "lora_low": f"{lora_base}_low.safetensors",
            "strength_high": strength,
            "strength_low": strength,
        })
    node_ids = {PATCH_FIELDS[key][0] for key in patches}
    expected = {node_id: cls for node_id, cls in EXPECTED_NODE_CLASSES.items() if node_id in node_ids}
    validate_patch_fields(workflow, {key: PATCH_FIELDS[key] for key in patches}, expected)
    for key, value in patches.items():
        node_id, field_path = PATCH_FIELDS[key]
        set_nested(workflow[node_id], field_path, value)
    return workflow


def queue_prompt(workflow) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id, timeout=2400) -> tuple[str, dict]:
    """Returns (status, history_entry) based on real run status + output presence."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10).json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return "error", entry
            if status.get("completed") or entry.get("outputs"):
                return ("ok" if any(entry.get("outputs", {}).values()) else "error"), entry
        time.sleep(2)
    return "timeout", {}


def load_prompts(category, prompts_file=None):
    p = Path(prompts_file) if prompts_file else Path(__file__).parent / "prompts" / f"{category}.txt"
    return [l.strip() for l in p.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def main():
    global COMFYUI_URL
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, choices=[c for c in enabled_categories() if CATEGORIES[c].i2v_lora_base])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Single start frame (conditioning image)")
    g.add_argument("--image-dir", help="Directory of start frames; cycled across samples")
    p.add_argument("--count", type=int, default=4, help="Total samples to generate")
    p.add_argument("--prompt", default=None, help="Override prompt (else uses prompts/<category>.txt)")
    p.add_argument("--prompts-file", default=None, help="File of prompts to cycle (one per line)")
    p.add_argument("--auto-prompt", default=None, metavar="IDEA",
                   help="Qwen-VL enhances this short idea into a detailed, scene-grounded prompt "
                        "per start frame (e.g. 'the yellow taxi explodes')")
    p.add_argument("--frames", type=int, default=config_value("wan", "short_frames", default=33), help="33 -> ~2s, 81 -> ~5s @16fps")
    p.add_argument("--width", type=int, default=config_value("generation", "default_width", default=512), help="must be divisible by 16")
    p.add_argument("--height", type=int, default=config_value("generation", "default_height", default=768), help="must be divisible by 16")
    p.add_argument("--strength", type=float, default=config_value("generation", "default_lora_strength", default=LORA_STRENGTH))
    p.add_argument("--no-lora", action="store_true", help="Stock Wan2.2 I2V, no LoRA (baseline)")
    p.add_argument("--port", type=int, default=8188, help="ComfyUI port to target")
    p.add_argument("--shard", type=int, default=0, help="This worker's index (0-based)")
    p.add_argument("--num-shards", type=int, default=1, help="Total workers (for dual-GPU split)")
    p.add_argument("--seed", type=int, default=None, help="Fixed base seed (else random per sample)")
    args = p.parse_args()
    category_cfg = require_category(args.category)
    COMFYUI_URL = f"http://localhost:{args.port}"

    # Collect start frames (single or directory), upload each once, keep the order stable.
    if args.image:
        frame_paths = [Path(args.image)]
    else:
        frame_paths = sorted(Path(args.image_dir).glob("*.png")) + sorted(Path(args.image_dir).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError("No start frames found.")
    uploaded = {}  # local path -> ComfyUI name (upload each distinct frame only once)

    # Prompt source: --auto-prompt (Qwen per-frame) takes precedence; else --prompt / file / library.
    enhanced = {}  # frame_path -> Qwen-enhanced prompt (cache per distinct frame)
    prompts = None if args.auto_prompt else ([args.prompt] if args.prompt else load_prompts(args.category, args.prompts_file))
    out_sub = f"{args.category}_i2v_base" if args.no_lora else f"{args.category}_i2v"
    out_dir = OUTPUT_ROOT / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # This shard handles global indices i where i % num_shards == shard.
    indices = [i for i in range(args.count) if i % args.num_shards == args.shard]
    rows = []
    for n, i in enumerate(indices):
        frame_path = frame_paths[i % len(frame_paths)]
        if frame_path not in uploaded:
            uploaded[frame_path] = upload_image(frame_path)
        image_name = uploaded[frame_path]
        if args.auto_prompt:
            if frame_path not in enhanced:
                enhanced[frame_path] = enhance_prompt(str(frame_path), args.auto_prompt)
                print(f"  [auto-prompt {frame_path.name}] {enhanced[frame_path]}", flush=True)
            prompt = enhanced[frame_path]
        else:
            prompt = prompts[i % len(prompts)]
        seed = (args.seed + i) if args.seed is not None else random.randint(0, 2**32 - 1)
        sub = f"{args.category}_i2v_base" if args.no_lora else f"{args.category}_i2v"
        prefix = f"{sub}/sample_{i:04d}"   # unique per global index -> no collisions
        wf = build_workflow(prompt, seed, args.category, image_name, args.frames, args.strength, prefix,
                            width=args.width, height=args.height, no_lora=args.no_lora)
        t0 = time.time()
        pid = queue_prompt(wf)
        status, entry = wait_for_completion(pid)
        outputs = comfy_history_output_paths(entry, OUTPUT_ROOT)
        output_path = outputs[0] if outputs else latest_matching_output(OUTPUT_ROOT, prefix, t0)
        lora_base = category_cfg.i2v_lora_base or ""
        workflow_template = BASE_WORKFLOW_TEMPLATE_PATH if args.no_lora else WORKFLOW_TEMPLATE_PATH
        rows.append(build_manifest_row(
            id=i,
            category=args.category,
            method="i2v_base" if args.no_lora else "i2v_lora",
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            seed=seed,
            start_frame=frame_path,
            frames=args.frames,
            width=args.width,
            height=args.height,
            workflow_template=workflow_template,
            workflow_sha256=sha256_file(workflow_template),
            model_stack="wan2.2-i2v-a14b",
            lora_high="" if args.no_lora else f"{lora_base}_high.safetensors",
            lora_low="" if args.no_lora else f"{lora_base}_low.safetensors",
            lora_strength="" if args.no_lora else args.strength,
            output_path=output_path,
            prompt_id=pid,
            status=status,
            error_reason=entry.get("status", {}).get("messages", "") if status != "ok" else "",
            shard=args.shard,
            num_shards=args.num_shards,
        ))
        print(f"[shard {args.shard} {n+1}/{len(indices)}] global#{i} {status} — {frame_path.name} seed {seed}", flush=True)

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}"
    write_manifest(rows, out_dir / f"{args.category}_i2v_manifest{suffix}.csv")
    print(f"Manifest saved: {out_dir}/{args.category}_i2v_manifest{suffix}.csv")


if __name__ == "__main__":
    main()
