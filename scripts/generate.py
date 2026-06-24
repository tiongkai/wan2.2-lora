# scripts/generate.py
"""
Batch synthetic video generation via ComfyUI HTTP API.
Requires ComfyUI running at localhost:8188 with Wan2.2 T2V workflow loaded.

Usage:
  python scripts/generate.py --category fighting --count 50
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
    sha256_file,
    write_manifest,
)
from scripts.workflow_utils import set_nested, validate_patch_fields
from scripts.pipeline_config import config_value

COMFYUI_URL = f"http://{config_value('runtime', 'comfyui_host', default='localhost')}:{config_value('runtime', 'comfyui_port', default=8188)}"
OUTPUT_ROOT = Path(__file__).parent.parent / "generated" / "clips"

# LoRA names as ComfyUI sees them (relative to the loras dir wired in
# ComfyUI/extra_model_paths.yaml -> wan2.2-lora/loras). "_high"/"_low" +
# ".safetensors" are appended per expert.
BASE_LORA_PATHS = {name: c.t2v_lora_base for name, c in CATEGORIES.items() if c.enabled and c.t2v_lora_base}

WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "comfyui_wan22_t2v_workflow.json"
# Dual-expert (high+low noise) + dual-LoRA Wan 2.2 T2V workflow on stock models.
# Node IDs/fields below match comfyui_wan22_t2v_workflow.json, validated against
# ComfyUI /object_info. Wan 2.2 loads the umt5 (T5) text encoder as type=wan.
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
    "filename":      ("9",  ["inputs", "filename_prefix"]),
}
EXPECTED_NODE_CLASSES = {
    "6": "CLIPTextEncode",
    "7": "CLIPTextEncode",
    "9": "SaveVideo",
    "12": "LoraLoaderModelOnly",
    "13": "LoraLoaderModelOnly",
    "19": "KSamplerAdvanced",
    "20": "KSamplerAdvanced",
}


def build_workflow(prompt: str, seed: int, category: str) -> dict:
    if not WORKFLOW_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Workflow template not found at {WORKFLOW_TEMPLATE_PATH}. "
            "Export a Wan 2.2 T2V + LoRA workflow from ComfyUI in API format."
        )
    workflow = json.loads(WORKFLOW_TEMPLATE_PATH.read_text())
    validate_patch_fields(workflow, PATCH_FIELDS, EXPECTED_NODE_CLASSES)

    lora_base = BASE_LORA_PATHS[category]
    patches = {
        "prompt": prompt,
        "neg_prompt": NEG_PROMPT,
        "seed_high": seed,
        "seed_low": seed,
        "lora_high": f"{lora_base}_high.safetensors",
        "lora_low": f"{lora_base}_low.safetensors",
        "strength_high": LORA_STRENGTH,
        "strength_low": LORA_STRENGTH,
        "filename": f"{category}/synthetic_{category}",
    }
    for key, value in patches.items():
        if key in PATCH_FIELDS:
            node_id, field_path = PATCH_FIELDS[key]
            set_nested(workflow[node_id], field_path, value)

    return workflow


def queue_prompt(workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 900) -> tuple[str, dict]:
    """Poll history until the prompt finishes. Returns (status, history_entry).
    Presence in history is NOT success — ComfyUI records failed/OOM runs too, so
    we inspect status.status_str and require a produced output."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        entry = resp.json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return "error", entry
            # success only when completed and at least one output was saved
            if status.get("completed") or entry.get("outputs"):
                has_output = any(entry.get("outputs", {}).values())
                return ("ok" if has_output else "error"), entry
        time.sleep(2)
    return "timeout", {}


def load_prompts(category: str) -> list[str]:
    p = Path(__file__).parent / "prompts" / f"{category}.txt"
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return lines


def generate_batch(category: str, count: int, out_dir: Path):
    category_cfg = require_category(category)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(category)
    metadata_rows = []

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        seed = random.randint(0, 2**32 - 1)
        workflow = build_workflow(prompt, seed=seed, category=category)
        t0 = time.time()
        prompt_id = queue_prompt(workflow)
        status, entry = wait_for_completion(prompt_id)
        prefix = f"{category}/synthetic_{category}"
        outputs = comfy_history_output_paths(entry, OUTPUT_ROOT)
        output_path = outputs[0] if outputs else latest_matching_output(OUTPUT_ROOT, prefix, t0)
        lora_base = category_cfg.t2v_lora_base or ""
        metadata_rows.append(build_manifest_row(
            id=i,
            category=category,
            method="t2v",
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            seed=seed,
            frames=33,
            width="",
            height="",
            workflow_template=WORKFLOW_TEMPLATE_PATH,
            workflow_sha256=sha256_file(WORKFLOW_TEMPLATE_PATH),
            model_stack="wan2.2-t2v-a14b",
            lora_high=f"{lora_base}_high.safetensors",
            lora_low=f"{lora_base}_low.safetensors",
            lora_strength=LORA_STRENGTH,
            output_path=output_path,
            prompt_id=prompt_id,
            status=status,
            error_reason=entry.get("status", {}).get("messages", "") if status != "ok" else "",
        ))
        print(f"[{i+1}/{count}] {status} — seed {seed}")

    write_manifest(metadata_rows, out_dir / f"{category}_manifest.csv")
    print(f"Manifest saved: {out_dir}/{category}_manifest.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=[c for c in enabled_categories() if CATEGORIES[c].t2v_lora_base])
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()
    base = Path(__file__).parent.parent
    generate_batch(args.category, args.count,
                   out_dir=base / "generated" / "clips" / args.category)
