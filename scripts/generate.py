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

COMFYUI_URL = "http://localhost:8188"

# LoRA names as ComfyUI sees them (relative to the loras dir wired in
# ComfyUI/extra_model_paths.yaml -> wan2.2-lora/loras). "_high"/"_low" +
# ".safetensors" are appended per expert.
BASE_LORA_PATHS = {
    "fighting":  "fighting/fighting_lora_r32",
    "vandalism": "vandalism/vandalism_lora_r32",
    "stabbing":  "stabbing/stabbing_lora_r32",
    "shooting":  "shooting/shooting_lora_r32",
}

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


def build_workflow(prompt: str, seed: int, category: str) -> dict:
    if not WORKFLOW_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Workflow template not found at {WORKFLOW_TEMPLATE_PATH}. "
            "Export a Wan 2.2 T2V + LoRA workflow from ComfyUI in API format."
        )
    workflow = json.loads(WORKFLOW_TEMPLATE_PATH.read_text())

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
            node = workflow[node_id]
            target = node
            for part in field_path[:-1]:
                target = target[part]
            target[field_path[-1]] = value

    return workflow


def queue_prompt(workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 900) -> str:
    """Poll history until the prompt finishes. Returns 'ok', 'error', or 'timeout'.
    Presence in history is NOT success — ComfyUI records failed/OOM runs too, so
    we inspect status.status_str and require a produced output."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        entry = resp.json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return "error"
            # success only when completed and at least one output was saved
            if status.get("completed") or entry.get("outputs"):
                has_output = any(entry.get("outputs", {}).values())
                return "ok" if has_output else "error"
        time.sleep(2)
    return "timeout"


def load_prompts(category: str) -> list[str]:
    p = Path(__file__).parent / "prompts" / f"{category}.txt"
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return lines


def generate_batch(category: str, count: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(category)
    metadata_rows = []

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        seed = random.randint(0, 2**32 - 1)
        workflow = build_workflow(prompt, seed=seed, category=category)
        prompt_id = queue_prompt(workflow)
        status = wait_for_completion(prompt_id)
        metadata_rows.append({
            "id": i, "category": category, "prompt": prompt,
            "seed": seed, "prompt_id": prompt_id, "status": status
        })
        print(f"[{i+1}/{count}] {status} — seed {seed}")

    import pandas as pd
    df = pd.DataFrame(metadata_rows)
    df.to_csv(out_dir / f"{category}_manifest.csv", index=False)
    print(f"Manifest saved: {out_dir}/{category}_manifest.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   choices=["fighting", "vandalism", "stabbing", "shooting"])
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()
    base = Path(__file__).parent.parent
    generate_batch(args.category, args.count,
                   out_dir=base / "generated" / "clips" / args.category)
