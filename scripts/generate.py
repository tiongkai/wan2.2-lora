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

BASE_LORA_PATHS = {
    "fighting":  "../../wan2.2-lora/loras/fighting/fighting_lora_r32",
    "vandalism": "../../wan2.2-lora/loras/vandalism/vandalism_lora_r32",
    "stabbing":  "../../wan2.2-lora/loras/stabbing/stabbing_lora_r32",
    "shooting":  "../../wan2.2-lora/loras/shooting/shooting_lora_r32",
}

WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "comfyui_wan22_t2v_workflow.json"
# Export a working Wan 2.2 T2V + LoRA workflow from ComfyUI as API JSON.
# Wan 2.2 uses a T5-based text encoder, NOT CLIP.
# Node IDs below are placeholders — update them to match YOUR workflow.

PATCH_FIELDS = {
    "prompt":            ("6", ["inputs", "text"]),
    "neg_prompt":        ("7", ["inputs", "text"]),
    "seed":              ("3", ["inputs", "seed"]),
    "lora_path_high":    ("4", ["inputs", "lora_path"]),
    "lora_strength_high":("4", ["inputs", "strength"]),
    "lora_path_low":     ("10", ["inputs", "lora_path"]),
    "lora_strength_low": ("10", ["inputs", "strength"]),
    "filename":          ("9", ["inputs", "filename_prefix"]),
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
        "neg_prompt": "blurry, low quality, watermark, text, distorted",
        "seed": seed,
        "lora_path_high": f"{lora_base}_high.safetensors",
        "lora_strength_high": 0.75,
        "lora_path_low": f"{lora_base}_low.safetensors",
        "lora_strength_low": 0.75,
        "filename": f"synthetic_{category}",
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


def wait_for_completion(prompt_id: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        history = resp.json()
        if prompt_id in history:
            return True
        time.sleep(2)
    return False


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
        success = wait_for_completion(prompt_id)
        status = "ok" if success else "timeout"
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
