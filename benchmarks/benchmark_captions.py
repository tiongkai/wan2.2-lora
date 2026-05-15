"""
Benchmark caption quality: Qwen2.5-VL 7B (Ollama) vs Gemma 4 E4B OBLITERATED (transformers).

Usage: PYTHONPATH=. uv run python benchmarks/benchmark_captions.py
"""
import time, json
from pathlib import Path
from PIL import Image
from scripts.caption import extract_multi_keyframes, build_caption, CAPTION_SYSTEM, CATEGORY_CONTEXT
from scripts.utils import probe_duration

TEST_CLIPS = [
    "1dsLuL5Lvbc_1",
    "4RqDG4I4iAk_2",
    "8pKKdS7rhXo_0",
    "HVSv9sgz2ZI_0",
    "kExp3yhspR0_2",
    "r8PX3ZJ1_II_0",
]

TRIGGER = "fght99"
CATEGORY = "fighting"
CLIP_DIR = Path("datasets/processed/fighting")
TEMPORAL_WORDS = {"then", "next", "after", "before", "begins", "starts", "moves",
                  "falls", "pushes", "pulls", "grabs", "hits", "kicks", "throws",
                  "approaches", "retreats", "turns", "walks", "runs", "lunges",
                  "swings", "continues", "suddenly", "sequence", "first", "finally"}


def score_caption(caption: str) -> dict:
    words = caption.lower().split()
    temporal_hits = [w for w in words if w.strip(".,;:") in TEMPORAL_WORDS]
    return {
        "word_count": len(words),
        "has_trigger": caption.lower().startswith(TRIGGER.lower()),
        "temporal_words": len(temporal_hits),
        "temporal_examples": temporal_hits[:5],
        "is_fallback": caption.strip() == f"{TRIGGER}, security camera footage",
    }


def make_prompt(n_frames: int, clip_duration: float) -> str:
    return (
        f"These {n_frames} frames are sampled sequentially from a {clip_duration:.1f}s security camera video. "
        f"Trigger word: '{TRIGGER}'. "
        f"Context: {CATEGORY_CONTEXT[CATEGORY]} "
        "Describe the complete action sequence starting with the trigger word."
    )


# --- Qwen via Ollama ---
def query_qwen(frames: list[Path], clip_duration: float) -> str:
    import base64, requests
    prompt = make_prompt(len(frames), clip_duration)
    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": prompt,
        "system": CAPTION_SYSTEM,
        "images": [base64.b64encode(f.read_bytes()).decode() for f in frames],
        "stream": False,
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"].strip()


# --- Gemma 4 via transformers ---
_gemma_model = None
_gemma_processor = None

def _load_gemma():
    global _gemma_model, _gemma_processor
    if _gemma_model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print("  Loading Gemma 4 OBLITERATED...")
        _gemma_processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
        _gemma_model = AutoModelForImageTextToText.from_pretrained(
            "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
            dtype=torch.bfloat16, device_map="auto",
        )
        print(f"  Gemma loaded, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

def query_gemma(frames: list[Path], clip_duration: float) -> str:
    import torch
    _load_gemma()
    images = [Image.open(f) for f in frames]
    prompt = make_prompt(len(frames), clip_duration)
    full_prompt = f"{CAPTION_SYSTEM}\n\n{prompt}"
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": full_prompt})
    messages = [{"role": "user", "content": content}]
    inputs = _gemma_processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(_gemma_model.device)
    with torch.no_grad():
        out = _gemma_model.generate(**inputs, max_new_tokens=150)
    response = _gemma_processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def benchmark_model(name: str, query_fn, clips: list[str]) -> list[dict]:
    results = []
    for clip_name in clips:
        clip_path = CLIP_DIR / f"{clip_name}.mp4"
        if not clip_path.exists():
            print(f"  SKIP {clip_name}")
            continue
        frames = []
        try:
            frames = extract_multi_keyframes(clip_path, n_frames=5)
            duration = probe_duration(clip_path)
            start = time.time()
            raw = query_fn(frames, duration)
            elapsed = time.time() - start
            caption = build_caption(raw, trigger=TRIGGER)
            scores = score_caption(caption)
            results.append({"clip": clip_name, "caption": caption,
                            "time_s": round(elapsed, 1), "success": True, **scores})
            print(f"  OK   {clip_name} — {elapsed:.0f}s, {scores['word_count']}w, "
                  f"{scores['temporal_words']} temporal")
        except Exception as e:
            results.append({"clip": clip_name, "caption": str(e)[:100],
                            "time_s": 0, "success": False, "word_count": 0,
                            "has_trigger": False, "temporal_words": 0,
                            "temporal_examples": [], "is_fallback": False})
            print(f"  FAIL {clip_name} — {e}")
        finally:
            for f in frames:
                f.unlink(missing_ok=True)
    return results


def print_summary(label: str, results: list[dict]):
    total = len(results)
    ok = [r for r in results if r["success"]]
    sr = len(ok) / total * 100 if total else 0
    avg_t = sum(r["time_s"] for r in ok) / len(ok) if ok else 0
    avg_w = sum(r["word_count"] for r in ok) / len(ok) if ok else 0
    avg_tmp = sum(r["temporal_words"] for r in ok) / len(ok) if ok else 0
    trig = sum(1 for r in ok if r["has_trigger"]) / len(ok) * 100 if ok else 0
    fb = sum(1 for r in results if r["is_fallback"])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Success rate:       {len(ok)}/{total} ({sr:.0f}%)")
    print(f"  Fallback captions:  {fb}/{total}")
    print(f"  Avg response time:  {avg_t:.1f}s")
    print(f"  Avg word count:     {avg_w:.0f}")
    print(f"  Avg temporal words: {avg_tmp:.1f}")
    print(f"  Trigger compliance: {trig:.0f}%")
    if ok:
        best = max(ok, key=lambda r: r["temporal_words"])
        print(f"\n  Best caption [{best['clip']}]:")
        print(f"    {best['caption'][:300]}")


if __name__ == "__main__":
    models = [
        ("Qwen 2.5 VL 7B (Ollama)", query_qwen),
        ("Gemma 4 E4B OBLITERATED (transformers)", query_gemma),
    ]
    all_results = {}
    for label, fn in models:
        print(f"\n--- Benchmarking: {label} ---")
        results = benchmark_model(label, fn, TEST_CLIPS)
        all_results[label] = results
        print_summary(label, results)

    out = Path("benchmarks/results.json")
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nRaw results saved to {out}")
