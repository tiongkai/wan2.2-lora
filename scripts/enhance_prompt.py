#!/usr/bin/env python
"""
Enhance a short idea into a detailed, scene-grounded video-generation prompt using the
abliterated Qwen2.5-VL (via Ollama). Because it's a vision model, it looks at the START
IMAGE and grounds the prompt in the actual frame — collapsing the manual
caption -> hand-craft step into one call.

Standalone (needs only `requests` + a running Ollama with the model pulled).

Usage:
  python scripts/enhance_prompt.py carpark-singapore.jpg "the yellow taxi explodes"
"""
import argparse
import base64
import requests

# Encodes the prompt-writing rules we found work best (see docs/QUICKSTART.md):
# scene-grounded, single clear action, motion-descriptive/kinematic, surveillance framing,
# and output-only-the-prompt so it drops straight into generation.
INSTRUCTION = (
    "You write prompts for an image-to-video model. Look at the image, then write ONE prompt that "
    "makes this happen: \"{idea}\".\n"
    "Rules:\n"
    "- Ground it in the ACTUAL scene you see: reuse concrete objects, colours, the setting, the "
    "camera angle/height, and the lighting.\n"
    "- Describe the action as PHYSICAL MOTION — what moves and how (kinematic, concrete steps), "
    "not an abstract verb.\n"
    "- Keep it to ONE clear action (the idea above).\n"
    "- Prefer a surveillance / security-camera framing if it suits the scene.\n"
    "Output ONLY the final prompt as a single line. No preamble, no quotes, no explanation, no markdown."
)


def enhance_prompt(image_path, idea,
                   model="huihui_ai/qwen2.5-vl-abliterated:7b",
                   url="http://localhost:11434/api/generate",
                   num_predict=220, temperature=0.6):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    r = requests.post(url, json={
        "model": model,
        "prompt": INSTRUCTION.format(idea=idea),
        "images": [img],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature},
    }, timeout=300)
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    # Collapse any newlines/extra whitespace to a single clean line, strip stray quotes.
    return " ".join(text.split()).strip('"“”')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("idea", help="short action idea, e.g. 'the yellow taxi explodes'")
    ap.add_argument("--model", default="huihui_ai/qwen2.5-vl-abliterated:7b")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    a = ap.parse_args()
    print(enhance_prompt(a.image, a.idea, a.model, a.url))
