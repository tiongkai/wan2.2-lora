#!/usr/bin/env python
"""
Caption a single image with the abliterated Qwen2.5-VL (via Ollama), to extract scene
context for grounding a video-generation prompt. Standalone client — needs only `requests`
and a running Ollama with the model pulled.

Usage:
  python scripts/caption_image.py carpark-singapore.jpg
  python scripts/caption_image.py img.jpg --prompt "List every car and its colour/position."
"""
import argparse, base64, requests

DEFAULT_PROMPT = (
    "Describe this scene for a video-generation prompt. Cover: (1) the setting/location and "
    "notable objects/signage, (2) the main subjects (people or vehicles) with position, colour, "
    "and clothing, (3) the camera angle/height, (4) the lighting and mood. Be concrete and concise."
)


def caption(image_path, prompt=DEFAULT_PROMPT,
            model="huihui_ai/qwen3.5-abliterated:9b",
            url="http://localhost:11434/api/generate", num_predict=400):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    r = requests.post(url, json={"model": model, "prompt": prompt, "images": [img],
                                 "stream": False, "think": False, "options": {"num_predict": num_predict}},
                      timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default="huihui_ai/qwen3.5-abliterated:9b")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    a = ap.parse_args()
    print(caption(a.image, a.prompt, a.model, a.url))
