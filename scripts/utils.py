import subprocess, json
from pathlib import Path


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True
    )
    streams = json.loads(result.stdout).get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    return float(video_streams[0]["duration"]) if video_streams else 0.0
