from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.manifest import load_manifests, validate_manifest_df


def main() -> None:
    p = argparse.ArgumentParser(description="Validate canonical generation manifests.")
    p.add_argument("--generated-dir", type=Path, default=Path(__file__).parent.parent / "generated")
    p.add_argument("--require-files", action="store_true")
    args = p.parse_args()

    df = load_manifests(args.generated_dir)
    if df.empty:
        raise SystemExit("No manifests found")
    errors = validate_manifest_df(df, require_files=args.require_files)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Manifest validation ok: {len(df)} rows")


if __name__ == "__main__":
    main()
