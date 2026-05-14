import json
import pandas as pd
from pathlib import Path

CLASS_MAP = {"fighting": 0, "vandalism": 1, "stabbing": 2, "shooting": 3}


def build_annotation_csv(df: pd.DataFrame, class_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["class_id"] = df["category"].map(class_map)
    df["filename"] = df.apply(
        lambda r: f"synthetic_{r['category']}_{r['id']:05d}.mp4", axis=1
    )
    return df[["filename", "category", "class_id", "prompt", "seed", "status"]]


def build_coco_json(df: pd.DataFrame, class_map: dict, clips_dir: Path) -> dict:
    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    videos, annotations = [], []
    for _, row in df.iterrows():
        vid_id = int(row["id"])
        fname = f"synthetic_{row['category']}_{vid_id:05d}.mp4"
        videos.append({"id": vid_id, "file_name": fname,
                        "category": row["category"]})
        annotations.append({
            "id": vid_id, "video_id": vid_id,
            "category_id": class_map[row["category"]],
            "prompt": row["prompt"], "seed": int(row["seed"]),
        })
    return {"categories": categories, "videos": videos, "annotations": annotations}


def export_dataset(generated_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for category in CLASS_MAP:
        manifest = generated_dir / "clips" / category / f"{category}_manifest.csv"
        if manifest.exists():
            all_dfs.append(pd.read_csv(manifest))

    if not all_dfs:
        raise FileNotFoundError("No manifest CSVs found. Run generate.py first.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[combined["status"] == "ok"]
    combined["id"] = range(len(combined))  # globally unique IDs

    csv_df = build_annotation_csv(combined, CLASS_MAP)
    csv_df.to_csv(out_dir / "annotations.csv", index=False)
    print(f"Saved annotations.csv ({len(csv_df)} clips)")

    coco = build_coco_json(combined, CLASS_MAP, clips_dir=generated_dir / "clips")
    (out_dir / "annotations_coco.json").write_text(json.dumps(coco, indent=2))
    print(f"Saved annotations_coco.json")

    print("\nClass distribution:")
    print(csv_df["category"].value_counts().to_string())


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    export_dataset(
        generated_dir=base / "generated",
        out_dir=base / "generated" / "annotations"
    )
