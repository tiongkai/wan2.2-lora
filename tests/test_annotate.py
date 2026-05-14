import pytest, json
from pathlib import Path


SAMPLE_MANIFEST = [
    {"id": 0, "category": "fighting", "prompt": "fght99, two men fighting",
     "seed": 12345, "prompt_id": "abc-1", "status": "ok"},
    {"id": 1, "category": "fighting", "prompt": "fght99, brawl in subway",
     "seed": 67890, "prompt_id": "abc-2", "status": "ok"},
]

CLASS_MAP = {"fighting": 0, "vandalism": 1, "stabbing": 2, "shooting": 3}


def test_csv_has_required_columns(tmp_path):
    import pandas as pd
    from scripts.annotate import build_annotation_csv
    df = pd.DataFrame(SAMPLE_MANIFEST)
    result = build_annotation_csv(df, class_map=CLASS_MAP)
    for col in ["filename", "category", "class_id", "prompt", "seed"]:
        assert col in result.columns


def test_coco_json_structure(tmp_path):
    import pandas as pd
    from scripts.annotate import build_coco_json
    df = pd.DataFrame(SAMPLE_MANIFEST)
    coco = build_coco_json(df, class_map=CLASS_MAP, clips_dir=tmp_path)
    assert "categories" in coco
    assert "annotations" in coco
    assert "videos" in coco
    assert len(coco["categories"]) == 4


def test_class_ids_correct(tmp_path):
    import pandas as pd
    from scripts.annotate import build_annotation_csv
    df = pd.DataFrame(SAMPLE_MANIFEST)
    result = build_annotation_csv(df, class_map=CLASS_MAP)
    assert result.iloc[0]["class_id"] == 0  # fighting = 0
