import json
import os
from pathlib import Path

import pandas as pd
import pytest


def test_manifest_schema_and_legacy_coercion(tmp_path):
    from scripts.manifest import CANONICAL_COLUMNS, coerce_manifest, validate_manifest_df

    df = pd.DataFrame([{
        "id": 0,
        "category": "fighting",
        "prompt": "fght99, fight",
        "seed": 1,
        "status": "ok",
        "output_path": "generated/clips/fighting_i2v_base/sample.mp4",
    }])
    df["manifest_path"] = "generated/clips/fighting_i2v_base/fighting_i2v_manifest.csv"
    coerced = coerce_manifest(df)
    for column in CANONICAL_COLUMNS:
        assert column in coerced.columns
    assert coerced.iloc[0]["method"] == "i2v_base"
    assert validate_manifest_df(coerced) == []


def test_export_discovers_manifest_and_uses_real_output_path(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    clip = generated / "clips" / "fighting_i2v_base" / "sample_0000.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"not a real video but exists")
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="i2v_base",
        prompt="fght99, fight",
        seed=1,
        frames=81,
        output_path=clip,
        status="ok",
    )
    write_manifest([row], generated / "clips" / "fighting_i2v_base" / "fighting_i2v_manifest.csv")
    out = generated / "annotations"
    exported = export_dataset(generated, out, dry_run=True)
    assert len(exported) == 1
    assert exported.iloc[0]["filename"].endswith("sample_0000.mp4")


def test_full_export_allows_empty_seed(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    clip = generated / "clips" / "svi" / "manual.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"exists")
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="svi",
        prompt="fght99, manual long clip",
        seed="",
        frames=81,
        output_path=clip,
        status="ok",
    )
    write_manifest([row], generated / "clips" / "svi" / "manual_manifest.csv")

    out = generated / "annotations"
    export_dataset(generated, out, dry_run=False)

    annotations = pd.read_csv(out / "annotations.csv")
    coco = json.loads((out / "annotations_coco.json").read_text())
    assert len(annotations) == 1
    assert coco["annotations"][0]["seed"] == ""


def test_export_fails_when_ok_row_file_missing(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="i2v_base",
        prompt="fght99, fight",
        seed=1,
        frames=81,
        output_path=tmp_path / "missing.mp4",
        status="ok",
    )
    write_manifest([row], generated / "clips" / "fighting_i2v_base" / "manifest.csv")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        export_dataset(generated, generated / "annotations", dry_run=True)


def test_workflow_validation_checks_class_type():
    from scripts.workflow_utils import validate_patch_fields

    workflow = {"6": {"class_type": "Wrong", "inputs": {"text": ""}}}
    with pytest.raises(ValueError, match="class_type"):
        validate_patch_fields(workflow, {"prompt": ("6", ["inputs", "text"])}, {"6": "CLIPTextEncode"})


def test_current_workflow_templates_validate():
    from scripts.generate import build_workflow as build_t2v
    from scripts.generate_i2v import build_workflow as build_i2v

    build_t2v("fght99, test", 1, "fighting")
    build_i2v("fght99, test", 1, "fighting", "sample.png", 33, 0.75, "tmp/test", no_lora=False)
    build_i2v("fght99, test", 1, "fighting", "sample.png", 81, 0.75, "tmp/test", no_lora=True)


def test_wan_frame_validation_cases():
    from scripts.preprocess import validate_target_frames

    validate_target_frames(5.0, 33)
    validate_target_frames(5.0, 77)
    with pytest.raises(ValueError, match="Max valid target is 77"):
        validate_target_frames(5.0, 81)


def test_long_mode_windows_adds_window_metadata(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    clip = generated / "clips" / "svi" / "long.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"exists")
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="svi",
        prompt="fght99, fight",
        seed=1,
        frames=200,
        output_path=clip,
        status="ok",
    )
    write_manifest([row], generated / "clips" / "svi" / "long_manifest.csv")
    exported = export_dataset(generated, generated / "annotations", dry_run=True, long_mode="windows", window_frames=81)
    assert len(exported) == 3
    assert {"source_video", "window_start_frame", "window_end_frame"}.issubset(exported.columns)


def test_long_mode_windows_writes_coco_window_metadata(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    clip = generated / "clips" / "svi" / "long.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"exists")
    row = build_manifest_row(
        id=0,
        category="fighting",
        method="svi",
        prompt="fght99, fight",
        seed=1,
        frames=200,
        output_path=clip,
        status="ok",
    )
    write_manifest([row], generated / "clips" / "svi" / "long_manifest.csv")

    out = generated / "annotations"
    export_dataset(generated, out, dry_run=False, long_mode="windows", window_frames=81)

    csv_df = pd.read_csv(out / "annotations.csv")
    coco = json.loads((out / "annotations_coco.json").read_text())
    assert list(csv_df["window_index"]) == [0, 1, 2]
    assert list(csv_df["window_start_frame"]) == [0, 81, 162]
    assert list(csv_df["window_end_frame"]) == [81, 162, 200]
    assert [video["window_index"] for video in coco["videos"]] == [0, 1, 2]
    assert [video["window_start_frame"] for video in coco["videos"]] == [0, 81, 162]
    assert [video["window_end_frame"] for video in coco["videos"]] == [81, 162, 200]
    assert all(video["source_video"] == csv_df.iloc[0]["source_video"] for video in coco["videos"])


def test_export_requires_explicit_qa_pass_when_report_exists(tmp_path):
    from scripts.annotate import export_dataset
    from scripts.manifest import build_manifest_row, write_manifest

    generated = tmp_path / "generated"
    rows = []
    clips = {}
    for idx, name in enumerate(["passed", "failed", "missing_qa"]):
        clip = generated / "clips" / "fighting_i2v_base" / f"{name}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"exists")
        clips[name] = clip
        rows.append(build_manifest_row(
            id=idx,
            category="fighting",
            method="i2v_base",
            prompt=f"fght99, {name}",
            seed=idx,
            frames=81,
            output_path=clip,
            status="ok",
        ))
    write_manifest(rows, generated / "clips" / "fighting_i2v_base" / "manifest.csv")
    report_dir = generated / "annotations"
    report_dir.mkdir(parents=True)
    pd.DataFrame([
        {"output_path": str(clips["passed"]), "valid": True, "reason": "ok"},
        {"output_path": str(clips["failed"]), "valid": False, "reason": "black frames"},
    ]).to_csv(report_dir / "validation_report.csv", index=False)

    strict = export_dataset(generated, report_dir, dry_run=True)
    exploratory = export_dataset(generated, report_dir, dry_run=True, allow_unvalidated=True)

    assert list(strict["filename"]) == [str(clips["passed"])]
    assert sorted(Path(p).stem for p in exploratory["filename"]) == ["missing_qa", "passed"]


def test_extend_video_resolves_segment_from_history(tmp_path):
    from scripts.extend_video import resolve_segment_output

    output_root = tmp_path / "generated" / "clips"
    expected = output_root / "long_chain" / "run123" / "seg_00_00001.mp4"
    unrelated = output_root / "other" / "newer.mp4"
    expected.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    expected.write_bytes(b"segment")
    unrelated.write_bytes(b"unrelated")
    history_entry = {
        "outputs": {
            "9": {
                "gifs": [
                    {"filename": "seg_00_00001.mp4", "subfolder": "long_chain/run123"},
                    {"filename": "newer.mp4", "subfolder": "other"},
                ]
            }
        }
    }

    resolved = resolve_segment_output(history_entry, output_root, "long_chain/run123/seg_00")
    assert resolved == expected


def test_extend_video_fallback_ignores_unrelated_newer_files(tmp_path):
    from scripts.extend_video import resolve_segment_output

    output_root = tmp_path / "generated" / "clips"
    expected = output_root / "long_chain" / "run123" / "seg_01_00001.mp4"
    unrelated = output_root / "long_chain" / "run123" / "seg_99_00001.mp4"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"segment")
    unrelated.write_bytes(b"unrelated")
    os.utime(expected, (1001, 1001))
    os.utime(unrelated, (1002, 1002))

    resolved = resolve_segment_output({}, output_root, "long_chain/run123/seg_01", since=1000)
    assert resolved == expected


def test_enabled_categories_have_required_config():
    from scripts.categories import CATEGORIES

    for name, category in CATEGORIES.items():
        if not category.enabled:
            continue
        assert category.class_id >= 0, name
        assert category.trigger, name
        assert category.prompt_file, name
    assert "self_injury" in CATEGORIES
    assert CATEGORIES["self_injury"].enabled


def test_pipeline_config_loads_and_matches_categories():
    from scripts.categories import CATEGORIES
    from scripts.pipeline_config import load_pipeline_config

    cfg = load_pipeline_config()
    assert cfg["wan"]["fps"] == 16
    assert set(CATEGORIES).issubset(set(cfg["categories"]))
