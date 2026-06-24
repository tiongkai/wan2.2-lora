from __future__ import annotations

from typing import Any


def get_nested(mapping: dict[str, Any], path: list[str]) -> Any:
    target: Any = mapping
    for part in path:
        target = target[part]
    return target


def set_nested(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    target: Any = mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def validate_patch_fields(
    workflow: dict[str, Any],
    patch_fields: dict[str, tuple[str, list[str]]],
    expected_classes: dict[str, str] | None = None,
) -> None:
    expected_classes = expected_classes or {}
    errors: list[str] = []
    for key, (node_id, field_path) in patch_fields.items():
        node = workflow.get(node_id)
        if node is None:
            errors.append(f"{key}: missing node {node_id}")
            continue
        expected = expected_classes.get(node_id)
        actual = node.get("class_type")
        if expected and actual != expected:
            errors.append(f"{key}: node {node_id} class_type {actual!r}, expected {expected!r}")
        try:
            get_nested(node, field_path)
        except KeyError:
            errors.append(f"{key}: missing field path {'.'.join(field_path)} on node {node_id}")
    if errors:
        raise ValueError("Invalid ComfyUI workflow template:\n" + "\n".join(f"- {e}" for e in errors))

