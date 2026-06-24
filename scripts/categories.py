from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    name: str
    class_id: int
    trigger: str
    prompt_file: str
    enabled: bool = True
    t2v_lora_base: str | None = None
    i2v_lora_base: str | None = None


CATEGORIES: dict[str, Category] = {
    "fighting": Category(
        name="fighting",
        class_id=0,
        trigger="fght99",
        prompt_file="fighting.txt",
        t2v_lora_base="fighting/fighting_lora_r32",
        i2v_lora_base="fighting_i2v/fighting_i2v_lora_r32",
    ),
    "vandalism": Category(
        name="vandalism",
        class_id=1,
        trigger="vndl77",
        prompt_file="vandalism.txt",
        t2v_lora_base="vandalism/vandalism_lora_r32",
        i2v_lora_base="vandalism_i2v/vandalism_i2v_lora_r32",
    ),
    "stabbing": Category(
        name="stabbing",
        class_id=2,
        trigger="stbb44",
        prompt_file="stabbing.txt",
        t2v_lora_base="stabbing/stabbing_lora_r32",
        i2v_lora_base="stabbing_i2v/stabbing_i2v_lora_r32",
    ),
    "shooting": Category(
        name="shooting",
        class_id=3,
        trigger="shtn22",
        prompt_file="shooting.txt",
        t2v_lora_base="shooting/shooting_lora_r32",
        i2v_lora_base="shooting_i2v/shooting_i2v_lora_r32",
    ),
    "self_injury": Category(
        name="self_injury",
        class_id=4,
        trigger="slfh55",
        prompt_file="self_injury_prison.txt",
        t2v_lora_base=None,
        i2v_lora_base="self_injury_i2v/self_injury_i2v_lora_r32",
    ),
}


CLASS_MAP = {name: category.class_id for name, category in CATEGORIES.items() if category.enabled}
TRIGGERS = {name: category.trigger for name, category in CATEGORIES.items() if category.enabled}


def enabled_categories() -> list[str]:
    return [name for name, category in CATEGORIES.items() if category.enabled]


def require_category(name: str) -> Category:
    try:
        category = CATEGORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported category: {name}") from exc
    if not category.enabled:
        raise ValueError(f"Category is disabled: {name}")
    return category

