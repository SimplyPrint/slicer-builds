#!/usr/bin/env python3
"""Convert PrusaSlicer 3's native config model to the legacy web artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


BOXES = {
    "machine.json": ("printer",),
    "process.json": ("print", "tool_print"),
    "filament.json": ("filament",),
}

SCALAR_TYPES = {
    "Bool": "bool",
    "Int": "int",
    "OptInt": "int",
    "Float": "float",
    "String": "string",
    "Vec2d": "point",
    "FloatOrPercentage": "float_or_percent",
    "Percentage": "percent",
    "Enum": "enum",
}

ARRAY_TYPES = {
    "Bool": "bools",
    "Int": "ints",
    "OptInt": "ints",
    "Float": "floats",
    "String": "strings",
    "Vec2d": "points",
    "FloatOrPercentage": "floats_or_percents",
    "Percentage": "percents",
    "Enum": "enums",
}


def _words(value: str, prefix: str = "") -> str:
    if prefix and value.startswith(prefix + "_"):
        value = value[len(prefix) + 1 :]
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return value.strip() or "Other"


def _legacy_type(item: dict[str, Any]) -> str:
    type_name = item.get("type")
    if type_name == "Array":
        return ARRAY_TYPES.get(str(item.get("element_type")), "strings")
    return SCALAR_TYPES.get(str(type_name), "string")


def _number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _legacy_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "value" in value:
        suffix = "%" if value.get("is_percent") else ""
        return _number(value["value"]) + suffix
    if isinstance(value, list):
        if value and all(isinstance(point, list) and len(point) == 2 for point in value):
            return ",".join(f"{_number(point[0])}x{_number(point[1])}" for point in value)
        return ",".join(_legacy_value(part) for part in value if part is not None)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _items(model: dict[str, Any], boxes: Iterable[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for box_name in boxes:
        box = model.get(box_name)
        if not isinstance(box, dict):
            raise ValueError(f"config model has no {box_name!r} box")
        for collection in ("items", "overrides"):
            values = box.get(collection, [])
            if not isinstance(values, list):
                raise ValueError(f"{box_name}.{collection} must be an array")
            for item in values:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise ValueError(f"invalid item in {box_name}.{collection}")
                if item["name"] not in seen:
                    result.append(item)
                    seen.add(item["name"])
    return result


def navigation(items: Iterable[dict[str, Any]], role: str) -> dict[str, Any]:
    result: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    prefix = {"machine": "Printer", "process": "Print", "filament": "Filament"}[role]
    for item in sorted(items, key=lambda value: (value.get("category", ""), value.get("order", 0), value["name"])):
        category_name = str(item.get("category", "Unknown"))
        if category_name == "Hidden":
            continue
        category = _words(category_name, prefix)
        group = _words(str(item.get("option_group", "Unknown")), category_name)
        result.setdefault(category, OrderedDict()).setdefault(group, []).append(item["name"])
    return result


def definition(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": _legacy_type(item),
        "label": item.get("label") or item["name"].replace("_", " ").title(),
        "tooltip": item.get("tooltip", ""),
    }
    for source, destination in (("full_label", "full_label"), ("gui_type", "gui_type")):
        value = item.get(source)
        if value and value != "undefined":
            result[destination] = value
    units = item.get("units")
    if isinstance(units, list) and units:
        result["sidetext"] = units[0]
    for bound in ("min", "max"):
        if bound in item:
            result[bound] = _number(item[bound])
    if item.get("cli") == "~~nocli":
        result["nocli"] = "true"

    enum_values = item.get("enum_values") or item.get("choice_values")
    enum_labels = item.get("choice_labels")
    if isinstance(enum_values, list):
        result["enum_values"] = enum_values
        if isinstance(enum_labels, list) and len(enum_labels) == len(enum_values):
            result["enum_labels"] = enum_labels

    result["mode"] = "simple"
    default_value = item.get("value")
    if result["type"] == "enum" and isinstance(enum_values, list) and isinstance(enum_labels, list):
        try:
            default_value = enum_labels[enum_values.index(default_value)]
        except ValueError:
            pass
    result["default_value"] = _legacy_value(default_value)
    return result


def convert(model_path: Path, output: Path) -> None:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ValueError("config model root must be an object")

    all_items: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for artifact, boxes in BOXES.items():
        items = _items(model, boxes)
        role = artifact.removesuffix(".json")
        (output / artifact).write_text(
            json.dumps(navigation(items, role), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for item in items:
            all_items.setdefault(item["name"], item)

    definitions = {name: definition(item) for name, item in all_items.items()}
    (output / "print_config_def.json").write_text(
        json.dumps(definitions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    convert(args.model, args.output)


if __name__ == "__main__":
    main()
