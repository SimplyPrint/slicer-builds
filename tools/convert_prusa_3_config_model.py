"""Convert PrusaSlicer 3's config model to SimplyPrint UI artifacts."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

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
    "FloatOrPercentage": "floatorpercent",
    "Percentage": "percent",
    "Enum": "enum",
}
ARRAY_TYPES = {
    **{key: f"{value}s" for key, value in SCALAR_TYPES.items()},
    "FloatOrPercentage": "floatorpercents",
}
def _words(value: str, prefix: str = "") -> str:
    value = value.removeprefix(f"{prefix}_") if prefix else value
    return (
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value.replace("_", " ")).strip()
        or "Other"
    )


def _value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, dict) and "value" in value:
        return _value(value["value"]) + ("%" if value.get("is_percent") else "")
    if isinstance(value, list):
        if value and all(
            isinstance(point, list) and len(point) == 2 for point in value
        ):
            return ",".join(f"{_value(x)}x{_value(y)}" for x, y in value)
        return ",".join(_value(item) for item in value if item is not None)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _items(model: dict, boxes: tuple[str, ...]):
    seen = set()
    for box_name in boxes:
        box = model.get(box_name)
        if not isinstance(box, dict):
            raise TypeError(f"config model has no {box_name!r} box")
        for collection in ("items", "overrides"):
            for item in box.get(collection, []):
                name = item.get("name") if isinstance(item, dict) else None
                if not isinstance(name, str):
                    raise TypeError(f"invalid item in {box_name}.{collection}")
                if name not in seen:
                    seen.add(name)
                    yield item


def _definition(item: dict) -> dict:
    array = item.get("type") == "Array"
    type_name = item.get("element_type") if array else item.get("type")
    result = {
        "type": (ARRAY_TYPES if array else SCALAR_TYPES).get(
            str(type_name), "strings" if array else "string"
        ),
        "label": item.get("label") or _words(item["name"]).title(),
        "tooltip": item.get("tooltip", ""),
        "mode": "simple",
    }
    if array and type_name == "String":
        result["native_type"] = "array"
        result["editor"] = {"kind": "list", "value_type": "string"}
    for key in ("full_label", "gui_type"):
        if item.get(key) not in (None, "", "undefined"):
            result[key] = item[key]
    if item.get("units"):
        result["sidetext"] = item["units"][0]
    for bound in ("min", "max"):
        if bound in item:
            result[bound] = _value(item[bound])
    if item.get("cli") == "~~nocli":
        result["nocli"] = "true"
    values = item.get("enum_values") or item.get("choice_values")
    labels = item.get("choice_labels")
    if isinstance(values, list):
        result["enum_values"] = values
        if isinstance(labels, list) and len(labels) == len(values):
            result["enum_labels"] = labels
    default = item.get("value")
    if (
        result["type"] == "enum"
        and isinstance(values, list)
        and isinstance(labels, list)
    ):
        try:
            default = labels[values.index(default)]
        except ValueError:
            pass
    result["default_value"] = default if array or type_name == "Vec2d" else _value(default)
    return result


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sort_key(item: dict) -> tuple[int, int, int, str]:
    return (
        int(item.get("category_order", 65535)),
        int(item.get("option_group_order", 65535)),
        int(item.get("order", 0)),
        item["name"],
    )


def _icon(icon_dir: Path | None, icon_name: str) -> dict | None:
    if icon_dir is None or not icon_name:
        return None
    path = icon_dir / f"{icon_name}.svg"
    if not path.is_file():
        return None
    return {
        "source_name": path.name,
        "svg": path.read_text(encoding="utf-8"),
    }


def convert(model_path: Path, output: Path, icon_dir: Path | None = None) -> None:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise TypeError("config model root must be an object")
    definitions = {}
    panels = {}
    icons = {}
    for artifact, boxes in BOXES.items():
        navigation = {}
        panel = []
        categories = {}
        for item in sorted(_items(model, boxes), key=_sort_key):
            definitions.setdefault(item["name"], _definition(item))
            category_name = str(item.get("category", "Unknown"))
            if category_name == "Hidden":
                continue
            role = artifact.removesuffix(".json")
            prefix = {"machine": "Printer", "process": "Print", "filament": "Filament"}[
                role
            ]
            category = item.get("category_label") or _words(category_name, prefix)
            group_name = str(item.get("option_group", "Unknown"))
            group = item.get("option_group_label") or _words(group_name, category_name)
            navigation.setdefault(category, {}).setdefault(group, []).append(
                item["name"]
            )
            category_entry = categories.get(category_name)
            if category_entry is None:
                icon_name = str(item.get("category_icon", ""))
                icon_id = f"prusa:{icon_name}" if icon_name else None
                icon_value = _icon(icon_dir, icon_name)
                if icon_id and icon_value:
                    icons.setdefault(icon_id, icon_value)
                category_entry = {
                    "id": category_name,
                    "label": category,
                    "groups": [],
                }
                if icon_id in icons:
                    category_entry["icon"] = icon_id
                categories[category_name] = category_entry
                panel.append(category_entry)
            groups = category_entry["groups"]
            group_entry = next((entry for entry in groups if entry["id"] == group_name), None)
            if group_entry is None:
                group_entry = {"id": group_name, "label": group, "setting_keys": []}
                groups.append(group_entry)
            group_entry["setting_keys"].append(item["name"])
        _write(output / artifact, navigation)
        panels[role] = panel
    _write(output / "print_config_def.json", definitions)
    _write(
        output / "ui_metadata.json",
        {
            "schema_version": 1,
            "capabilities": {"profile_metadata_transport": "envelope.v1"},
            "settings_codec": "prusa-domain-config.v1",
            "runtime_setting_scopes": {
                name: "global" for name in sorted(definitions)
            },
            "panels": panels,
            "icons": icons,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--icons", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    convert(args.model, args.output, args.icons)


if __name__ == "__main__":
    main()
