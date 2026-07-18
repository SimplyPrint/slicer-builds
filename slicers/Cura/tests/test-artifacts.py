#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
resources = Path(sys.argv[2])
definitions = json.loads((root / "print_config_def.json").read_text())
allowed_types = {
    "bool",
    "enum",
    "float",
    "int",
    "ints",
    "points",
    "pointsgroups",
    "string",
}

source_definitions: dict[str, dict] = {}
source_runtime_only: dict[str, bool] = {}
source_files: list[str] = []


def collect_source_settings(settings: dict, runtime_only: bool) -> None:
    for key, definition in settings.items():
        if definition.get("type") != "category":
            source_definitions.setdefault(key, definition)
            source_runtime_only[key] = (
                source_runtime_only.get(key, True) and runtime_only
            )
        children = definition.get("children")
        if isinstance(children, dict):
            collect_source_settings(children, runtime_only)


def expected_default(source: dict, normalized_type: str):
    value = source.get("default_value", source.get("value"))
    if normalized_type in {"int", "float"} and isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    if normalized_type == "ints" and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if normalized_type == "ints" and isinstance(value, list):
        return [int(item) for item in value]
    return value


def source_order(item: tuple[Path, dict]) -> tuple[bool, str]:
    path, document = item
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return metadata.get("type") != "machine", path.name


documents = [
    (path, json.loads(path.read_text()))
    for path in sorted((resources / "definitions").glob("*.def.json"))
]
documents.sort(key=source_order)
for path, document in documents:
    settings = document.get("settings")
    if not isinstance(settings, dict):
        continue
    source_files.append(path.name)
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    runtime_only = (
        metadata.get("visible") is False and metadata.get("type") != "machine"
    )
    collect_source_settings(settings, runtime_only)

assert source_definitions
assert set(definitions) == set(source_definitions), (
    f"missing source settings: {sorted(set(source_definitions) - set(definitions))}",
    f"unknown generated settings: {sorted(set(definitions) - set(source_definitions))}",
)
assert "machine_settings" not in definitions
assert "layer_height" in definitions
assert definitions["layer_height"]["type"] == "float"
assert definitions["infill_pattern"]["type"] == "enum"
assert definitions["infill_pattern"]["enum_values"]
assert len(definitions["infill_pattern"]["enum_values"]) == len(
    definitions["infill_pattern"]["enum_labels"]
)

expected_runtime_only = {
    key for key, runtime_only in source_runtime_only.items() if runtime_only
}
actual_runtime_only = {
    key for key, definition in definitions.items() if definition.get("runtime_only")
}
assert expected_runtime_only
assert actual_runtime_only == expected_runtime_only, (
    f"missing runtime-only settings: {sorted(expected_runtime_only - actual_runtime_only)}",
    f"unexpected runtime-only settings: {sorted(actual_runtime_only - expected_runtime_only)}",
)

for key, definition in definitions.items():
    assert definition["type"] in allowed_types, (key, definition["type"])
    assert definition.get("label"), key
    assert definition.get("runtime_only", False) is (key in expected_runtime_only), key
    for bound in ("min", "max"):
        value = definition.get(bound)
        assert value is None or isinstance(value, (int, float)), (key, bound, value)

    source = source_definitions[key]
    source_default = source.get("default_value", source.get("value"))
    if isinstance(source_default, (str, bool, int, float, list)):
        assert definition.get("default_value") == expected_default(
            source, definition["type"]
        ), key
    default = definition.get("default_value")
    if definition["type"] == "bool":
        assert isinstance(default, bool), key
    elif definition["type"] == "int":
        assert isinstance(default, int) and not isinstance(default, bool), key
    elif definition["type"] == "float":
        assert isinstance(default, (int, float)) and not isinstance(default, bool), key
    elif definition["type"] == "ints":
        assert isinstance(default, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in default
        ), key
    if source.get("type") == "enum":
        assert definition["enum_values"] == list(source["options"]), key
        assert definition["enum_labels"] == list(source["options"].values()), key

references: set[str] = set()
for panel_name in ("machine", "filament", "process"):
    panel = json.loads((root / f"{panel_name}.json").read_text())
    assert panel, panel_name
    for groups in panel.values():
        for settings in groups.values():
            references.update(settings)

expected_panel_settings = set(definitions) - expected_runtime_only
assert references == expected_panel_settings, (
    f"missing from panels: {sorted(expected_panel_settings - references)[:10]}",
    f"unknown panel keys: {sorted(references - set(definitions))[:10]}",
)

metadata = json.loads((root / "metadata.json").read_text())
assert metadata["settings_count"] == len(definitions)
assert metadata["runtime_only_settings_count"] == len(expected_runtime_only)
assert metadata["settings_contract"] == "cura-resolved-v1"
assert metadata["source_definition_files"] == source_files

visibility = json.loads((root / "conditional_visibility.json").read_text())
conditions = visibility["conditions"]
assert visibility["functions"] == []
assert visibility["variables"] == []
expected_conditions = {
    key
    for key, definition in source_definitions.items()
    if definition.get("enabled") is not True and "enabled" in definition
}
assert set(conditions) == expected_conditions, (
    f"missing conditions: {sorted(expected_conditions - set(conditions))}",
    f"unknown conditions: {sorted(set(conditions) - expected_conditions)}",
)
assert metadata["visibility_condition_count"] == len(conditions)

print(
    f"Validated {len(definitions)} normalized Cura settings and "
    f"{len(conditions)} visibility conditions"
)
