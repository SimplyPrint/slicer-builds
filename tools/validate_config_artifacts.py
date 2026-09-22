#!/usr/bin/env python3
"""Validate generated slicer configuration artifacts declared by a manifest."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_ARTIFACTS = (
    "machine.json",
    "filament.json",
    "process.json",
    "print_config_def.json",
)
SAFE_ARTIFACT_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.json$")


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Invalid JSON artifact {path}: {error}")


def required_artifacts(manifest_path: Path) -> tuple[str, ...]:
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        fail(f"Cannot read slicer manifest {manifest_path}: {error}")

    config = manifest.get("ci", {}).get("config", {})
    artifacts = config.get("required_artifacts", DEFAULT_REQUIRED_ARTIFACTS)
    if not isinstance(artifacts, list) and not isinstance(artifacts, tuple):
        fail(f"{manifest_path}: ci.config.required_artifacts must be an array")
    if not artifacts:
        fail(f"{manifest_path}: ci.config.required_artifacts must not be empty")

    result: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, str) or not SAFE_ARTIFACT_PATH.fullmatch(artifact):
            fail(f"{manifest_path}: unsafe required artifact path {artifact!r}")
        if artifact in result:
            fail(f"{manifest_path}: duplicate required artifact {artifact!r}")
        result.append(artifact)
    return tuple(result)


def validate_translation_index(index_path: Path, output_path: Path) -> None:
    index = load_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("locales"), dict):
        fail(f"Translation index {index_path} must contain a locales object")

    for locale, filename in index["locales"].items():
        if (
            not isinstance(locale, str)
            or not isinstance(filename, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", filename)
        ):
            fail(f"Translation index {index_path} contains an unsafe locale entry")
        catalog_path = index_path.parent / filename
        catalog = load_json(catalog_path)
        if not isinstance(catalog, dict) or catalog.get("locale") != locale:
            fail(f"Translation catalog {catalog_path} does not match locale {locale!r}")
        if not isinstance(catalog.get("messages"), dict):
            fail(f"Translation catalog {catalog_path} must contain a messages object")
        if "settings" in catalog and not isinstance(catalog["settings"], dict):
            fail(f"Translation catalog {catalog_path} has an invalid settings object")
        if "ui" in catalog and not isinstance(catalog["ui"], dict):
            fail(f"Translation catalog {catalog_path} has an invalid ui object")
        if not catalog_path.is_relative_to(output_path):
            fail(f"Translation catalog escapes output directory: {catalog_path}")


def validate_editor_semantics(config_path: Path) -> None:
    """Every engine must say which of its settings pick a filament.

    Consumers that cannot read it off the config have to spell slicer setting
    keys themselves, which goes stale silently — OrcaSlicer renamed its region
    choosers to ``*_filament_id`` and nothing noticed.
    """

    defs = load_json(config_path)
    if not isinstance(defs, dict):
        fail(f"{config_path} must contain an object")

    tool_references = [
        key
        for key, definition in defs.items()
        if isinstance(definition, dict)
        and isinstance(definition.get("editor"), dict)
        and definition["editor"].get("kind") == "tool_reference"
    ]
    if not tool_references:
        fail(
            f"{config_path} declares no editor.kind=tool_reference. Run "
            "tools/annotate_editor_semantics.py, and check the derivation rule "
            "against this slicer's dump."
        )

    for key in tool_references:
        editor = defs[key]["editor"]
        if not isinstance(editor.get("index_base"), int):
            fail(f"{config_path}: {key} tool reference has no integer index_base")
        if editor.get("selection") not in {"required", "optional"}:
            fail(f"{config_path}: {key} tool reference has an invalid selection")


def main() -> None:
    if len(sys.argv) != 3:
        fail("Usage: validate_config_artifacts.py <slicer.toml> <output-directory>")

    manifest_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]).resolve()
    for artifact in required_artifacts(manifest_path):
        path = output_path / artifact
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"Missing or empty config artifact: {path}")
        load_json(path)
        if path.name == "_index.json" and path.parent.name == "translations":
            validate_translation_index(path, output_path)
        if path.name == "print_config_def.json":
            validate_editor_semantics(path)

    for path in output_path.rglob("*.json"):
        load_json(path)


if __name__ == "__main__":
    main()
