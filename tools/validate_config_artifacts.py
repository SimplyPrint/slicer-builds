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
        if not catalog_path.is_relative_to(output_path):
            fail(f"Translation catalog escapes output directory: {catalog_path}")


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

    for path in output_path.rglob("*.json"):
        load_json(path)


if __name__ == "__main__":
    main()
