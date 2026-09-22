#!/usr/bin/env python3
"""Annotate a dumped ``print_config_def.json`` with generic editor semantics.

Consumers need to answer "which setting picks a filament?" for an engine at a
version. Answering it downstream means spelling slicer setting keys in our own
source, which silently rots: OrcaSlicer 2.4 renamed its region selectors to
``*_filament_id`` while BambuStudio kept ``wall_filament``, and nothing told us.

Cura's generator already publishes the answer as ``editor.kind`` (see
``slicers/Cura/tools/generate-configs.py``). This brings the Slic3r-family dumps
to the same vocabulary, so every engine describes itself the same way.

``tool_reference`` is derived from the dump wherever the slicer marks it, and
declared per slicer only where it does not. Declared keys are checked against
the dump, so a rename fails the build instead of going quiet.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

CONFIG_DEF = "print_config_def.json"

#: Slic3r-family tool references count from 1, and 0 means "whatever the object
#: already uses". Cura counts from 0 with -1 as that sentinel, which its own
#: generator emits; this module never touches an entry that already has one.
FAMILY_INDEX_BASE = 1
INHERIT_SENTINEL = 0


def fail(message: str) -> None:
    raise SystemExit(f"annotate_editor_semantics: {message}")


def load_config_def(path: Path) -> dict[str, Any]:
    try:
        defs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(defs, dict):
        fail(f"{path} is not a JSON object")
    return defs


def is_derived_tool_reference(definition: dict[str, Any]) -> bool:
    """Recognise a filament chooser from what the slicer already publishes.

    An integer rendered as an open enumeration, counting from a real tool, with
    no unit suffix. The suffix is what separates it from the interface-layer
    counts, which are spelled the same way but measured in layers.
    """

    if definition.get("type") != "int":
        return False
    if definition.get("gui_type") != "i_enum_open":
        return False
    if definition.get("sidetext"):
        return False
    return str(definition.get("min", "")) == str(INHERIT_SENTINEL)


def declared_tool_references(path: Path | None, defs: dict[str, Any]) -> list[str]:
    """Read the per-slicer overlay, for engines that publish no marker at all.

    PrusaSlicer is the case: ``support_material_extruder`` and its siblings
    carry no ``gui_type``, so there is nothing to derive from.
    """

    if path is None or not path.is_file():
        return []
    try:
        overlay = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot read {path}: {error}")

    declared = overlay.get("tool_reference", {}).get("keys", [])
    if not isinstance(declared, list) or any(not isinstance(k, str) for k in declared):
        fail(f"{path}: [tool_reference] keys must be a list of strings")

    missing = [key for key in declared if key not in defs]
    if missing:
        fail(
            f"{path} declares {', '.join(sorted(missing))}, absent from {CONFIG_DEF}. "
            "The slicer renamed or removed it; update the overlay."
        )
    return declared


def tool_reference_editor(definition: dict[str, Any]) -> dict[str, Any]:
    """Describe how to read the value, so no consumer has to assume it."""

    editor: dict[str, Any] = {
        "kind": "tool_reference",
        "index_base": FAMILY_INDEX_BASE,
        "display_index_base": FAMILY_INDEX_BASE,
    }
    # min 0 admits the sentinel; min 1 means a tool must be named.
    if str(definition.get("min", "")) == str(INHERIT_SENTINEL):
        editor["selection"] = "optional"
        editor["sentinels"] = [{"value": INHERIT_SENTINEL, "semantic": "inherit"}]
    else:
        editor["selection"] = "required"
    return editor


def annotate(defs: dict[str, Any], overlay: Path | None) -> list[str]:
    """Add editor semantics in place; return the keys that carry them."""

    keys = {key for key, d in defs.items() if isinstance(d, dict) and is_derived_tool_reference(d)}
    keys.update(declared_tool_references(overlay, defs))

    annotated = []
    for key in sorted(keys):
        definition = defs[key]
        if not isinstance(definition, dict):
            continue
        # Never overwrite a generator that already described itself (Cura).
        if isinstance(definition.get("editor"), dict):
            continue
        definition["editor"] = tool_reference_editor(definition)
        annotated.append(key)
    return annotated


def dump_like_source(defs: dict[str, Any]) -> str:
    """Serialize the way the dump patch's Boost writer does.

    Verified byte-identical on a round-trip of the BambuStudio, OrcaSlicer,
    PrusaSlicer and CrealityPrint artifacts, so annotating an existing dump
    shows only the blocks this tool added rather than reformatting the file.
    """

    return json.dumps(defs, indent=4, ensure_ascii=False).replace("/", "\\/") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory holding the dumped artifacts")
    parser.add_argument("--overlay", type=Path, default=None, help="per-slicer semantics.toml")
    parser.add_argument("--check", action="store_true", help="report without writing")
    parser.add_argument(
        "--require-tool-reference",
        action="store_true",
        help="fail when the dump yields no tool reference at all",
    )
    args = parser.parse_args(argv)

    config_path = args.output / CONFIG_DEF
    if not config_path.is_file():
        fail(f"missing {config_path}")

    defs = load_config_def(config_path)
    annotated = annotate(defs, args.overlay)

    already = sum(
        1
        for d in defs.values()
        if isinstance(d, dict) and isinstance(d.get("editor"), dict)
        and d["editor"].get("kind") == "tool_reference"
    )
    if args.require_tool_reference and already == 0:
        fail(f"{config_path} declares no tool reference; the derivation rule or overlay is stale")

    print(f"tool references: {already} ({len(annotated)} added) in {config_path}")
    if not args.check and annotated:
        config_path.write_text(dump_like_source(defs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
