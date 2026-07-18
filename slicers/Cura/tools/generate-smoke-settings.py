#!/usr/bin/env python3
"""Create a literal, fully resolved CuraEngine -r payload for a smoke cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_defaults(settings: dict[str, Any], result: dict[str, Any]) -> None:
    for key, definition in settings.items():
        if "default_value" in definition and isinstance(
            definition["default_value"], (str, bool, int, float, list)
        ):
            result[key] = definition["default_value"]
        elif isinstance(definition.get("value"), (bool, int, float, list)):
            result[key] = definition["value"]
        children = definition.get("children")
        if isinstance(children, dict):
            collect_defaults(children, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("definition", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="smoke.stl")
    args = parser.parse_args()

    with args.definition.open(encoding="utf-8") as source:
        definition = json.load(source)

    defaults: dict[str, Any] = {}
    collect_defaults(definition["settings"], defaults)
    defaults.update(
        {
            "adhesion_type": "none",
            "infill_sparse_density": 10,
            "layer_height": 0.2,
            "layer_height_0": 0.2,
            "machine_depth": 120,
            "machine_extruder_count": 1,
            "machine_gcode_flavor": "RepRap (Marlin/Sprinter)",
            "machine_height": 120,
            "machine_heated_bed": True,
            "machine_width": 120,
            "material_bed_temperature": 60,
            "material_bed_temperature_layer_0": 60,
            "material_diameter": 1.75,
            "material_print_temperature": 205,
            "material_print_temperature_layer_0": 205,
            "support_enable": False,
            "wall_line_count": 2,
        }
    )

    extruder_defaults: dict[str, Any] = {}
    extruder_definition_path = args.definition.with_name("fdmextruder.def.json")
    with extruder_definition_path.open(encoding="utf-8") as source:
        extruder_definition = json.load(source)
    collect_defaults(extruder_definition["settings"], extruder_defaults)
    extruder_defaults.update(defaults)
    extruder_defaults["extruder_nr"] = 0
    payload = {
        "global": defaults,
        "extruder.0": extruder_defaults,
        args.model: {"extruder_nr": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
