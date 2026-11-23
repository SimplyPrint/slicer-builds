#!/usr/bin/env python3
"""Give the command line the system profiles Bambu's own cloud gives it.

The Bambu-family CLI reads two folders that only Bambu's cloud ships:

- ``profiles/BBL/machine_full/<printer_model>.json``: a plate's ``printer_model_id`` (an X1E
  is ``C13``), which the printer reads. Without it every plate said ``printer_model_id=""``.
- ``profiles/BBL/filament_full/<name>.json``: each system filament, complete. A user preset
  names one as its parent; the CLI extends the preset's per-nozzle settings from it, and the
  cloud slicer fills in every setting the preset leaves out, as the desktop app does.

Filaments are completed as the app's preset loader completes them: the parent's settings, then
each ``include`` template's, then the preset's own. The ``filament_id`` is the first one up the
chain. A parent is looked up in the preset's own vendor first, then in any vendor
(``OrcaFilamentLibrary`` presets are parents of other vendors' presets).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

#: Not settings: the loader reads these as the preset's identity and links.
_LINKS = ("inherits", "include", "instantiation")
_IDENTITY = ("name", "type", "from", "setting_id", "filament_id", "description", "version")


def stage_machine_models(resources: Path) -> list[str]:
    source = resources / "profiles" / "BBL" / "machine"
    target = resources / "profiles" / "BBL" / "machine_full"
    staged = []
    for path in sorted(source.glob("*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("type") != "machine_model" or not model.get("model_id"):
            continue
        name = model.get("name") or path.stem
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / f"{name}.json")
        staged.append(name)
    return staged


def _filament_presets(profiles: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_vendor: dict[str, dict[str, dict[str, Any]]] = {}
    for vendor in sorted(p for p in profiles.iterdir() if p.is_dir()):
        for path in sorted((vendor / "filament").rglob("*.json")):
            try:
                preset = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if isinstance(preset, dict) and isinstance(preset.get("name"), str):
                by_vendor.setdefault(vendor.name, {}).setdefault(preset["name"], preset)
    return by_vendor


def complete_filaments(profiles: Path) -> dict[str, dict[str, Any]]:
    """Every instantiable system filament, by name, as the app loads it."""
    by_vendor = _filament_presets(profiles)
    done: dict[tuple[str, str], dict[str, Any]] = {}

    def find(vendor: str, name: str) -> tuple[str, dict[str, Any]] | None:
        if name in by_vendor.get(vendor, {}):
            return vendor, by_vendor[vendor][name]
        owner = next((v for v, named in by_vendor.items() if name in named), None)
        return (owner, by_vendor[owner][name]) if owner else None

    def complete(vendor: str, name: str, seen: frozenset[str]) -> dict[str, Any]:
        if (vendor, name) in done:
            return done[(vendor, name)]
        found = find(vendor, name)
        if found is None or name in seen:
            return {}
        vendor, preset = found
        seen = seen | {name}
        out: dict[str, Any] = {}
        if preset.get("inherits"):
            parent = complete(vendor, preset["inherits"], seen)
            out.update({k: v for k, v in parent.items() if k not in _IDENTITY})
            if parent.get("filament_id"):
                out["filament_id"] = parent["filament_id"]
        for include in preset.get("include") or []:
            included = complete(vendor, include, seen)
            out.update({k: v for k, v in included.items() if k not in _IDENTITY})
        out.update({k: v for k, v in preset.items() if k not in _LINKS})
        done[(vendor, name)] = out
        return out

    completed: dict[str, dict[str, Any]] = {}
    # BBL first: on a name two vendors share, the Bambu preset is the one Bambu's CLI means.
    for vendor in sorted(by_vendor, key=lambda v: (v != "BBL", v)):
        for name, preset in by_vendor[vendor].items():
            if str(preset.get("instantiation", "true")).lower() != "true" or name in completed:
                continue
            filament = dict(complete(vendor, name, frozenset()))
            filament.update({"name": name, "type": "filament", "from": "system"})
            completed[name] = filament
    return completed


def stage_filaments(resources: Path) -> int:
    target = resources / "profiles" / "BBL" / "filament_full"
    filaments = complete_filaments(resources / "profiles")
    for name, filament in filaments.items():
        # The CLI joins the name into the path as it is, so a "/" in a name is a folder.
        path = target / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(filament, indent=1, sort_keys=True), encoding="utf-8")
    return len(filaments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", type=Path, help="the slicer's resources directory")
    args = parser.parse_args(argv)
    models = stage_machine_models(args.resources)
    filaments = stage_filaments(args.resources)
    if not filaments:
        print(f"no system filaments under {args.resources}/profiles", file=sys.stderr)
        return 1
    print(f"staged {len(models)} printer models and {filaments} system filaments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
