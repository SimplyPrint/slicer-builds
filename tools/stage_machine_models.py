#!/usr/bin/env python3
"""Give the command line the printer models it reads a model id from.

Bambu Studio and OrcaSlicer write a plate's ``printer_model_id`` (an X1E is ``C13``) into the
sliced 3MF, and the printer reads it. Their command line looks it up in
``profiles/BBL/machine_full/<printer_model>.json``, a folder only Bambu's own cloud has, so
every plate sliced from a bundle said ``printer_model_id=""``. This copies each BBL printer
model (``"type": "machine_model"``) from ``profiles/BBL/machine`` to where the CLI looks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def stage_machine_models(resources: Path) -> list[str]:
    source = resources / "profiles" / "BBL" / "machine"
    target = resources / "profiles" / "BBL" / "machine_full"
    staged = []
    for path in sorted(source.glob("*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("type") != "machine_model" or not model.get("model_id"):
            continue
        name = model.get("name") or path.stem
        target.mkdir(exist_ok=True)
        shutil.copyfile(path, target / f"{name}.json")
        staged.append(name)
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", type=Path, help="the slicer's resources directory")
    args = parser.parse_args(argv)
    staged = stage_machine_models(args.resources)
    if not staged:
        print(f"no BBL printer models under {args.resources}", file=sys.stderr)
        return 1
    print(f"staged {len(staged)} printer models: {', '.join(staged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
