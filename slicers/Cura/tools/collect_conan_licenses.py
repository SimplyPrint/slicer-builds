#!/usr/bin/env python3
"""Collect notices for the distributable host side of a Conan 2 graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


LICENSE_LIKE_NAME = re.compile(
    r"^(?:licen[cs]e|copying|copyright|notice|authors?|patents?)(?:[._ -].*)?$",
    re.IGNORECASE,
)
NON_DISTRIBUTION_PACKAGE_TYPES = frozenset({"build-scripts", "python-require"})
INVENTORY_GENERATOR = "simplyprint-conan-license-collector-v1"
ZIP_EPOCH = 315532800


class LicenseCollectionError(RuntimeError):
    """The graph cannot produce a complete distributable notice bundle."""


def dependency_is_distributable(edge: dict[str, Any]) -> bool:
    """Use Conan dependency traits, rather than package names, as the boundary."""
    if edge.get("skip") or edge.get("build") or edge.get("test"):
        return False
    return any(edge.get(trait) for trait in ("headers", "libs", "run"))


def distribution_nodes(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    consumers = [node for node in nodes.values() if node.get("recipe") == "Consumer"]
    if not consumers:
        raise LicenseCollectionError("Conan graph has no consumer node")

    reached: set[str] = set()
    pending = list(consumers)
    while pending:
        node = pending.pop()
        for node_id, edge in node.get("dependencies", {}).items():
            if not dependency_is_distributable(edge) or node_id in reached:
                continue
            dependency = nodes.get(node_id)
            if dependency is None:
                raise LicenseCollectionError(
                    f"Conan graph references missing dependency node {node_id}"
                )
            reached.add(node_id)
            if dependency.get("context") != "host":
                continue
            if dependency.get("package_type") in NON_DISTRIBUTION_PACKAGE_TYPES:
                continue
            pending.append(dependency)

    result = []
    for node_id in reached:
        node = nodes[node_id]
        if node.get("context") != "host":
            continue
        if node.get("package_type") in NON_DISTRIBUTION_PACKAGE_TYPES:
            continue
        if node.get("recipe") == "Consumer":
            continue
        result.append(node)
    return sorted(result, key=lambda node: (node.get("ref", ""), node.get("package_id", "")))


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _notice_files(node: dict[str, Any]) -> tuple[str, Path, list[Path]]:
    package_folder = node.get("package_folder")
    if package_folder:
        package_licenses = Path(package_folder) / "licenses"
        files = _regular_files(package_licenses)
        if files:
            return "package_licenses", package_licenses, files

    recipe_folder = node.get("recipe_folder")
    if recipe_folder:
        recipe_export = Path(recipe_folder)
        files = [
            path
            for path in _regular_files(recipe_export)
            if LICENSE_LIKE_NAME.fullmatch(path.name)
        ]
        if files:
            return "recipe_export", recipe_export, files

    ref = node.get("ref", "<unknown Conan reference>")
    raise LicenseCollectionError(
        f"Distribution dependency {ref} has neither package licenses nor "
        "license-like files in its recipe export"
    )


def _destination_name(ref: str, package_id: str | None) -> str:
    readable = re.sub(r"[^A-Za-z0-9._+-]+", "_", ref).strip("._-") or "dependency"
    readable = readable[:180].rstrip("._-") or "dependency"
    identity = f"{ref}\0{package_id or ''}".encode()
    return f"{readable}--{hashlib.sha256(identity).hexdigest()[:12]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_owned_output(output: Path, staging: Path) -> None:
    if output.name in {"", ".", ".."} or output == output.parent:
        raise LicenseCollectionError(f"Unsafe license output path: {output}")
    if output.is_symlink():
        raise LicenseCollectionError(f"License output path must not be a symlink: {output}")
    if output.exists():
        if not output.is_dir():
            raise LicenseCollectionError(f"License output path is not a directory: {output}")
        entries = list(output.iterdir())
        if entries:
            inventory_path = output / "inventory.json"
            try:
                existing = json.loads(inventory_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise LicenseCollectionError(
                    f"Refusing to replace unrecognized license output directory: {output}"
                ) from error
            if existing.get("generator") != INVENTORY_GENERATOR:
                raise LicenseCollectionError(
                    f"Refusing to replace unrecognized license output directory: {output}"
                )
        shutil.rmtree(output)
    staging.replace(output)


def collect(graph_file: Path, output: Path) -> dict[str, Any]:
    with graph_file.open(encoding="utf-8") as stream:
        document = json.load(stream)
    try:
        nodes = document["graph"]["nodes"]
    except (KeyError, TypeError) as error:
        raise LicenseCollectionError("Input is not a Conan 2 install graph") from error
    if not isinstance(nodes, dict):
        raise LicenseCollectionError("Conan graph nodes must be an object")

    dependencies = []
    missing = []
    selected = distribution_nodes(nodes)
    selected_files: list[tuple[dict[str, Any], str, Path, list[Path]]] = []
    for node in selected:
        try:
            source_kind, source_root, files = _notice_files(node)
        except LicenseCollectionError as error:
            missing.append(str(error))
            continue
        selected_files.append((node, source_kind, source_root, files))
    if missing:
        raise LicenseCollectionError("\n".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for node, source_kind, source_root, files in selected_files:
            ref = str(node.get("ref", ""))
            package_id = node.get("package_id")
            destination = _destination_name(ref, package_id)
            inventory_files = []
            for source in files:
                relative = source.relative_to(source_root)
                target = staging / destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                os.chmod(target, 0o644)
                os.utime(target, (ZIP_EPOCH, ZIP_EPOCH))
                inventory_files.append(
                    {
                        "path": (Path(destination) / relative).as_posix(),
                        "sha256": _sha256(target),
                        "source_path": relative.as_posix(),
                    }
                )

            dependencies.append(
                {
                    "declared_license": node.get("license"),
                    "destination": destination,
                    "file_count": len(inventory_files),
                    "files": inventory_files,
                    "package_id": package_id,
                    "package_revision": node.get("prev"),
                    "package_type": node.get("package_type"),
                    "ref": ref,
                    "source": source_kind,
                }
            )

        inventory = {
            "dependencies": dependencies,
            "dependency_count": len(dependencies),
            "file_count": sum(item["file_count"] for item in dependencies),
            "format_version": 1,
            "generator": INVENTORY_GENERATOR,
        }
        inventory_path = staging / "inventory.json"
        with inventory_path.open("w", encoding="utf-8") as stream:
            json.dump(inventory, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(inventory_path, 0o644)
        os.utime(inventory_path, (ZIP_EPOCH, ZIP_EPOCH))
        for directory in [staging, *(path for path in staging.rglob("*") if path.is_dir())]:
            os.utime(directory, (ZIP_EPOCH, ZIP_EPOCH))
        _replace_owned_output(output, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        inventory = collect(args.graph, args.output)
    except (LicenseCollectionError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"Conan license collection failed: {error}\n")
    print(
        "Collected notices for "
        f"{inventory['dependency_count']} Conan dependencies "
        f"({inventory['file_count']} files)"
    )


if __name__ == "__main__":
    main()
