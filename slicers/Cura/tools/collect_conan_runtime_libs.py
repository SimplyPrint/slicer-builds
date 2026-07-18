#!/usr/bin/env python3
"""Collect Linux runtime shared objects from a Conan 2 host graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable

from collect_conan_licenses import LicenseCollectionError, distribution_nodes


ELF_MAGIC = b"\x7fELF"
SHARED_OBJECT_NAME = re.compile(r"\.so(?:\.|$)")


class RuntimeLibraryCollectionError(RuntimeError):
    """The Conan graph cannot produce an unambiguous runtime library set."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(len(ELF_MAGIC)) == ELF_MAGIC
    except OSError as error:
        raise RuntimeLibraryCollectionError(
            f"Could not inspect runtime library candidate {path}: {error}"
        ) from error


def read_elf_soname(path: Path) -> str | None:
    """Return an ELF SONAME using the same patchelf required by packaging."""
    result = subprocess.run(
        ["patchelf", "--print-soname", os.fspath(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"patchelf exited {result.returncode}"
        raise RuntimeLibraryCollectionError(
            f"Could not read the ELF SONAME for {path}: {detail}"
        )
    return result.stdout.strip() or None


def _safe_destination_name(name: str, source: Path) -> str:
    if name in {"", ".", ".."} or Path(name).name != name:
        raise RuntimeLibraryCollectionError(
            f"Runtime library {source} has unsafe bundle name {name!r}"
        )
    return name


def _runtime_payloads(package_root: Path) -> list[Path]:
    """Resolve every ELF reached through a Linux shared-object filename."""
    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise RuntimeLibraryCollectionError(
            f"Conan package folder does not exist: {package_root}"
        )

    payloads: set[Path] = set()
    entries = sorted(
        package_root.rglob("*"),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    for entry in entries:
        if not SHARED_OBJECT_NAME.search(entry.name):
            continue
        if entry.is_symlink():
            try:
                payload = entry.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise RuntimeLibraryCollectionError(
                    f"Broken Conan runtime library link {entry}: {error}"
                ) from error
        elif entry.is_file():
            payload = entry.resolve()
        else:
            continue
        try:
            payload.relative_to(package_root)
        except ValueError as error:
            raise RuntimeLibraryCollectionError(
                f"Conan runtime library link escapes its package: {entry} -> {payload}"
            ) from error
        if payload.is_file() and _is_elf(payload):
            payloads.add(payload)
    return sorted(payloads, key=lambda path: path.relative_to(package_root).as_posix())


def _load_nodes(graph_file: Path) -> dict[str, dict[str, Any]]:
    with graph_file.open(encoding="utf-8") as stream:
        document = json.load(stream)
    try:
        nodes = document["graph"]["nodes"]
    except (KeyError, TypeError) as error:
        raise RuntimeLibraryCollectionError(
            "Input is not a Conan 2 install graph"
        ) from error
    if not isinstance(nodes, dict):
        raise RuntimeLibraryCollectionError("Conan graph nodes must be an object")
    return nodes


def collect(
    graph_file: Path,
    output: Path,
    *,
    soname_reader: Callable[[Path], str | None] = read_elf_soname,
) -> dict[str, Any]:
    """Copy a deterministic, conflict-free runtime library set into output."""
    nodes = _load_nodes(graph_file)
    candidates: dict[str, dict[str, str | Path]] = {}

    for node in distribution_nodes(nodes):
        ref = str(node.get("ref", "<unknown Conan reference>"))
        package_folder = node.get("package_folder")
        if not package_folder:
            raise RuntimeLibraryCollectionError(
                f"Distribution dependency {ref} has no package folder to inspect"
            )
        package_root = Path(package_folder)
        for payload in _runtime_payloads(package_root):
            soname = soname_reader(payload)
            names = {_safe_destination_name(payload.name, payload)}
            if soname is not None:
                names.add(_safe_destination_name(soname, payload))
            digest = _sha256(payload)
            for name in sorted(names):
                previous = candidates.get(name)
                if previous is not None and previous["sha256"] != digest:
                    raise RuntimeLibraryCollectionError(
                        f"Conflicting Conan runtime library {name}: "
                        f"{previous['ref']} ({previous['source']}) and {ref} ({payload})"
                    )
                if previous is None:
                    candidates[name] = {
                        "ref": ref,
                        "sha256": digest,
                        "source": payload,
                    }

    output.mkdir(parents=True, exist_ok=True)
    for name, candidate in sorted(candidates.items()):
        target = output / name
        source = Path(candidate["source"])
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeLibraryCollectionError(
                f"Runtime library destination is not a regular file: {target}"
            )
        if target.exists():
            if _sha256(target) != candidate["sha256"]:
                raise RuntimeLibraryCollectionError(
                    f"Conflicting existing runtime library {name}: {target} and {source}"
                )

    for name, candidate in sorted(candidates.items()):
        target = output / name
        source = Path(candidate["source"])
        if target.exists():
            continue
        shutil.copyfile(source, target)
        os.chmod(target, 0o755)

    libraries = [
        {
            "name": name,
            "ref": str(candidate["ref"]),
            "sha256": str(candidate["sha256"]),
            "source": os.fspath(candidate["source"]),
        }
        for name, candidate in sorted(candidates.items())
    ]
    return {"libraries": libraries, "library_count": len(libraries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        inventory = collect(args.graph, args.output)
    except (
        json.JSONDecodeError,
        LicenseCollectionError,
        OSError,
        RuntimeLibraryCollectionError,
    ) as error:
        parser.exit(1, f"Conan runtime library collection failed: {error}\n")
    print(
        f"Collected {inventory['library_count']} Conan runtime library filenames"
    )


if __name__ == "__main__":
    main()
