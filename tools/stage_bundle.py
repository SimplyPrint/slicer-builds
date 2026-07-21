#!/usr/bin/env python3
"""Stage a small slicer CLI bundle for the cloud-slicer runtime.

Only shared libraries resolved below explicitly supplied ``--library-root``
directories are copied. Libraries supplied by the operating system stay in the
runtime image instead of being duplicated in every slicer bundle. Private
libraries are colocated with the executable; the backend adds ``bundle/bin`` to
``LD_LIBRARY_PATH`` when it launches a slicer.

Resource trees are copied in full by default. Repeated ``--resource-include``
arguments opt into a fail-closed relative-path/glob allowlist, and ``--json``
emits byte and file inventory for the source, selection, and staged bundle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import fnmatch
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Sequence


class BundleError(RuntimeError):
    """Raised for an invalid or incomplete staged bundle."""


class MissingDependencies(BundleError):
    """Raised when ldd cannot resolve one or more loader names."""

    def __init__(self, binary: Path, names: Iterable[str]) -> None:
        self.binary = binary
        self.names = tuple(sorted(set(names)))
        super().__init__(
            f"unresolved shared libraries for {binary}: {', '.join(self.names)}"
        )


@dataclass(frozen=True)
class Dependency:
    name: str
    path: Path


@dataclass(frozen=True)
class ResourceEntry:
    """One filesystem entry below a resource root."""

    root: Path
    relative: PurePosixPath
    kind: str
    size: int = 0
    link_target: str | None = None
    resolved_link_target: PurePosixPath | None = None


@dataclass(frozen=True)
class ResourceGroupInventory:
    """Resource byte counts grouped by the first path component."""

    path: str
    source_files: int
    source_bytes: int
    selected_files: int
    selected_bytes: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "path": self.path,
            "source_files": self.source_files,
            "source_bytes": self.source_bytes,
            "selected_files": self.selected_files,
            "selected_bytes": self.selected_bytes,
            "omitted_files": self.source_files - self.selected_files,
            "omitted_bytes": self.source_bytes - self.selected_bytes,
        }


@dataclass(frozen=True)
class ResourceInventory:
    """Source, policy-selection, and final staged resource counts."""

    mode: str
    includes: tuple[str, ...]
    source_files: int
    source_bytes: int
    selected_files: int
    selected_bytes: int
    staged_files: int
    staged_bytes: int
    source_symlinks: int
    selected_symlinks: int
    staged_symlinks: int
    groups: tuple[ResourceGroupInventory, ...]

    @property
    def omitted_files(self) -> int:
        return self.source_files - self.selected_files

    @property
    def omitted_bytes(self) -> int:
        return self.source_bytes - self.selected_bytes

    @property
    def omitted_symlinks(self) -> int:
        return self.source_symlinks - self.selected_symlinks

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "includes": list(self.includes),
            "source_files": self.source_files,
            "source_bytes": self.source_bytes,
            "selected_files": self.selected_files,
            "selected_bytes": self.selected_bytes,
            "staged_files": self.staged_files,
            "staged_bytes": self.staged_bytes,
            "omitted_files": self.omitted_files,
            "omitted_bytes": self.omitted_bytes,
            "source_symlinks": self.source_symlinks,
            "selected_symlinks": self.selected_symlinks,
            "staged_symlinks": self.staged_symlinks,
            "omitted_symlinks": self.omitted_symlinks,
            "groups": [group.as_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class BundleResult:
    """Result of staging, with legacy three-value unpacking support."""

    executable: Path
    architecture: str
    library_count: int
    byte_count: int
    resources: ResourceInventory

    def __iter__(self) -> Iterator[Path | int]:
        # stage_bundle historically returned this three-tuple. Keep external
        # callers that unpack it working while exposing the resource inventory.
        yield self.executable
        yield self.library_count
        yield self.byte_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "executable": str(self.executable),
            "architecture": self.architecture,
            "library_count": self.library_count,
            "bundle_bytes": self.byte_count,
            "bundle_resource_bytes": self.resources.staged_bytes,
            "resources": self.resources.as_dict(),
        }


_RESOLVED_DEPENDENCY = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(")
_DIRECT_DEPENDENCY = re.compile(r"^\s*(/\S+)\s+\(")
_MISSING_DEPENDENCY = re.compile(r"^\s*(\S+)\s+=>\s+not found\s*$")
_PROGRAM_INTERPRETER = re.compile(r"\[Requesting program interpreter: (.+)]")
_LOADER_MISSING_DEPENDENCY = re.compile(
    r"error while loading shared libraries: ([^:]+):"
)


def _configured_resource_patterns(args: argparse.Namespace) -> Sequence[str]:
    """Return explicit CLI patterns, or newline-delimited build policy."""
    explicit = getattr(args, "resource_include", None)
    if explicit is not None:
        return explicit
    configured = os.environ.get("SLICER_RESOURCE_INCLUDES", "")
    return configured.splitlines() if configured else ()


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _resolve_missing_search_dirs(
    roots: Sequence[Path], names: Sequence[str]
) -> list[Path]:
    """Find one unambiguous candidate for each unresolved loader name."""
    wanted = set(names)
    candidates: dict[str, set[Path]] = defaultdict(set)
    for root in roots:
        for directory, subdirs, filenames in os.walk(root):
            subdirs.sort()
            for filename in sorted(wanted & set(filenames)):
                candidates[filename].add((Path(directory) / filename).resolve())

    search_dirs: list[Path] = []
    for name in sorted(wanted):
        matches = candidates[name]
        if not matches:
            raise BundleError(
                f"unresolved shared library has no eligible candidate: {name}"
            )
        if len(matches) != 1:
            rendered = ", ".join(str(path) for path in sorted(matches, key=str))
            raise BundleError(
                f"ambiguous shared-library candidates for {name}: {rendered}"
            )
        directory = next(iter(matches)).parent
        if directory not in search_dirs:
            search_dirs.append(directory)
    return search_dirs


def _parse_dependencies(binary: Path, output: str) -> list[Dependency]:
    dependencies: list[Dependency] = []
    missing: list[str] = []
    for line in output.splitlines():
        if match := _MISSING_DEPENDENCY.match(line):
            missing.append(match.group(1))
            continue
        if match := _RESOLVED_DEPENDENCY.match(line):
            name, raw_path = match.groups()
        elif match := _DIRECT_DEPENDENCY.match(line):
            raw_path = match.group(1)
            name = Path(raw_path).name
        else:
            continue
        path = Path(raw_path)
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise BundleError(
                f"ldd returned a missing path for {name}: {raw_path}"
            ) from exc
        dependencies.append(Dependency(name=name, path=path))

    if missing:
        raise MissingDependencies(binary, missing)
    return dependencies


def _ldd(binary: Path, search_dirs: Sequence[Path]) -> list[Dependency]:
    env = os.environ.copy()
    configured_path = env.get("LD_LIBRARY_PATH")
    prefixes = [str(path) for path in search_dirs]
    if configured_path:
        prefixes.append(configured_path)
    if prefixes:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(prefixes)

    try:
        process = subprocess.run(
            ["ldd", str(binary)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except OSError as exc:
        raise BundleError(f"cannot run ldd for {binary}: {exc}") from exc
    output = process.stdout
    if process.returncode != 0:
        lower_output = output.lower()
        if (
            "not a dynamic executable" in lower_output
            or "statically linked" in lower_output
        ):
            return []
        raise BundleError(f"ldd failed for {binary}:\n{output.rstrip()}")
    return _parse_dependencies(binary, output)


def _elf_interpreter(binary: Path) -> Path | None:
    """Return the native ELF loader, or ``None`` for a static executable."""

    try:
        process = subprocess.run(
            ["readelf", "--program-headers", "--wide", str(binary)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        raise BundleError(
            f"cannot inspect ELF interpreter for {binary}: {exc}"
        ) from exc
    if process.returncode != 0:
        raise BundleError(f"readelf failed for {binary}:\n{process.stdout.rstrip()}")
    match = _PROGRAM_INTERPRETER.search(process.stdout)
    if match is None:
        return None
    interpreter = Path(match.group(1))
    try:
        return interpreter.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BundleError(
            f"ELF interpreter does not exist for {binary}: {interpreter}"
        ) from exc


def _hermetic_dependencies(
    binary: Path, search_dirs: Sequence[Path]
) -> list[Dependency]:
    """Resolve dependencies without consulting build-time RPATH/RUNPATH."""

    interpreter = _elf_interpreter(binary)
    if interpreter is None:
        return []
    command = [
        str(interpreter),
        "--inhibit-rpath",
        "",
        "--library-path",
        os.pathsep.join(str(path) for path in search_dirs),
        "--list",
        str(binary),
    ]
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    try:
        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except OSError as exc:
        raise BundleError(
            f"cannot run ELF interpreter {interpreter} for {binary}: {exc}"
        ) from exc
    output = process.stdout
    if process.returncode != 0:
        if match := _LOADER_MISSING_DEPENDENCY.search(output):
            raise MissingDependencies(binary, [match.group(1)])
        raise BundleError(
            f"ELF interpreter dependency check failed for {binary}:\n{output.rstrip()}"
        )
    return _parse_dependencies(binary, output)


def _validate_elf(
    path: Path, description: str, expected_architecture: str | None = None
) -> str:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError as exc:
        raise BundleError(f"cannot read {description} {path}: {exc}") from exc
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise BundleError(f"{description} is not an ELF file: {path}")

    elf_class = header[4]
    elf_data = header[5]
    if elf_class != 2:
        raise BundleError(f"{description} is not a 64-bit ELF file: {path}")
    if elf_data == 1:
        byte_order = "little"
    elif elf_data == 2:
        byte_order = "big"
    else:
        raise BundleError(f"{description} has an invalid ELF byte order: {path}")
    machine = int.from_bytes(header[18:20], byte_order)
    architectures = {62: "x86-64", 183: "arm64"}
    architecture = architectures.get(machine)
    if architecture is None:
        raise BundleError(
            f"{description} uses unsupported ELF machine {machine}: {path}"
        )
    if expected_architecture is not None and architecture != expected_architecture:
        raise BundleError(
            f"{description} architecture is {architecture}, expected "
            f"{expected_architecture}: {path}"
        )
    return architecture


def _resolve_executable(candidates: Sequence[Path], expected_architecture: str) -> Path:
    rejected: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            rejected.append(f"{candidate} (missing)")
            continue
        if not candidate.is_file():
            rejected.append(f"{candidate} (not a file)")
            continue
        if not os.access(candidate, os.X_OK):
            rejected.append(f"{candidate} (not executable)")
            continue
        resolved = candidate.resolve(strict=True)
        _validate_elf(resolved, "executable", expected_architecture)
        return resolved
    details = "\n  ".join(rejected)
    raise BundleError(f"no usable executable candidate found:\n  {details}")


def _dependency_closure(
    executable: Path, library_roots: Sequence[Path]
) -> dict[Path, set[str]]:
    """Map each bundled real library to the loader names that reference it."""
    # Most builds encode exact private dependency directories in RUNPATH. Avoid
    # recursively walking a large Conan/dependency cache unless ldd proves that
    # an older build layout actually needs the fallback search path.
    pending: deque[Path] = deque([executable])
    inspected: set[Path] = set()
    by_source: dict[Path, set[str]] = defaultdict(set)
    name_owner: dict[str, Path] = {}

    while pending:
        binary = pending.popleft().resolve(strict=True)
        if binary in inspected:
            continue
        inspected.add(binary)
        try:
            dependencies = _ldd(binary, [])
        except MissingDependencies as error:
            # Only scan for loader names that were actually unresolved. Broadly
            # prepending every build directory can silently select a debug or
            # stale library with the same SONAME.
            search_dirs = _resolve_missing_search_dirs(library_roots, error.names)
            dependencies = _ldd(binary, search_dirs)
        for dependency in dependencies:
            if not _is_within(dependency.path, library_roots):
                continue
            if Path(dependency.name).name != dependency.name:
                raise BundleError(f"unsafe shared-library name: {dependency.name!r}")
            previous = name_owner.get(dependency.name)
            if previous is not None and previous != dependency.path:
                raise BundleError(
                    f"shared-library name collision for {dependency.name}: "
                    f"{previous} and {dependency.path}"
                )
            name_owner[dependency.name] = dependency.path
            by_source[dependency.path].add(dependency.name)
            pending.append(dependency.path)
    return dict(by_source)


def _normalize_resource_patterns(patterns: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for pattern in patterns:
        if not pattern or "\x00" in pattern or "\\" in pattern:
            raise BundleError(f"unsafe resource include pattern: {pattern!r}")
        if pattern.startswith("/"):
            raise BundleError(f"unsafe resource include pattern: {pattern!r}")
        parts = pattern.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise BundleError(f"unsafe resource include pattern: {pattern!r}")
        if any("**" in part and part != "**" for part in parts):
            raise BundleError(f"unsafe resource include pattern: {pattern!r}")
        if any(part.count("[") != part.count("]") for part in parts):
            raise BundleError(f"invalid resource include pattern: {pattern!r}")
        if pattern not in normalized:
            normalized.append(pattern)
    return tuple(normalized)


def _resource_pattern_matches(pattern: str, relative: PurePosixPath) -> bool:
    pattern_parts = tuple(pattern.split("/"))
    path_parts = relative.parts

    @lru_cache(maxsize=None)
    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        component = pattern_parts[pattern_index]
        if component == "**":
            return matches(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and matches(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], component)
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


def _normalize_link_target(
    root: Path, relative: PurePosixPath, target: str
) -> PurePosixPath:
    if not target or "\x00" in target or PurePosixPath(target).is_absolute():
        raise BundleError(f"unsafe resource symlink {relative} -> {target!r}")

    components = list(relative.parent.parts)
    for component in PurePosixPath(target).parts:
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                raise BundleError(
                    f"resource symlink escapes its root: {relative} -> {target}"
                )
            components.pop()
        else:
            components.append(component)
    lexical_target = PurePosixPath(*components)
    target_path = root.joinpath(*lexical_target.parts)
    try:
        resolved_target = target_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleError(
            f"resource symlink has an invalid target: {relative} -> {target}"
        ) from exc
    if not _is_within(resolved_target, [root]):
        raise BundleError(f"resource symlink escapes its root: {relative} -> {target}")
    resolved_relative = resolved_target.relative_to(root)
    if not resolved_relative.parts:
        return PurePosixPath(".")
    return PurePosixPath(resolved_relative.as_posix())


def _scan_resource_root(source: Path) -> tuple[Path, list[ResourceEntry]]:
    if not source.is_dir():
        raise BundleError(
            f"resource root does not exist or is not a directory: {source}"
        )
    try:
        root = source.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve resource root {source}: {exc}") from exc

    found: list[ResourceEntry] = []
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath("."))]
    while pending:
        directory, relative_parent = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda child: child.name)
        except OSError as exc:
            raise BundleError(
                f"cannot scan resource directory {directory}: {exc}"
            ) from exc
        child_directories: list[tuple[Path, PurePosixPath]] = []
        for child in children:
            relative = relative_parent / child.name
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise BundleError(
                    f"cannot inspect resource path {child.path}: {exc}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    link_target = os.readlink(child.path)
                except OSError as exc:
                    raise BundleError(
                        f"cannot read resource symlink {child.path}: {exc}"
                    ) from exc
                found.append(
                    ResourceEntry(
                        root=root,
                        relative=relative,
                        kind="symlink",
                        link_target=link_target,
                        resolved_link_target=_normalize_link_target(
                            root, relative, link_target
                        ),
                    )
                )
            elif stat.S_ISDIR(metadata.st_mode):
                found.append(
                    ResourceEntry(root=root, relative=relative, kind="directory")
                )
                child_directories.append((Path(child.path), relative))
            elif stat.S_ISREG(metadata.st_mode):
                found.append(
                    ResourceEntry(
                        root=root,
                        relative=relative,
                        kind="file",
                        size=metadata.st_size,
                    )
                )
            else:
                raise BundleError(f"unsupported resource file type: {child.path}")
        pending.extend(reversed(child_directories))
    return root, found


def _resource_entry_selected(entry: ResourceEntry, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    candidates = (entry.relative, *entry.relative.parents)
    return any(
        _resource_pattern_matches(pattern, candidate)
        for candidate in candidates
        if candidate.parts
        for pattern in patterns
    )


def _validate_selected_symlinks(
    entries: Sequence[ResourceEntry], selected: set[PurePosixPath]
) -> None:
    by_path = {entry.relative: entry for entry in entries}
    for entry in entries:
        if entry.kind != "symlink" or entry.relative not in selected:
            continue
        target = entry.resolved_link_target
        assert target is not None
        if target == PurePosixPath("."):
            required = entries
        else:
            target_entry = by_path.get(target)
            if target_entry is None:
                raise BundleError(
                    f"resource symlink target was not scanned: "
                    f"{entry.relative} -> {entry.link_target}"
                )
            required = [target_entry]
            if target_entry.kind == "directory":
                required.extend(
                    candidate
                    for candidate in entries
                    if candidate.relative != target
                    and target in candidate.relative.parents
                )
        omitted_targets = sorted(
            (
                candidate.relative
                for candidate in required
                if candidate.relative not in selected
            ),
            key=str,
        )
        if omitted_targets:
            raise BundleError(
                "resource allowlist selects symlink "
                f"{entry.relative} but omits its target tree at {target}"
            )


def _destination_path(destination: Path, relative: PurePosixPath) -> Path:
    result = destination.joinpath(*relative.parts)
    parent = result.parent
    while parent != destination:
        if parent.is_symlink():
            raise BundleError(f"resource destination parent is a symlink: {parent}")
        parent = parent.parent
    return result


def _copy_resource_entry(entry: ResourceEntry, destination: Path) -> None:
    source = entry.root.joinpath(*entry.relative.parts)
    target = _destination_path(destination, entry.relative)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise BundleError(f"cannot re-read resource path {source}: {exc}") from exc

    if entry.kind == "file":
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleError(f"resource path changed while staging: {source}")
        if target.is_dir() and not target.is_symlink():
            raise BundleError(f"resource path type collision: {target}")
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(source, target, follow_symlinks=False)
        return

    if entry.kind == "symlink":
        if not stat.S_ISLNK(metadata.st_mode):
            raise BundleError(f"resource path changed while staging: {source}")
        try:
            link_target = os.readlink(source)
        except OSError as exc:
            raise BundleError(
                f"cannot re-read resource symlink {source}: {exc}"
            ) from exc
        if link_target != entry.link_target:
            raise BundleError(f"resource symlink changed while staging: {source}")
        _normalize_link_target(entry.root, entry.relative, link_target)
        if target.is_dir() and not target.is_symlink():
            raise BundleError(f"resource path type collision: {target}")
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(link_target, target_is_directory=source.resolve().is_dir())
        return

    raise AssertionError(f"unexpected resource entry kind: {entry.kind}")


def _staged_resource_counts(destination: Path) -> tuple[int, int, int]:
    files = 0
    byte_count = 0
    symlinks = 0
    pending = [destination]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for child in iterator:
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    symlinks += 1
                elif stat.S_ISDIR(metadata.st_mode):
                    pending.append(Path(child.path))
                elif stat.S_ISREG(metadata.st_mode):
                    files += 1
                    byte_count += metadata.st_size
                else:
                    raise BundleError(f"unsupported staged resource type: {child.path}")
    return files, byte_count, symlinks


def _copy_resources(
    sources: Iterable[Path],
    destination: Path,
    include_patterns: Sequence[str] = (),
) -> ResourceInventory:
    patterns = _normalize_resource_patterns(include_patterns)
    scanned = [_scan_resource_root(source) for source in sources]
    entries = [entry for _root, root_entries in scanned for entry in root_entries]

    for pattern in patterns:
        if not any(
            _resource_pattern_matches(pattern, entry.relative) for entry in entries
        ):
            raise BundleError(f"resource include pattern matched nothing: {pattern!r}")

    selected_by_root: list[set[PurePosixPath]] = []
    for _root, root_entries in scanned:
        selected = {
            entry.relative
            for entry in root_entries
            if _resource_entry_selected(entry, patterns)
        }
        _validate_selected_symlinks(root_entries, selected)
        selected_by_root.append(selected)

    owners: dict[PurePosixPath, ResourceEntry] = {}
    for (_root, root_entries), selected in zip(scanned, selected_by_root, strict=True):
        for entry in root_entries:
            if entry.relative not in selected:
                continue
            previous = owners.get(entry.relative)
            if previous is None:
                owners[entry.relative] = entry
                continue
            if previous.kind == entry.kind == "directory":
                continue
            raise BundleError(
                "resource roots contain a selected path collision: "
                f"{previous.root} and {entry.root} both provide {entry.relative}"
            )

    for (_root, root_entries), selected in zip(scanned, selected_by_root, strict=True):
        required_directories: set[PurePosixPath] = set()
        for entry in root_entries:
            if entry.relative not in selected:
                continue
            required_directories.update(
                parent for parent in entry.relative.parents if parent.parts
            )
            if entry.kind == "directory":
                required_directories.add(entry.relative)
        for relative in sorted(
            required_directories, key=lambda path: (len(path.parts), str(path))
        ):
            target = _destination_path(destination, relative)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise BundleError(f"resource path type collision: {target}")
            target.mkdir(exist_ok=True)
        for entry in root_entries:
            if entry.relative in selected and entry.kind != "directory":
                _copy_resource_entry(entry, destination)

    for path in destination.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise BundleError(f"staged resource symlink is broken: {path}") from exc
        if not _is_within(resolved, [destination.resolve()]):
            raise BundleError(f"staged resource symlink escapes the bundle: {path}")

    staged_files, staged_bytes, staged_symlinks = _staged_resource_counts(destination)
    source_files = sum(entry.kind == "file" for entry in entries)
    source_bytes = sum(entry.size for entry in entries if entry.kind == "file")
    source_symlinks = sum(entry.kind == "symlink" for entry in entries)
    selected_entries = [
        entry
        for (_root, root_entries), selected in zip(
            scanned, selected_by_root, strict=True
        )
        for entry in root_entries
        if entry.relative in selected
    ]
    selected_files = sum(entry.kind == "file" for entry in selected_entries)
    selected_bytes = sum(
        entry.size for entry in selected_entries if entry.kind == "file"
    )
    selected_symlinks = sum(entry.kind == "symlink" for entry in selected_entries)

    group_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    selected_entry_ids = {id(entry) for entry in selected_entries}
    for entry in entries:
        if entry.kind != "file":
            continue
        group = entry.relative.parts[0]
        values = group_counts[group]
        values[0] += 1
        values[1] += entry.size
        if id(entry) in selected_entry_ids:
            values[2] += 1
            values[3] += entry.size
    groups = tuple(
        ResourceGroupInventory(group, *group_counts[group])
        for group in sorted(group_counts)
    )
    return ResourceInventory(
        mode="allowlist" if patterns else "copy-all",
        includes=patterns,
        source_files=source_files,
        source_bytes=source_bytes,
        selected_files=selected_files,
        selected_bytes=selected_bytes,
        staged_files=staged_files,
        staged_bytes=staged_bytes,
        source_symlinks=source_symlinks,
        selected_symlinks=selected_symlinks,
        staged_symlinks=staged_symlinks,
        groups=groups,
    )


def _copy_libraries(libraries: dict[Path, set[str]], destination: Path) -> list[Path]:
    copied: list[Path] = []
    for source in sorted(libraries, key=str):
        names = sorted(libraries[source], key=lambda name: (len(name), name))
        canonical = destination / names[0]
        if canonical.exists() or canonical.is_symlink():
            raise BundleError(f"duplicate staged path: {canonical}")
        shutil.copy2(source, canonical, follow_symlinks=True)
        copied.append(canonical)
        for name in names[1:]:
            alias = destination / name
            if alias.exists() or alias.is_symlink():
                raise BundleError(f"duplicate staged path: {alias}")
            alias.symlink_to(canonical.name)
    return copied


def _strip(paths: Iterable[Path], strip_program: str) -> None:
    for path in paths:
        try:
            process = subprocess.run(
                [strip_program, "--strip-unneeded", str(path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            raise BundleError(
                f"cannot run strip program {strip_program!r}: {exc}"
            ) from exc
        if process.returncode != 0:
            raise BundleError(f"strip failed for {path}:\n{process.stdout.rstrip()}")


def _validate_staged(
    executable: Path,
    bin_dir: Path,
    expected_architecture: str,
    private_roots: Sequence[Path],
) -> None:
    _validate_elf(executable, "staged executable", expected_architecture)
    if not os.access(executable, os.X_OK):
        raise BundleError(f"staged executable is not executable: {executable}")
    # Ask the ELF interpreter to ignore every original RPATH/RUNPATH and use
    # only the staged bin plus the loader's normal system paths. Never classify
    # an arbitrary directory discovered through the original RUNPATH as a
    # system path: doing so would let an entirely omitted private root conceal
    # a broken deployment bundle. --list resolves the complete dependency graph
    # without executing slicer application code.
    resolved_bin = bin_dir.resolve(strict=True)
    dependencies = _hermetic_dependencies(executable, [resolved_bin])
    leaked = sorted(
        {
            dependency.path
            for dependency in dependencies
            if _is_within(dependency.path, private_roots)
            and not _is_within(dependency.path, [resolved_bin])
        },
        key=str,
    )
    if leaked:
        rendered = ", ".join(str(path) for path in leaked)
        raise BundleError(
            "staged executable still resolves dependencies from private build "
            f"roots: {rendered}"
        )


def stage_bundle(args: argparse.Namespace) -> BundleResult:
    executable = _resolve_executable(args.executable, args.arch)
    library_roots: list[Path] = []
    seen_roots: set[Path] = set()
    for root in args.library_root:
        if not root.is_dir():
            raise BundleError(
                f"library root does not exist or is not a directory: {root}"
            )
        resolved_root = root.resolve(strict=True)
        if resolved_root not in seen_roots:
            seen_roots.add(resolved_root)
            library_roots.append(resolved_root)

    # Keep the leaf lexical: Path.resolve() would turn an existing output
    # symlink into its target and make the replacement logic delete that target.
    output = Path(os.path.abspath(args.output))
    output_parent = output.parent
    if output.is_symlink() or output_parent.resolve() != output_parent:
        raise BundleError(f"refusing symlinked output path: {output}")
    if (
        output == Path(output.anchor)
        or output == output_parent
        or _is_within(Path.cwd().resolve(), [output])
        or _is_within(executable, [output])
        or any(
            _is_within(Path(os.path.abspath(source)), [output])
            or _is_within(source.resolve(), [output])
            for source in args.resources
        )
        or any(_is_within(root, [output]) for root in library_roots)
    ):
        raise BundleError(f"refusing unsafe output path: {output}")
    output_parent.mkdir(parents=True, exist_ok=True)

    libraries = _dependency_closure(executable, library_roots)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        bin_dir = staging / "bin"
        bin_dir.mkdir()
        staged_executable = bin_dir / args.name
        if staged_executable.name != args.name or args.name in {"", ".", ".."}:
            raise BundleError(f"unsafe executable output name: {args.name!r}")
        shutil.copy2(executable, staged_executable, follow_symlinks=True)

        resource_inventory = ResourceInventory(
            mode="copy-all",
            includes=(),
            source_files=0,
            source_bytes=0,
            selected_files=0,
            selected_bytes=0,
            staged_files=0,
            staged_bytes=0,
            source_symlinks=0,
            selected_symlinks=0,
            staged_symlinks=0,
            groups=(),
        )
        resources = staging / "resources"
        if args.resources:
            resources.mkdir()
            resource_inventory = _copy_resources(
                args.resources,
                resources,
                _configured_resource_patterns(args),
            )
        elif patterns := _configured_resource_patterns(args):
            patterns = _normalize_resource_patterns(patterns)
            raise BundleError(
                "resource include patterns require at least one --resources root: "
                + ", ".join(patterns)
            )

        staged_libraries = _copy_libraries(libraries, bin_dir)
        if args.strip:
            _strip([staged_executable, *staged_libraries], args.strip_program)
        _validate_staged(staged_executable, bin_dir, args.arch, library_roots)

        if output.is_symlink() or output_parent.resolve() != output_parent:
            raise BundleError(f"refusing symlinked output path: {output}")
        if output.is_file():
            output.unlink()
        elif output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        size = sum(
            path.stat().st_size
            for path in output.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        return BundleResult(
            executable=output / "bin" / args.name,
            architecture=args.arch,
            library_count=len(staged_libraries),
            byte_count=size,
            resources=resource_inventory,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        action="append",
        required=True,
        type=Path,
        help="executable candidate; repeat in fallback order",
    )
    parser.add_argument(
        "--name", required=True, help="executable name inside bundle/bin"
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=("x86-64", "arm64"),
        help="required architecture of the staged ELF executable",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="bundle output directory"
    )
    parser.add_argument(
        "--resources",
        action="append",
        default=[],
        type=Path,
        help="resource tree whose contents are copied to bundle/resources",
    )
    parser.add_argument(
        "--resource-include",
        action="append",
        default=None,
        help=(
            "relative POSIX glob/path to stage from every resource root; repeat "
            "to form an allowlist (explicit options override newline-delimited "
            "SLICER_RESOURCE_INCLUDES; with neither, all resources are copied)"
        ),
    )
    parser.add_argument(
        "--library-root",
        action="append",
        default=[],
        type=Path,
        help="build/dependency root eligible for shared-library bundling",
    )
    parser.add_argument("--strip", action="store_true", help="strip staged ELF files")
    parser.add_argument(
        "--strip-program",
        default=os.environ.get("STRIP", "strip"),
        help="strip executable to use (default: STRIP or strip)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the bundle and resource inventory as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = stage_bundle(args)
    except BundleError as exc:
        parser.exit(1, f"stage_bundle.py: error: {exc}\n")
    if args.json:
        print(json.dumps(result.as_dict(), sort_keys=True))
    else:
        resources = result.resources
        print(
            f"staged {result.executable} with {result.library_count} project "
            f"libraries ({result.byte_count / (1024 * 1024):.1f} MiB total "
            f"before archive compression); resources: {resources.staged_files} "
            f"files/{resources.staged_bytes / (1024 * 1024):.1f} MiB staged, "
            f"{resources.omitted_files} files/"
            f"{resources.omitted_bytes / (1024 * 1024):.1f} MiB omitted "
            f"({resources.mode})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
