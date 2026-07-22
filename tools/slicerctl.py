#!/usr/bin/env python3
"""Manifest-driven Docker builds and production-shaped slicer smoke tests."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = ROOT / ".work"
DEFAULT_BUILDER_IMAGE = "slicer-builds-builder:ubuntu24.04"
DEFAULT_SMOKE_IMAGE = "ghcr.io/simplyprint/site-cloud-slicer:latest"
DEFAULT_LOCAL_SMOKE_IMAGE = "slicer-builds-smoke:latest"
DEFAULT_BAMBU_INPUT = ROOT / "tests/integration/fixtures/calicat-bambu-v1.3mf"
SUPPORTED_ARCHITECTURES = {"x86-64", "arm64"}
SUPPORTED_FAMILIES = {"bambu", "orca", "prusa", "cura"}
SUPPORTED_CONTRACTS = {"bambu", "prusa", "cura"}
BACKEND_ENGINE_CONTRACTS = {
    "BambuStudio": "bambu",
    "CrealityPrint": "bambu",
    "Cura": "cura",
    "ElegooSlicer": "bambu",
    "OrcaSlicer": "bambu",
    "PrusaSlicer": "prusa",
    "SuperSlicer": "prusa",
}
SUPPORTED_BACKEND_ENGINES = set(BACKEND_ENGINE_CONTRACTS)
SUPPORTED_PROFILE_SOURCES = {
    "backend-example",
    "bundle-resources",
    "profiles-db",
    "profiles-direct",
}
FORWARDED_BUILD_ENV = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "CMAKE_GENERATOR",
    "SLICER_GUI",
    "SLICER_BUILD_DATE",
    "SLICER_PCH",
    "SLICER_RESOURCE_INCLUDES",
    "SLICER_STRIP",
    "ORCA_EXTRA_BUILD_ARGS",
    "ELEGOO_EXTRA_BUILD_ARGS",
    "ORCA_UPDATER_SIG_KEY",
    "CURA_CONAN_CONFIG_INSTALL",
    "CURA_CONAN_CONFIG_REF",
    "CURA_CONAN_CONFIG_URL",
)
BUILD_ENV_DEFAULTS = {
    "CMAKE_GENERATOR": "Ninja",
    "SLICER_GUI": "0",
    "SLICER_PCH": "ON",
    "SLICER_STRIP": "1",
}
VERSION_DATE_STAMP_SLICERS = {
    "AnycubicSlicerNext",
    "BambuStudio",
    "CrealityPrint",
    "ElegooSlicer",
    "OrcaSlicer",
    "QIDIStudio",
}
PCH_SLICERS = {
    "AnycubicSlicerNext",
    "ElegooSlicer",
    "OrcaSlicer",
    "PrusaSlicer",
    "SuperSlicer",
}
PACKAGING_ONLY_BUILD_ENV = {"SLICER_RESOURCE_INCLUDES", "SLICER_STRIP"}
SCOPED_BUILD_ENV_SLICERS = {
    "ORCA_EXTRA_BUILD_ARGS": {"AnycubicSlicerNext", "OrcaSlicer"},
    "ELEGOO_EXTRA_BUILD_ARGS": {"ElegooSlicer"},
    "ORCA_UPDATER_SIG_KEY": {
        "AnycubicSlicerNext",
        "ElegooSlicer",
        "OrcaSlicer",
    },
}
_IMAGE_IDENTITIES: dict[tuple[str, str], str] = {}
# Bump these only when orchestration changes alter cache compatibility. Hashing
# this whole controller would invalidate multi-hour dependency builds for help,
# smoke-test, or comment-only edits.
DEPENDENCY_CACHE_SCHEMA = "4"
BUILD_TREE_CACHE_SCHEMA = "3"
BUILD_CACHE_SCHEMA = "8"


@dataclass(frozen=True)
class Manifest:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def ref(self) -> str:
        return self.data["default_ref"]


@dataclass(frozen=True)
class ReleaseTarget:
    """One immutable stable tag selected for a historical build."""

    slicer: str
    ref: str
    version: str
    version_parts: tuple[int, ...]
    version_group: str
    expected_commit: str


def manifest_ref_specs(manifest: Manifest) -> list[dict[str, str | None]]:
    test_backend_version = manifest.data.get("test", {}).get("backend_version")
    result: list[dict[str, str | None]] = [
        {
            "ref": manifest.ref,
            "expected_commit": manifest.data.get("expected_commit"),
            "backend_version": test_backend_version,
        }
    ]
    result.extend(manifest.data.get("supported_refs", []))
    return result


def manifest_ref_spec(manifest: Manifest, ref: str) -> dict[str, str | None] | None:
    return next(
        (spec for spec in manifest_ref_specs(manifest) if spec["ref"] == ref), None
    )


def patch_verification_ref_specs(
    manifest: Manifest, include_head: bool
) -> list[dict[str, str | None]]:
    specs = manifest_ref_specs(manifest)
    ci = manifest.data.get("ci", {})
    enabled_nightly = any(
        isinstance(lane, dict)
        and lane.get("enabled", False)
        and lane.get("nightly", False)
        and lane.get("repository", manifest.data["repository"])
        == manifest.data["repository"]
        for lane in (
            ci.get("binary", {}) if isinstance(ci, dict) else {},
            ci.get("config", {}) if isinstance(ci, dict) else {},
        )
    )
    if (
        include_head
        and enabled_nightly
        and all(spec["ref"] != "HEAD" for spec in specs)
    ):
        specs.append(
            {
                "ref": "HEAD",
                "expected_commit": None,
                "backend_version": manifest.data.get("test", {}).get("backend_version"),
            }
        )
    return specs


def _require_keys(
    value: dict[str, Any], keys: Iterable[str], context: str, errors: list[str]
) -> None:
    for key in keys:
        if key not in value:
            errors.append(f"{context} is missing {key!r}")


def _is_safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _is_safe_git_ref(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*", value) is not None
        and ".." not in value
        and not value.endswith((".", "/"))
    )


def validate_manifest(manifest: Manifest) -> None:
    data = manifest.data
    errors: list[str] = []
    _require_keys(
        data,
        (
            "schema_version",
            "name",
            "repository",
            "default_ref",
            "family",
            "executable",
            "architectures",
            "backend_engine",
            "backend_supported",
            "capabilities",
            "test",
        ),
        str(manifest.path),
        errors,
    )
    if errors:
        raise SystemExit("Invalid slicer manifest:\n- " + "\n- ".join(errors))

    if data["schema_version"] != 1:
        errors.append(f"unsupported schema_version {data['schema_version']!r}")
    if data["name"] != manifest.directory.name:
        errors.append(
            f"name {data['name']!r} must match directory {manifest.directory.name!r}"
        )
    for key in ("repository", "default_ref", "executable", "backend_engine"):
        if not isinstance(data[key], str) or not data[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if not _is_safe_git_ref(data["default_ref"]):
        errors.append("default_ref is not a safe Git ref")
    expected_commit = data.get("expected_commit")
    if expected_commit is not None and (
        not isinstance(expected_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", expected_commit)
    ):
        errors.append("expected_commit must be a lowercase full Git object ID")
    if data["default_ref"] != "HEAD" and expected_commit is None:
        errors.append("non-HEAD default_ref requires expected_commit")
    supported_refs = data.get("supported_refs", [])
    if not isinstance(supported_refs, list):
        errors.append("supported_refs must be an array of tables")
    else:
        seen_refs = {data["default_ref"]}
        for index, item in enumerate(supported_refs):
            context = f"supported_refs[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{context} must be a table")
                continue
            unknown = set(item) - {"ref", "expected_commit", "backend_version"}
            if unknown:
                errors.append(
                    f"{context} has unknown keys: {', '.join(sorted(unknown))}"
                )
            _require_keys(item, ("ref", "expected_commit"), context, errors)
            item_ref = item.get("ref")
            if not _is_safe_git_ref(item_ref):
                errors.append(f"{context}.ref is not a safe Git ref")
            elif item_ref in seen_refs:
                errors.append(f"duplicate supported ref {item_ref!r}")
            else:
                seen_refs.add(item_ref)
            item_commit = item.get("expected_commit")
            if not isinstance(item_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40,64}", item_commit
            ):
                errors.append(
                    f"{context}.expected_commit must be a lowercase full Git object ID"
                )
            item_backend_version = item.get("backend_version")
            if item_backend_version is not None and (
                not isinstance(item_backend_version, str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.+-]*", item_backend_version
                )
            ):
                errors.append(f"{context}.backend_version must be a safe version")
    if Path(data["executable"]).name != data["executable"]:
        errors.append("executable must be a plain filename")
    if data["family"] not in SUPPORTED_FAMILIES:
        errors.append(f"unsupported family {data['family']!r}")
    if data["backend_engine"] not in SUPPORTED_BACKEND_ENGINES:
        errors.append(f"unsupported backend engine {data['backend_engine']!r}")

    architectures = data["architectures"]
    if (
        not isinstance(architectures, list)
        or not architectures
        or any(arch not in SUPPORTED_ARCHITECTURES for arch in architectures)
        or len(set(architectures)) != len(architectures)
    ):
        errors.append(
            "architectures must be a unique, non-empty list containing only "
            + ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        )
    if not isinstance(data["backend_supported"], bool):
        errors.append("backend_supported must be a boolean")

    capabilities = data["capabilities"]
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be a table")
    else:
        _require_keys(
            capabilities,
            ("binary", "config_dump", "thumbnail"),
            "capabilities",
            errors,
        )
        if any(not isinstance(value, bool) for value in capabilities.values()):
            errors.append("all capabilities must be booleans")

    test = data["test"]
    if not isinstance(test, dict):
        errors.append("test must be a table")
    else:
        _require_keys(test, ("contract", "profile_source"), "test", errors)
        contract = test.get("contract")
        source = test.get("profile_source")
        if contract not in SUPPORTED_CONTRACTS:
            errors.append(f"unsupported test contract {contract!r}")
        expected_contract = BACKEND_ENGINE_CONTRACTS.get(data["backend_engine"])
        if expected_contract is not None and contract != expected_contract:
            errors.append(
                f"backend engine {data['backend_engine']} requires "
                f"test contract {expected_contract!r}"
            )
        if (
            isinstance(capabilities, dict)
            and capabilities.get("thumbnail")
            and contract != "bambu"
        ):
            errors.append(
                "thumbnail capability requires the 3MF-producing bambu contract"
            )
        if source not in SUPPORTED_PROFILE_SOURCES:
            errors.append(f"unsupported profile source {source!r}")
        test_backend_version = test.get("backend_version")
        if test_backend_version is not None and (
            not isinstance(test_backend_version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", test_backend_version)
        ):
            errors.append("test.backend_version must be a safe non-empty version")
        if source == "profiles-db":
            _require_keys(
                test,
                ("profile_engine", "model_id", "machine_name", "machine_variant"),
                "profiles-db test",
                errors,
            )
            if not ({"process_name", "process_native_id"} & test.keys()):
                errors.append(
                    "profiles-db test needs process_name or process_native_id"
                )
            if not ({"filament_name", "filament_native_id"} & test.keys()):
                errors.append(
                    "profiles-db test needs filament_name or filament_native_id"
                )
            if not isinstance(test.get("model_id"), int) or test.get("model_id", 0) < 1:
                errors.append("profiles-db model_id must be a positive integer")
        elif source in {"profiles-direct", "bundle-resources"}:
            profile_keys = ("machine_profile", "filament_profile", "process_profile")
            _require_keys(test, profile_keys, f"{source} test", errors)
            for key in profile_keys:
                if key in test and not _is_safe_relative(test[key]):
                    errors.append(f"{key} must be a safe relative path")
            if source == "profiles-direct":
                _require_keys(test, ("profile_engine",), source, errors)
            else:
                _require_keys(test, ("profile_root",), source, errors)
                if "profile_root" in test and not _is_safe_relative(
                    test["profile_root"]
                ):
                    errors.append("profile_root must be a safe relative path")

    build = data.get("build", {})
    if not isinstance(build, dict):
        errors.append("build must be a table")
    else:
        if "dependency_marker" in build and not _is_safe_relative(
            build["dependency_marker"]
        ):
            errors.append("build.dependency_marker must be a safe relative path")
        for key in ("dependency_inputs", "source_dependency_inputs"):
            dependency_inputs = build.get(key, [])
            if (
                not isinstance(dependency_inputs, list)
                or any(not _is_safe_relative(value) for value in dependency_inputs)
                or len(set(dependency_inputs)) != len(dependency_inputs)
            ):
                errors.append(
                    f"build.{key} must be a unique list of safe relative paths"
                )
                continue
            if key == "dependency_inputs":
                for value in dependency_inputs:
                    path = manifest.directory / value
                    if not path.exists():
                        errors.append(
                            "build dependency input does not exist: "
                            f"{path.relative_to(ROOT)}"
                        )

    patch_config = data.get("patches", {})
    if not isinstance(patch_config, dict):
        errors.append("patches must be a table")
    else:
        unknown_patch_keys = set(patch_config) - {"shared", "binary_ref_aliases"}
        if unknown_patch_keys:
            errors.append(
                "patches has unknown keys: " + ", ".join(sorted(unknown_patch_keys))
            )
        shared = patch_config.get("shared", [])
        if (
            not isinstance(shared, list)
            or any(
                not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                for value in shared
            )
            or len(set(shared)) != len(shared)
        ):
            errors.append(
                "patches.shared must be a unique list of shared patch-set names"
            )
        else:
            for value in shared:
                path = ROOT / "patches" / value
                if not path.is_dir():
                    errors.append(
                        f"shared patch set does not exist: {path.relative_to(ROOT)}"
                    )
        binary_ref_aliases = patch_config.get("binary_ref_aliases", {})
        if not isinstance(binary_ref_aliases, dict):
            errors.append("patches.binary_ref_aliases must be a table")
        else:
            for source_ref, patch_ref in binary_ref_aliases.items():
                if not _is_safe_git_ref(source_ref) or not _is_safe_git_ref(patch_ref):
                    errors.append(
                        "patches.binary_ref_aliases must map safe Git refs to safe Git refs"
                    )
                elif source_ref == patch_ref:
                    errors.append(
                        f"patches.binary_ref_aliases contains a self-alias for {source_ref}"
                    )

    for step in ("build-deps", "build", "package-binary"):
        path = manifest.directory / "steps" / f"{step}.sh"
        if not path.is_file():
            errors.append(f"missing required step script {path.relative_to(ROOT)}")

    if errors:
        raise SystemExit(
            f"Invalid slicer manifest {manifest.path.relative_to(ROOT)}:\n- "
            + "\n- ".join(errors)
        )


def manifests() -> dict[str, Manifest]:
    discovered: dict[str, Manifest] = {}
    for path in sorted((ROOT / "slicers").glob("*/slicer.toml")):
        with path.open("rb") as source:
            manifest = Manifest(path=path, data=tomllib.load(source))
        validate_manifest(manifest)
        if manifest.name in discovered:
            raise SystemExit(f"Duplicate slicer name: {manifest.name}")
        discovered[manifest.name] = manifest
    if not discovered:
        raise SystemExit("No slicer manifests found")
    return discovered


def get_manifest(name: str) -> Manifest:
    available = manifests()
    try:
        return available[name]
    except KeyError:
        choices = ", ".join(available)
        raise SystemExit(f"Unknown slicer {name!r}; choose one of: {choices}") from None


def run(
    command: Iterable[str | Path],
    *,
    cwd: Path = ROOT,
    capture: bool = False,
    timeout_seconds: int | None = None,
) -> str:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), file=sys.stderr)
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        timeout=timeout_seconds,
    )
    return completed.stdout.strip() if capture else ""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def file_lock(key: str) -> Iterator[None]:
    lock_root = WORK_ROOT / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    path = lock_root / f"{safe_name(key)[:48]}-{digest}.lock"
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def container_cli(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    configured = os.getenv("SLICER_CONTAINER_CLI")
    if configured:
        return configured
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit("Neither docker nor podman is installed; set SLICER_CONTAINER_CLI")


def container_image_identity(cli: str, image: str) -> str:
    key = (cli, image)
    if key not in _IMAGE_IDENTITIES:
        try:
            _IMAGE_IDENTITIES[key] = run(
                [cli, "image", "inspect", "--format", "{{.Id}}", image],
                capture=True,
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"Container image {image!r} is unavailable; build or pull it first"
            ) from error
    return _IMAGE_IDENTITIES[key]


def is_podman_cli(cli: str) -> bool:
    executable = shutil.which(cli) if Path(cli).name == cli else cli
    if not executable:
        return False
    return Path(executable).resolve().name == "podman"


def container_run_identity_options(cli: str) -> list[str]:
    options = ["--user", f"{os.getuid()}:{os.getgid()}"]
    if is_podman_cli(cli):
        # Rootless Podman otherwise presents host-owned bind mounts as uid 0 in
        # the container, while the explicitly selected host uid cannot write
        # them. keep-id provides Docker-equivalent ownership semantics.
        options[:0] = ["--security-opt", "label=disable", "--userns=keep-id"]
    return options


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "ref"


def storage_name(value: str) -> str:
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{safe_name(value)}-{suffix}"


def bounded_storage_name(value: str, max_length: int) -> str:
    """Keep a readable prefix without truncating the uniqueness suffix."""

    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    separator_length = 1
    prefix_length = max_length - len(suffix) - separator_length
    if prefix_length < 1:
        raise ValueError("bounded storage names need room for a prefix and hash")
    prefix = safe_name(value)[:prefix_length].rstrip("-.") or "r"
    return f"{prefix}-{suffix}"


def native_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86-64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise SystemExit(f"Unsupported native build architecture: {machine}")


def available_build_memory_bytes() -> int | None:
    candidates: list[int] = []
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if value != "max" and value.isdigit() and int(value) > 0:
            candidates.append(int(value))

    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2 and fields[1].isdigit():
                    candidates.append(int(fields[1]) * 1024)
                break
    except OSError:
        pass
    return min(candidates) if candidates else None


def default_build_jobs(
    cpu_count: int | None = None, memory_bytes: int | None = None
) -> int:
    cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    available = (
        memory_bytes if memory_bytes is not None else available_build_memory_bytes()
    )
    if available is None:
        return cpus
    # Match the conservative 2.5 GiB/job limit used by upstream Bambu drivers.
    memory_jobs = max(1, available // (2560 * 1024 * 1024))
    return min(cpus, memory_jobs)


def _patch_directory(root: Path, ref: str) -> Path:
    candidate = (root / ref).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"Unsafe patch ref path: {ref!r}") from None
    return candidate


def version_patch_directory(root: Path, ref: str) -> Path:
    exact = _patch_directory(root, ref)
    if ref != "HEAD" or (exact.is_dir() and any(exact.glob("*.patch"))):
        return exact
    return root / "nightly"


def manifest_patch_roots(manifest: Manifest) -> list[Path]:
    shared = manifest.data.get("patches", {}).get("shared", [])
    return [
        *(ROOT / "patches" / value for value in shared),
        manifest.directory / "patches",
    ]


def patches_in_directories(directories: Sequence[Path]) -> list[Path]:
    return [
        path
        for directory in directories
        if directory.is_dir()
        for path in sorted(directory.glob("*.patch"))
    ]


def common_patch_files(root: Path, ref: str) -> list[Path]:
    common_root = root / "common"
    return patches_in_directories(
        (common_root / "all", version_patch_directory(common_root, ref))
    )


def legacy_patch_files(manifest: Manifest, ref: str, mode: str) -> list[Path]:
    if mode not in {"binary", "dump"}:
        return []

    roots = manifest_patch_roots(manifest)
    patches: list[Path] = []
    if mode == "binary":
        patch_ref = (
            manifest.data.get("patches", {})
            .get("binary_ref_aliases", {})
            .get(ref, ref)
        )
        for root in roots:
            patches.extend(common_patch_files(root, ref))
            binary_root = root / "binary"
            patches.extend(
                patches_in_directories(
                    (
                        binary_root / "all",
                        version_patch_directory(binary_root, patch_ref),
                    )
                )
            )
        return patches

    for root in roots:
        patches.extend(common_patch_files(root, ref))
        version_directory = version_patch_directory(root, ref)
        patches.extend(patches_in_directories((root / "all", version_directory)))
        # A version directory containing an unrelated fix must not suppress
        # that root's legacy dump fallback.
        if not (version_directory / "dump_configs.patch").is_file():
            fallback = root / "dump_configs.patch"
            if fallback.is_file():
                patches.append(fallback)
    return patches


def patch_variant(mode: str, patch_files: Sequence[Path]) -> str:
    if not patch_files:
        return "unpatched"
    digest = hashlib.sha256()
    for index, path in enumerate(patch_files):
        _hash_file(digest, f"patch/{index}", path)
    return f"{mode}-{digest.hexdigest()[:12]}"


def prepare_source(
    manifest: Manifest,
    ref: str,
    seed: Path | None,
    variant: str = "unpatched",
    expected_commit: str | None = None,
) -> tuple[Path, str]:
    if expected_commit is not None and re.fullmatch(
        r"[0-9a-f]{40,64}", expected_commit
    ) is None:
        raise SystemExit(
            f"Invalid transient expected commit for {manifest.name}: {expected_commit!r}"
        )
    mirror = WORK_ROOT / "git" / f"{manifest.name}.git"
    mirror.parent.mkdir(parents=True, exist_ok=True)

    with file_lock(f"mirror:{manifest.name}"):
        mirror_created = False
        if not mirror.exists():
            source = seed.resolve() if seed else manifest.data["repository"]
            run(["git", "clone", "--mirror", source, mirror])
            mirror_created = True

        # A local seed is only a transport. Reconcile this on every use so a
        # changed manifest URL or an old seeded mirror cannot resolve refs from
        # a stale upstream.
        run(
            ["git", "remote", "set-url", "origin", manifest.data["repository"]],
            cwd=mirror,
        )
        if not mirror_created:
            if seed:
                run(
                    [
                        "git",
                        "fetch",
                        "--no-prune",
                        "--tags",
                        seed.resolve(),
                        "+refs/*:refs/*",
                    ],
                    cwd=mirror,
                )
            else:
                run(["git", "fetch", "--prune", "--tags", "origin"], cwd=mirror)
            if ref == "HEAD":
                refresh_mirror_head(mirror, seed.resolve() if seed else "origin")

        try:
            commit = run(
                ["git", "rev-parse", f"{ref}^{{commit}}"], cwd=mirror, capture=True
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Cannot resolve {manifest.name} ref {ref!r}") from error
        declared_ref = manifest_ref_spec(manifest, ref)
        declared_commit = (
            declared_ref.get("expected_commit") if declared_ref is not None else None
        )
        if expected_commit and declared_commit and expected_commit != declared_commit:
            raise SystemExit(
                f"{manifest.name} ref {ref!r} has conflicting commit locks: "
                f"{declared_commit} and {expected_commit}"
            )
        locked_commit = expected_commit or declared_commit
        if locked_commit and commit != locked_commit:
            raise SystemExit(
                f"{manifest.name} ref {ref!r} resolved to {commit}, "
                f"expected locked commit {locked_commit}"
            )

    checkout = (
        WORK_ROOT
        / "checkouts"
        / manifest.name
        / f"{storage_name(ref)}-{safe_name(variant)}-{commit[:12]}"
    )
    with file_lock(f"checkout-create:{checkout}"):
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            # A local clone uses hardlinks but retains a self-contained .git
            # directory, so Git and submodules work after Docker remounts it.
            run(["git", "clone", "--no-checkout", mirror, checkout])
            run(
                ["git", "remote", "set-url", "origin", manifest.data["repository"]],
                cwd=checkout,
            )
            run(["git", "checkout", "--detach", commit], cwd=checkout)
        elif not (checkout / ".git").is_dir():
            raise SystemExit(
                f"Existing source is not a self-contained Git clone: {checkout}"
            )
        elif run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True) != commit:
            raise SystemExit(f"Existing checkout has an unexpected commit: {checkout}")
    return checkout, commit


def refresh_mirror_head(mirror: Path, transport: str | Path) -> None:
    """Make a cached mirror's HEAD match the transport's advertised HEAD."""

    advertisement = run(
        ["git", "ls-remote", "--symref", transport, "HEAD"],
        cwd=mirror,
        capture=True,
    )
    head_ref: str | None = None
    head_commit: str | None = None
    for line in advertisement.splitlines():
        value, separator, name = line.partition("\t")
        if separator != "\t" or name != "HEAD":
            continue
        if value.startswith("ref: "):
            head_ref = value.removeprefix("ref: ")
        elif re.fullmatch(r"[0-9a-f]{40,64}", value):
            head_commit = value
    if head_commit is None:
        raise SystemExit(f"Transport did not advertise a valid HEAD: {transport}")

    if head_ref is not None:
        if not head_ref.startswith("refs/heads/"):
            raise SystemExit(
                f"Transport advertised an unsafe HEAD symref {head_ref!r}: {transport}"
            )
        local_commit = run(
            ["git", "rev-parse", f"{head_ref}^{{commit}}"],
            cwd=mirror,
            capture=True,
        )
        if local_commit != head_commit:
            raise SystemExit(
                f"Fetched {head_ref} at {local_commit}, but {transport} advertised "
                f"{head_commit}"
            )
        run(["git", "symbolic-ref", "HEAD", head_ref], cwd=mirror)
        return

    run(["git", "cat-file", "-e", f"{head_commit}^{{commit}}"], cwd=mirror)
    run(["git", "update-ref", "--no-deref", "HEAD", head_commit], cwd=mirror)


def prepare_patch_verification_mirror(
    manifest: Manifest, seed: Path | None = None
) -> Path:
    """Prepare one bare mirror without creating or mutating a checkout."""

    mirror = WORK_ROOT / "git" / f"{manifest.name}.git"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(f"mirror:{manifest.name}"):
        mirror_created = False
        if not mirror.exists():
            source = seed.resolve() if seed else manifest.data["repository"]
            run(["git", "clone", "--mirror", source, mirror])
            mirror_created = True
        elif not mirror.is_dir():
            raise SystemExit(f"Git mirror path is not a directory: {mirror}")

        run(
            ["git", "remote", "set-url", "origin", manifest.data["repository"]],
            cwd=mirror,
        )
        if not mirror_created:
            if seed:
                run(
                    [
                        "git",
                        "fetch",
                        "--no-prune",
                        "--tags",
                        seed.resolve(),
                        "+refs/*:refs/*",
                    ],
                    cwd=mirror,
                )
            else:
                run(["git", "fetch", "--prune", "--tags", "origin"], cwd=mirror)
            refresh_mirror_head(mirror, seed.resolve() if seed else "origin")
    return mirror


def resolve_patch_verification_ref(mirror: Path, ref: str) -> tuple[str | None, str]:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=mirror,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None, _patch_git_error(completed)
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None, f"Git returned an invalid commit ID for {ref!r}: {commit!r}"
    return commit, ""


def _patch_git_error(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    if not detail:
        detail = f"git exited with status {completed.returncode}"
    # A malformed patch can make Git print a great deal of context. Keep JSON
    # and Actions logs useful while retaining the end of the diagnostic.
    return detail[-4000:]


def verify_patch_stack(
    mirror: Path, commit: str, patch_files: Sequence[Path]
) -> dict[str, str] | None:
    """Check one ordered patch stack against a commit using a private index."""

    with tempfile.TemporaryDirectory(prefix="slicer-patch-index-") as temporary:
        index = Path(temporary) / "index"
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(index)

        read_tree = subprocess.run(
            ["git", "read-tree", commit],
            cwd=mirror,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if read_tree.returncode != 0:
            return {"stage": "read-tree", "error": _patch_git_error(read_tree)}

        # Apply each patch after checking it so later patches see the exact
        # index state produced by earlier patches in the selected stack.
        for patch_file in patch_files:
            check = subprocess.run(
                ["git", "apply", "--cached", "--check", patch_file],
                cwd=mirror,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if check.returncode != 0:
                return {
                    "stage": "check",
                    "patch": str(patch_file),
                    "error": _patch_git_error(check),
                }
            apply = subprocess.run(
                ["git", "apply", "--cached", "--whitespace=fix", patch_file],
                cwd=mirror,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if apply.returncode != 0:
                return {
                    "stage": "apply",
                    "patch": str(patch_file),
                    "error": _patch_git_error(apply),
                }
    return None


def apply_legacy_patches(source: Path, patch_files: Sequence[Path]) -> None:
    if not patch_files:
        return

    selection = {
        "schema_version": 1,
        "variant": patch_variant("selected", patch_files),
        # The content-derived variant is authoritative. Basenames keep the
        # marker useful to humans without coupling a checkout to this repo's
        # absolute location.
        "patches": [path.name for path in patch_files],
    }
    marker = source / ".git" / "slicer-builds-patches.json"
    if marker.is_file():
        if load_json(marker) != selection:
            raise SystemExit(
                f"Managed checkout has a different patch selection: {source}"
            )
        print(
            f"Patch selection already applied: {selection['variant']}",
            file=sys.stderr,
        )
        return

    index_dirty = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD"], cwd=source
    ).returncode
    worktree_dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"], cwd=source
    ).returncode
    if index_dirty or worktree_dirty:
        raise SystemExit(
            "Managed checkout was modified before its patch selection was recorded; "
            f"remove the disposable checkout and retry: {source}"
        )

    # Applying to the index as well as the worktree makes added, deleted, and
    # renamed files part of the exact cached source state. One invocation keeps
    # the complete ordered patch stack atomic.
    run(
        ["git", "apply", "--index", "--whitespace=fix", *patch_files],
        cwd=source,
    )
    atomic_json(marker, selection)


def patch_touched_paths(patch_files: Sequence[Path]) -> set[str]:
    touched: set[str] = set()
    for patch in patch_files:
        output = subprocess.run(
            ["git", "apply", "--numstat", "-z", patch],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        records = output.split(b"\0")
        index = 0
        while index < len(records) - 1:
            fields = records[index].split(b"\t", 2)
            if len(fields) != 3:
                raise SystemExit(f"Cannot parse patch paths from {patch}")
            raw_path = fields[2]
            if raw_path:
                touched.add(raw_path.decode(errors="surrogateescape"))
                index += 1
                continue
            # With -z, rename/copy records put the old and new path in the next
            # two NUL-delimited fields instead of rendering an ambiguous
            # ``old => new`` pathname.
            if index + 2 >= len(records) - 1:
                raise SystemExit(f"Cannot parse renamed patch paths from {patch}")
            touched.update(
                record.decode(errors="surrogateescape")
                for record in records[index + 1 : index + 3]
            )
            index += 3
        # git apply --numstat reports only the destination for a pure rename on
        # some Git versions. Extended headers preserve both sides.
        with patch.open("rb") as source:
            for line in source:
                for prefix in (
                    b"rename from ",
                    b"rename to ",
                    b"copy from ",
                    b"copy to ",
                ):
                    if line.startswith(prefix):
                        touched.add(
                            line.removeprefix(prefix)
                            .rstrip(b"\r\n")
                            .decode(errors="surrogateescape")
                        )
                        break
    return touched


def source_state(source: Path, patch_files: Sequence[Path]) -> str:
    """Validate tracked edits and identify the exact source fed to the build."""
    changed_raw = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "HEAD"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    changed = {
        value.decode(errors="surrogateescape")
        for value in changed_raw.split(b"\0")
        if value
    }
    unexpected = changed - patch_touched_paths(patch_files)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise SystemExit(
            f"Managed checkout has tracked edits outside the selected patches: {names}. "
            f"Remove the disposable checkout and retry: {source}"
        )

    worktree_raw = subprocess.run(
        ["git", "diff", "--name-only", "-z"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    worktree_changes = [
        value.decode(errors="surrogateescape")
        for value in worktree_raw.split(b"\0")
        if value
    ]
    if worktree_changes:
        names = ", ".join(sorted(worktree_changes))
        raise SystemExit(
            f"Managed checkout worktree differs from its selected patch state: {names}. "
            f"Remove the disposable checkout and retry: {source}"
        )

    # A path deleted in the index can reappear as an untracked file and would
    # not be reported by git diff. Assert that every patch path has the same
    # existence state as the index.
    for relative in patch_touched_paths(patch_files):
        in_index = (
            subprocess.run(
                ["git", "cat-file", "-e", f":{relative}"],
                cwd=source,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if in_index != os.path.lexists(source / relative):
            raise SystemExit(
                "Managed checkout path differs from its selected patch state: "
                f"{relative}"
            )

    difference = subprocess.run(
        ["git", "diff", "--cached", "--binary", "HEAD"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(difference).hexdigest()


def restore_managed_worktree(source: Path) -> None:
    """Undo tracked mutations made by upstream build drivers, preserving patches."""
    run(["git", "restore", "--worktree", ":/"], cwd=source)


def _hash_literal(digest: Any, label: str, value: str) -> None:
    digest.update(b"literal\0")
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(value.encode())
    digest.update(b"\0")


def _hash_file(digest: Any, label: str, path: Path) -> None:
    if path.is_symlink():
        _hash_literal(digest, label, f"symlink:{os.readlink(path)}")
        return
    if not path.is_file():
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    digest.update(f"file\0{label}\0{mode:o}\0".encode())
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    digest.update(b"\0")


def _hash_path(digest: Any, label: str, path: Path) -> None:
    if path.is_symlink() or path.is_file():
        _hash_file(digest, label, path)
        return
    if not path.is_dir():
        _hash_literal(digest, label, "missing")
        return
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path)
        if {".git", "__pycache__", "build"} & set(relative.parts):
            continue
        _hash_file(digest, f"{label}/{relative}", child)


def _hash_index_paths(
    digest: Any, label: str, source: Path, pathspecs: Sequence[str]
) -> None:
    """Hash tracked index state without generated/download cache contents."""
    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", *pathspecs],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    )
    _hash_literal(
        digest,
        label,
        hashlib.sha256(completed.stdout).hexdigest(),
    )


def resolve_remote_git_ref(repository: str, ref: str) -> str:
    """Resolve a symbolic Git ref to one unambiguous immutable commit."""
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", ref):
        return ref.lower()
    try:
        process = subprocess.run(
            ["git", "ls-remote", "--exit-code", repository, ref, f"{ref}^{{}}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(
            f"Cannot resolve Git ref {ref!r} from {repository}: {exc}"
        ) from exc
    if process.returncode != 0:
        detail = process.stderr.strip() or "no matching advertised ref"
        raise SystemExit(f"Cannot resolve Git ref {ref!r} from {repository}: {detail}")

    candidates: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in process.stdout.splitlines():
        fields = line.split("\t", 1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{40,64}", fields[0]):
            raise SystemExit(f"Cannot parse Git ref {ref!r} returned by {repository}")
        object_id, name = fields
        if name.endswith("^{}"):
            peeled[name.removesuffix("^{}")] = object_id.lower()
        else:
            candidates[name] = object_id.lower()
    candidates.update(peeled)
    if len(candidates) != 1:
        names = ", ".join(sorted(candidates)) or "none"
        raise SystemExit(
            f"Git ref {ref!r} is ambiguous at {repository}; matches: {names}"
        )
    return next(iter(candidates.values()))


_STABLE_RELEASE_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")


def clean_release_version(ref: str) -> str:
    """Apply the binary downloader's release-prefix normalization."""

    if ref.startswith("version_"):
        return ref.removeprefix("version_")
    if ref.startswith("v"):
        return ref.removeprefix("v")
    return ref


def release_version_parts(ref: str) -> tuple[int, ...] | None:
    """Return numeric stable-version components, excluding prerelease tags."""

    clean = clean_release_version(ref)
    if _STABLE_RELEASE_VERSION_RE.fullmatch(clean) is None:
        return None
    return tuple(int(part) for part in clean.split("."))


def _release_version_sort_key(parts: tuple[int, ...]) -> tuple[int, ...]:
    # Accepted versions contain at most four components. Padding makes 2.4 and
    # 2.4.0 compare as the same numeric release instead of relying on Python's
    # shorter-tuple ordering.
    return parts + (0,) * (4 - len(parts))


def _release_group(ref: str, parts: tuple[int, ...]) -> tuple[tuple[int, ...], str]:
    """Return the downloader-compatible first-three-component grouping."""

    component_count = min(3, len(parts))
    clean_components = clean_release_version(ref).split(".")
    return parts[:component_count], ".".join(clean_components[:component_count])


def select_release_tags(
    tags: Sequence[str],
    *,
    maximum: int | None = 3,
    minimum: str | None = None,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Select stable releases using the downloader's grouping and ordering.

    Four-part build revisions sharing their first three numeric components are
    collapsed to the highest build. Ordinary three-part versions remain
    distinct. The requested maximum is then applied newest-first.
    """

    if maximum is not None and maximum < 1:
        raise SystemExit("--max-versions must be a positive integer")
    minimum_parts = release_version_parts(minimum) if minimum is not None else None
    if minimum is not None and minimum_parts is None:
        raise SystemExit(f"Invalid stable minimum version: {minimum!r}")
    minimum_key = (
        _release_version_sort_key(minimum_parts)
        if minimum_parts is not None
        else None
    )

    # Match the downloader exactly: the grouping key is the normalized text,
    # not a numeric tuple. This intentionally keeps differently zero-padded
    # spellings in separate groups and preserves the first equal candidate.
    grouped: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for ref in tags:
        parts = release_version_parts(ref)
        if parts is None:
            continue
        if minimum_key is not None and _release_version_sort_key(parts) < minimum_key:
            continue
        _numeric_group, display_group = _release_group(ref, parts)
        candidate = (ref, display_group, parts)
        previous = grouped.get(display_group)
        if previous is None or parts > previous[2]:
            grouped[display_group] = candidate

    selected = sorted(
        grouped.values(),
        key=lambda item: item[2],
        reverse=True,
    )
    return selected if maximum is None else selected[:maximum]


def github_repository_slug(repository: str) -> str:
    """Extract ``owner/repository`` from a GitHub HTTPS or SSH remote."""

    if repository.startswith("git@github.com:"):
        path = repository.removeprefix("git@github.com:")
    else:
        parsed = urllib.parse.urlparse(repository)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            raise SystemExit(
                f"Historical release discovery requires a GitHub repository: {repository}"
            )
        path = parsed.path.lstrip("/")
    path = path.removesuffix(".git").strip("/")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path) is None:
        raise SystemExit(f"Cannot derive a GitHub repository name from {repository}")
    return path


def github_stable_release_tags(manifest: Manifest) -> list[str]:
    """Enumerate every published, non-prerelease GitHub release tag."""

    slug = github_repository_slug(manifest.data["repository"])
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "simplyprint-slicer-builds",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    tags: list[str] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{slug}/releases?per_page=100&page={page}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except (
            OSError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
        ) as error:
            detail = getattr(error, "reason", None) or str(error)
            raise SystemExit(
                f"Cannot enumerate GitHub releases for {manifest.name}: {detail}"
            ) from error
        if not isinstance(payload, list):
            raise SystemExit(
                f"GitHub returned an invalid releases response for {manifest.name}"
            )
        for release in payload:
            if not isinstance(release, dict):
                continue
            if release.get("draft") or release.get("prerelease"):
                continue
            tag = release.get("tag_name")
            if isinstance(tag, str) and release_version_parts(tag) is not None:
                tags.append(tag)
        if len(payload) < 100:
            break
        page += 1
    return tags


def resolve_release_tag_commits(
    manifest: Manifest, refs: Sequence[str]
) -> dict[str, str]:
    """Resolve selected tags to peeled commits with one remote advertisement."""

    if not refs:
        return {}
    requested = set(refs)
    patterns = [
        pattern
        for ref in refs
        for pattern in (f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}")
    ]
    try:
        process = subprocess.run(
            ["git", "ls-remote", "--exit-code", manifest.data["repository"], *patterns],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(
            f"Cannot resolve release tags for {manifest.name}: {error}"
        ) from error
    if process.returncode != 0:
        detail = process.stderr.strip() or "no selected tags were advertised"
        raise SystemExit(f"Cannot resolve release tags for {manifest.name}: {detail}")

    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in process.stdout.splitlines():
        fields = line.split("\t", 1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-fA-F]{40,64}", fields[0]) is None:
            raise SystemExit(
                f"Cannot parse advertised release tags for {manifest.name}"
            )
        object_id, full_ref = fields
        if not full_ref.startswith("refs/tags/"):
            continue
        tag = full_ref.removeprefix("refs/tags/")
        destination = direct
        if tag.endswith("^{}"):
            tag = tag.removesuffix("^{}")
            destination = peeled
        if tag in requested:
            destination[tag] = object_id.lower()

    missing = sorted(requested - direct.keys())
    if missing:
        raise SystemExit(
            f"Selected release tags disappeared for {manifest.name}: {', '.join(missing)}"
        )
    return {ref: peeled.get(ref, direct[ref]) for ref in refs}


def git_stable_release_tags(manifest: Manifest) -> tuple[list[str], dict[str, str]]:
    """Enumerate stable numeric tags and peeled objects from any Git remote."""

    try:
        process = subprocess.run(
            ["git", "ls-remote", "--tags", manifest.data["repository"]],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(
            f"Cannot enumerate Git tags for {manifest.name}: {error}"
        ) from error
    if process.returncode != 0:
        detail = process.stderr.strip() or f"git ls-remote exited {process.returncode}"
        raise SystemExit(f"Cannot enumerate Git tags for {manifest.name}: {detail}")

    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in process.stdout.splitlines():
        fields = line.split("\t", 1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-fA-F]{40,64}", fields[0]) is None:
            raise SystemExit(f"Cannot parse advertised Git tags for {manifest.name}")
        object_id, full_ref = fields
        if not full_ref.startswith("refs/tags/"):
            continue
        tag = full_ref.removeprefix("refs/tags/")
        destination = direct
        if tag.endswith("^{}"):
            tag = tag.removesuffix("^{}")
            destination = peeled
        if release_version_parts(tag) is not None:
            destination[tag] = object_id.lower()

    tags = list(direct)
    return tags, {tag: peeled.get(tag, direct[tag]) for tag in tags}


def prefer_manifest_release_aliases(
    manifest: Manifest, tags: Sequence[str], commits: Mapping[str, str]
) -> list[str]:
    """Prefer manifest-declared spellings when a remote advertises tag aliases."""

    preferred: list[str] = []
    for ref_spec in manifest_ref_specs(manifest):
        ref = ref_spec.get("ref")
        if isinstance(ref, str) and ref in commits and ref not in preferred:
            preferred.append(ref)
    return preferred + [tag for tag in tags if tag not in preferred]


def historical_release_targets(
    manifest: Manifest,
    *,
    maximum: int | None = 3,
    minimum: str | None = None,
    source: str = "index",
) -> list[ReleaseTarget]:
    known_commits: dict[str, str] = {}
    if source == "index":
        index_path = manifest.directory / "out" / "_index.json"
        if not index_path.is_file():
            return []
        try:
            index = load_json(index_path)
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"Cannot read release index for {manifest.name}") from error
        versions = index.get("versions") if isinstance(index, dict) else None
        if not isinstance(versions, dict):
            raise SystemExit(f"Invalid release index for {manifest.name}: {index_path}")
        tags = [ref for ref in versions if ref != "nightly"]
        for ref, value in versions.items():
            commit = value.get("upstream_ref") if isinstance(value, dict) else None
            if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit):
                known_commits[ref] = commit
    elif source == "github-releases":
        tags = github_stable_release_tags(manifest)
    elif source == "git-tags":
        tags, known_commits = git_stable_release_tags(manifest)
        tags = prefer_manifest_release_aliases(manifest, tags, known_commits)
    else:
        raise SystemExit(f"Unknown historical release source: {source}")

    selected = select_release_tags(tags, maximum=maximum, minimum=minimum)
    refs = [ref for ref, _group, _parts in selected]
    for ref_spec in manifest_ref_specs(manifest):
        ref = ref_spec.get("ref")
        commit = ref_spec.get("expected_commit")
        if ref in refs and isinstance(commit, str):
            recorded_commit = known_commits.get(str(ref))
            if recorded_commit is not None and recorded_commit != commit:
                raise SystemExit(
                    f"Release index and manifest disagree for {manifest.name} "
                    f"{ref}: {recorded_commit} != {commit}"
                )
            known_commits[str(ref)] = commit
    unresolved = [ref for ref in refs if ref not in known_commits]
    if unresolved:
        known_commits.update(resolve_release_tag_commits(manifest, unresolved))
    return [
        ReleaseTarget(
            slicer=manifest.name,
            ref=ref,
            version=clean_release_version(ref),
            version_parts=parts,
            version_group=group,
            expected_commit=known_commits[ref],
        )
        for ref, group, parts in selected
    ]


def effective_build_environment(manifest: Manifest | None = None) -> dict[str, str]:
    result = {
        key: os.environ.get(key, BUILD_ENV_DEFAULTS.get(key, ""))
        for key in FORWARDED_BUILD_ENV
    }
    if manifest is not None and manifest.data["family"] == "cura":
        build = manifest.data.get("build", {})
        result["CURA_CONAN_CONFIG_REF"] = os.environ.get(
            "CURA_CONAN_CONFIG_REF", build.get("conan_config_ref", "")
        )
        result["CURA_CONAN_CONFIG_URL"] = os.environ.get(
            "CURA_CONAN_CONFIG_URL", build.get("conan_config_url", "")
        )
        if result["CURA_CONAN_CONFIG_INSTALL"] != "0":
            result["CURA_CONAN_CONFIG_REF"] = resolve_remote_git_ref(
                result["CURA_CONAN_CONFIG_URL"], result["CURA_CONAN_CONFIG_REF"]
            )
    elif manifest is not None:
        result.pop("CURA_CONAN_CONFIG_INSTALL")
        result.pop("CURA_CONAN_CONFIG_REF")
        result.pop("CURA_CONAN_CONFIG_URL")
    if manifest is not None:
        if manifest.name not in VERSION_DATE_STAMP_SLICERS:
            result.pop("SLICER_BUILD_DATE")
        if manifest.name not in PCH_SLICERS:
            result.pop("SLICER_PCH")
        for key, slicers in SCOPED_BUILD_ENV_SLICERS.items():
            if manifest.name not in slicers:
                result.pop(key)
    if manifest is not None and manifest.data["family"] in {"bambu", "orca"}:
        # These upstream drivers select Ninja themselves and do not expose a
        # GUI-off build that can still render CLI thumbnails. Do not create
        # cache variants for ignored knobs.
        result.pop("CMAKE_GENERATOR")
        result.pop("SLICER_GUI")
    return result


def build_configuration_variant(image_identity: str, build_env: dict[str, str]) -> str:
    """Identify inputs CMake/Conan may persist inside a managed checkout."""
    digest = hashlib.sha256()
    _hash_literal(digest, "builder-image", image_identity)
    for key, value in sorted(build_env.items()):
        if key not in PACKAGING_ONLY_BUILD_ENV:
            _hash_literal(digest, f"env/{key}", value)
    return digest.hexdigest()[:12]


def dependency_fingerprint(
    manifest: Manifest,
    source: Path,
    arch: str,
    image_identity: str,
    build_env: dict[str, str],
) -> str:
    """Hash dependency inputs while allowing identical forks to share them."""
    digest = hashlib.sha256()
    _hash_literal(digest, "schema", DEPENDENCY_CACHE_SCHEMA)
    _hash_literal(digest, "arch", arch)
    _hash_literal(digest, "family", manifest.data["family"])
    _hash_literal(digest, "builder-image", image_identity)
    _hash_file(
        digest,
        "build-deps-step",
        manifest.directory / "steps" / "build-deps.sh",
    )
    build_config = manifest.data.get("build", {})
    for value in build_config.get("dependency_inputs", []):
        _hash_path(digest, f"dependency-input/{value}", manifest.directory / value)
    for name in ("BuildLinux.sh", "build_linux.sh", "build_release_linux.sh"):
        _hash_file(digest, f"upstream-driver/{name}", source / name)
    scripts = source / "scripts"
    if scripts.is_dir():
        for path in sorted(scripts.rglob("*")):
            relative = path.relative_to(scripts)
            lowered = path.name.lower()
            if "build" in lowered or "dep" in lowered:
                _hash_file(digest, f"upstream-driver/scripts/{relative}", path)
    dependency_env = [
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CMAKE_GENERATOR",
        "SLICER_GUI",
    ]
    if manifest.data["family"] == "cura":
        dependency_env.extend(
            [
                "CURA_CONAN_CONFIG_INSTALL",
                "CURA_CONAN_CONFIG_REF",
                "CURA_CONAN_CONFIG_URL",
            ]
        )
    for key in dependency_env:
        _hash_literal(digest, f"env/{key}", build_env.get(key, ""))
    source_dependency_inputs = (
        "CMakeLists.txt",
        "deps",
        "deps_src",
        "dependencies",
        "linux.d",
        "cmake",
        "CMakePresets.json",
        "CMakeUserPresets.json",
        "conandata.yml",
        "conanfile.py",
        "conanfile.txt",
        "pyproject.toml",
        "poetry.lock",
        "requirements*.txt",
        "scripts/linux.d",
        "vcpkg.json",
        "vcpkg-configuration.json",
        *build_config.get("source_dependency_inputs", []),
    )
    _hash_index_paths(
        digest,
        "dependency-index",
        source,
        source_dependency_inputs,
    )
    for pattern in ("conanfile.*", "conan*.lock"):
        for path in sorted(source.glob(pattern)):
            _hash_file(digest, f"source-root/{path.name}", path)
    return digest.hexdigest()


def build_tree_fingerprint(
    manifest: Manifest,
    commit: str,
    arch: str,
    image_identity: str,
    dependency_hash: str,
    source_state_hash: str,
    build_env: dict[str, str],
) -> str:
    """Identify inputs that may persist in a managed source build directory."""
    digest = hashlib.sha256()
    for label, value in (
        ("schema", BUILD_TREE_CACHE_SCHEMA),
        ("commit", commit),
        ("arch", arch),
        ("builder-image", image_identity),
        ("dependencies", dependency_hash),
        ("source-state", source_state_hash),
    ):
        _hash_literal(digest, label, value)
    for key, value in sorted(build_env.items()):
        if key not in PACKAGING_ONLY_BUILD_ENV:
            _hash_literal(digest, f"env/{key}", value)
    _hash_literal(
        digest,
        "manifest-build-config",
        json.dumps(
            {
                "name": manifest.name,
                "family": manifest.data["family"],
                "build": manifest.data.get("build", {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    build_step = manifest.directory / "steps" / "build.sh"
    _hash_file(digest, "step/build.sh", build_step)
    if "tools/build_cmake_target.sh" in build_step.read_text():
        _hash_file(
            digest,
            "tool/build-cmake-target",
            ROOT / "tools" / "build_cmake_target.sh",
        )
    if "tools/stamp_version_date.sh" in build_step.read_text():
        _hash_file(
            digest,
            "tool/stamp-version-date",
            ROOT / "tools" / "stamp_version_date.sh",
        )
    return digest.hexdigest()


def build_fingerprint(
    manifest: Manifest,
    commit: str,
    patch_files: Sequence[Path],
    arch: str,
    image_identity: str,
    dependency_hash: str,
    build_tree_hash: str,
    source_state_hash: str,
    build_env: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("schema", BUILD_CACHE_SCHEMA),
        ("commit", commit),
        ("arch", arch),
        ("builder-image", image_identity),
        ("dependencies", dependency_hash),
        ("build-tree", build_tree_hash),
        ("source-state", source_state_hash),
    ):
        _hash_literal(digest, label, value)
    for key, value in sorted(build_env.items()):
        _hash_literal(digest, f"env/{key}", value)
    _hash_literal(
        digest,
        "manifest-build-projection",
        json.dumps(
            {
                "name": manifest.name,
                "family": manifest.data["family"],
                "executable": manifest.data["executable"],
                "build": manifest.data.get("build", {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    _hash_file(digest, "bundle-tool", ROOT / "tools" / "stage_bundle.py")
    for name in ("build-deps.sh", "build.sh", "package-binary.sh"):
        _hash_file(digest, f"step/{name}", manifest.directory / "steps" / name)
    build_step = manifest.directory / "steps" / "build.sh"
    if "tools/build_cmake_target.sh" in build_step.read_text():
        _hash_file(
            digest,
            "tool/build-cmake-target",
            ROOT / "tools" / "build_cmake_target.sh",
        )
    for index, path in enumerate(patch_files):
        _hash_file(digest, f"patch/{index}", path)
    return digest.hexdigest()


def builder_image(args: argparse.Namespace) -> None:
    cli = container_cli(args.container_cli)
    image = args.image or (
        DEFAULT_LOCAL_SMOKE_IMAGE if args.target == "smoke" else DEFAULT_BUILDER_IMAGE
    )
    command: list[str | Path] = [
        cli,
        "build",
        "--file",
        ROOT / "docker" / "Dockerfile",
        "--target",
        args.target,
        "--tag",
        image,
    ]
    if args.target == "smoke":
        command.extend(["--build-arg", f"CLOUD_SLICER_IMAGE={args.cloud_slicer_image}"])
    command.append(ROOT)
    run(command)


def bundle_path(source: Path) -> Path:
    return source / "build" / "slicer_out"


def ensure_container_mountpoint(path: Path) -> None:
    """Create a directory that Docker must overlay below a read-only bind."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SystemExit(f"Container mountpoint is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def managed_node_cache(family_cache: Path, build_tree_hash: str) -> Path:
    """Keep source-selected Node tooling writable without sharing across variants."""
    return family_cache / "node" / build_tree_hash


def directory_tree_sha256(root: Path) -> str:
    """Hash bundle paths, modes, symlink targets, and regular-file contents."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit(f"Bundle root is not a directory: {root}")

    digest = hashlib.sha256()
    paths = sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            _hash_literal(digest, f"symlink/{relative}/{mode:o}", os.readlink(path))
        elif path.is_dir():
            _hash_literal(digest, f"directory/{relative}", f"{mode:o}")
        elif path.is_file():
            _hash_file(digest, f"file/{relative}", path)
        else:
            raise SystemExit(f"Unsupported bundle entry: {path}")
    return digest.hexdigest()


def validated_bundle_inventory(
    path: Path, arch: str, bundle_bytes: int, resource_bytes: int
) -> dict[str, Any]:
    try:
        report = load_json(path)
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Invalid bundle staging report: {path}") from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise SystemExit(f"Invalid bundle staging report schema: {path}")
    expected = {
        "architecture": arch,
        "bundle_bytes": bundle_bytes,
        "bundle_resource_bytes": resource_bytes,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise SystemExit(
                f"Bundle staging report {key} mismatch: expected {value!r}, "
                f"found {report.get(key)!r}"
            )
    library_count = report.get("library_count")
    resources = report.get("resources")
    if not isinstance(library_count, int) or library_count < 0:
        raise SystemExit("Bundle staging report has an invalid library count")
    if (
        not isinstance(resources, dict)
        or resources.get("staged_bytes") != resource_bytes
    ):
        raise SystemExit("Bundle staging report has invalid resource inventory")
    return {
        "schema_version": report["schema_version"],
        "architecture": report["architecture"],
        "library_count": library_count,
        "resources": resources,
    }


def recorded_build_bundle(
    manifest: Manifest,
    result: dict[str, Any],
    *,
    ref: str | None = None,
    arch: str | None = None,
) -> Path:
    """Resolve a recorded bundle only after verifying its complete tree digest."""
    expected_identity = {
        "slicer": manifest.name,
        **({"ref": ref} if ref is not None else {}),
        **({"arch": arch} if arch is not None else {}),
    }
    for key, expected in expected_identity.items():
        if result.get(key) != expected:
            raise SystemExit(
                f"Build result {key} mismatch: expected {expected!r}, "
                f"found {result.get(key)!r}"
            )
    bundle_value = result.get("bundle")
    if not isinstance(bundle_value, str) or not bundle_value:
        raise SystemExit("Build result does not record a bundle path")
    bundle = Path(bundle_value)
    executable = bundle / "bin" / manifest.data["executable"]
    if not executable.is_file():
        raise SystemExit(f"Recorded build bundle is incomplete: {bundle}")

    expected_digest = result.get("bundle_tree_sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise SystemExit("Build result has no valid bundle tree digest")
    actual_digest = directory_tree_sha256(bundle)
    if actual_digest != expected_digest:
        raise SystemExit(
            f"Recorded build bundle digest mismatch: {bundle} "
            f"(expected {expected_digest}, found {actual_digest})"
        )
    return bundle


def build_tree_stamp_path(source: Path) -> Path:
    return source / ".git" / "slicer-builds-build-tree.json"


def valid_build_tree_stamp(path: Path, fingerprint: str) -> bool:
    try:
        value = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("fingerprint") == fingerprint
    )


def prepare_managed_build_tree(source: Path, fingerprint: str) -> None:
    """Preserve a compatible partial build, or safely replace only source/build."""
    source = source.resolve(strict=True)
    git_directory = source / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise SystemExit(f"Refusing to manage build tree outside a clone: {source}")

    build = source / "build"
    if (
        build.is_symlink()
        or os.path.ismount(build)
        or (build.exists() and not build.is_dir())
    ):
        raise SystemExit(f"Refusing unsafe managed build path: {build}")

    stamp = build_tree_stamp_path(source)
    if not valid_build_tree_stamp(stamp, fingerprint):
        if build.exists():
            shutil.rmtree(build)
        build.mkdir()
        atomic_json(
            stamp,
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
            },
        )
    else:
        build.mkdir(exist_ok=True)


def build_result_directory(manifest: Manifest, ref: str, arch: str) -> Path:
    return WORK_ROOT / "results" / manifest.name / storage_name(ref) / safe_name(arch)


def build_result_path(manifest: Manifest, ref: str, arch: str) -> Path:
    return build_result_directory(manifest, ref, arch) / "build-result.json"


def optional_build_result(path: Path) -> dict[str, Any] | None:
    try:
        value = load_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def immutable_build_result_path(
    manifest: Manifest, ref: str, arch: str, commit: str, fingerprint: str
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise SystemExit(f"Invalid upstream commit for build result: {commit!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise SystemExit(f"Invalid fingerprint for build result: {fingerprint!r}")
    return (
        build_result_directory(manifest, ref, arch)
        / "records"
        / commit
        / f"{fingerprint}.json"
    )


def immutable_build_artifact_path(
    manifest: Manifest, ref: str, arch: str, commit: str, fingerprint: str
) -> Path:
    # Reuse the result-key validation before deriving a sibling artifact path.
    immutable_build_result_path(manifest, ref, arch, commit, fingerprint)
    return (
        build_result_directory(manifest, ref, arch)
        / "artifacts"
        / commit
        / fingerprint
        / "bundle"
    )


def quarantine_cache_path(path: Path, reason: str) -> Path | None:
    """Move one invalid managed-cache entry aside for recovery and inspection."""

    if not path.exists() and not path.is_symlink():
        return None
    work_root = WORK_ROOT.resolve()
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(work_root)
    except ValueError as error:
        raise SystemExit(
            f"Refusing to quarantine path outside work root: {path}"
        ) from error
    if not relative.parts or relative.parts[0] == "quarantine":
        raise SystemExit(f"Refusing unsafe cache quarantine path: {path}")
    try:
        lexical.parent.resolve(strict=True).relative_to(work_root)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Refusing unsafe cache quarantine path: {path}") from error

    quarantine_root = WORK_ROOT / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    if quarantine_root.is_symlink() or quarantine_root.resolve() != quarantine_root:
        raise SystemExit(f"Refusing symlinked cache quarantine root: {quarantine_root}")
    nonce = f"{time.time_ns()}-{os.getpid()}"
    digest = hashlib.sha256(f"{relative}:{nonce}".encode()).hexdigest()[:12]
    label = safe_name(relative.as_posix())[:80]
    destination = quarantine_root / f"{label}-{digest}"
    lexical.replace(destination)
    print(
        f"Quarantined invalid cache entry ({reason}): {path} -> {destination}",
        file=sys.stderr,
    )
    return destination


def validated_cached_build_result(
    manifest: Manifest,
    ref: str,
    arch: str,
    commit: str,
    fingerprint: str,
    result_path: Path,
    *,
    authoritative: bool,
) -> dict[str, Any] | None:
    """Load one cache record, quarantining corruption for a clean rebuild."""

    artifact = immutable_build_artifact_path(manifest, ref, arch, commit, fingerprint)
    if result_path.is_symlink() or artifact.is_symlink():
        quarantine_cache_path(result_path, "symlinked build-result cache entry")
        quarantine_cache_path(artifact, "symlinked build artifact")
        return None
    previous = optional_build_result(result_path)
    if previous is None:
        quarantine_cache_path(result_path, "invalid build-result JSON")
        if authoritative:
            quarantine_cache_path(artifact, "record metadata is invalid")
        return None
    if (
        previous.get("fingerprint") != fingerprint
        or previous.get("upstream_commit") != commit
    ):
        if authoritative:
            quarantine_cache_path(result_path, "immutable record identity mismatch")
            quarantine_cache_path(artifact, "immutable record identity mismatch")
        return None
    try:
        bundle = recorded_build_bundle(manifest, previous, ref=ref, arch=arch)
        if bundle.resolve(strict=True) != artifact.resolve(strict=True):
            raise SystemExit(
                f"Build result points outside its fingerprint artifact: {bundle}"
            )
    except (FileNotFoundError, SystemExit) as error:
        quarantine_cache_path(result_path, str(error))
        quarantine_cache_path(artifact, str(error))
        return None
    return previous


def publish_build_artifact(
    source_bundle: Path, destination: Path, expected_digest: str
) -> Path:
    """Move one staged bundle into its immutable fingerprint-owned directory."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise SystemExit("Cannot publish a bundle without a valid tree digest")
    if source_bundle.is_symlink():
        raise SystemExit(f"Refusing symlinked staged bundle: {source_bundle}")
    try:
        source = source_bundle.resolve(strict=True)
    except FileNotFoundError as error:
        raise SystemExit(f"Staged bundle does not exist: {source_bundle}") from error
    work_root = WORK_ROOT.resolve()
    try:
        source.relative_to(work_root)
        destination.absolute().relative_to(work_root)
    except ValueError as error:
        raise SystemExit(
            "Refusing to publish a bundle outside the work root"
        ) from error
    if not source.is_dir() or source == work_root:
        raise SystemExit(f"Refusing unsafe staged bundle path: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise SystemExit(f"Refusing symlinked artifact path: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise SystemExit(f"Build artifact path is not a directory: {destination}")
        actual_digest = directory_tree_sha256(destination)
        if actual_digest != expected_digest:
            raise SystemExit(
                "The same build identity produced different bundle contents: "
                f"{destination}"
            )
        if source != destination.resolve(strict=True):
            shutil.rmtree(source)
        return destination

    source.replace(destination)
    return destination


def publish_build_result(
    manifest: Manifest,
    ref: str,
    arch: str,
    commit: str,
    fingerprint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Write a content-addressed record before updating the legacy result pointer."""
    if payload.get("upstream_commit") != commit:
        raise SystemExit("Build result upstream commit does not match its record key")
    if payload.get("fingerprint") != fingerprint:
        raise SystemExit("Build result fingerprint does not match its record key")

    pointer = build_result_path(manifest, ref, arch)
    immutable = immutable_build_result_path(manifest, ref, arch, commit, fingerprint)
    published = dict(payload)
    published["result"] = str(pointer)
    published["immutable_result"] = str(immutable)

    if immutable.is_file():
        try:
            recorded = load_json(immutable)
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"Invalid immutable build result: {immutable}") from error
        if not isinstance(recorded, dict) or (
            recorded.get("upstream_commit") != commit
            or recorded.get("fingerprint") != fingerprint
        ):
            raise SystemExit(
                f"Immutable build result has the wrong identity: {immutable}"
            )
        bundle_digest = published.get("bundle_tree_sha256")
        if (
            bundle_digest is not None
            and recorded.get("bundle_tree_sha256") != bundle_digest
        ):
            raise SystemExit(
                "The same build identity produced different bundle contents: "
                f"{immutable}"
            )
    else:
        atomic_json(immutable, published)
    atomic_json(pointer, published)
    return published


def valid_dependency_stamp(path: Path, fingerprint: str, arch: str) -> bool:
    try:
        value = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("fingerprint") == fingerprint
        and value.get("arch") == arch
    )


def validate_dependency_skip(
    skip_dependencies: bool, cache_hit: bool, marker: str | None
) -> None:
    if skip_dependencies and not cache_hit:
        raise SystemExit(
            "--skip-deps requires a complete matching dependency cache"
            + (f" and checkout marker {marker}" if marker is not None else "")
        )


def build_ref_lock_key(manifest: Manifest, ref: str, arch: str) -> str:
    return f"ref-build:{manifest.name}:{arch}:{ref}"


def build_one(args: argparse.Namespace) -> dict[str, Any]:
    manifest = get_manifest(args.slicer)
    ref = args.ref or manifest.ref
    arch = args.arch
    if arch not in manifest.data["architectures"]:
        raise SystemExit(f"{manifest.name} does not declare support for {arch}")
    native_arch = native_architecture()
    if arch != native_arch:
        raise SystemExit(
            f"{manifest.name} {arch} must be built on a native {arch} runner; "
            f"this host is {native_arch}"
        )
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit("--jobs must be a positive integer")

    # A moving literal ref has one publication order per architecture. This
    # outermost lock is intentionally acquired before source resolution and is
    # never acquired by any inner cache/source lock.
    with file_lock(build_ref_lock_key(manifest, ref, arch)):
        return _build_one_locked(args, manifest, ref, arch)


def _build_one_locked(
    args: argparse.Namespace, manifest: Manifest, ref: str, arch: str
) -> dict[str, Any]:
    cli = container_cli(args.container_cli)
    image_identity = container_image_identity(cli, args.image)
    build_env = effective_build_environment(manifest)
    if (
        args.patches == "dump"
        and manifest.data["family"] == "prusa"
        and "SLICER_GUI" not in os.environ
    ):
        build_env["SLICER_GUI"] = "1"
    patches = legacy_patch_files(manifest, ref, args.patches)
    seed = args.source.resolve() if args.source else None
    source, commit = prepare_source(
        manifest,
        ref,
        seed,
        variant=(
            f"{patch_variant(args.patches, patches)}-{arch}-"
            f"{build_configuration_variant(image_identity, build_env)}"
        ),
        expected_commit=getattr(args, "expected_commit", None),
    )
    if manifest.name in VERSION_DATE_STAMP_SLICERS:
        build_date = build_env.get("SLICER_BUILD_DATE", "")
        if not build_date:
            build_date = run(
                ["git", "show", "-s", "--format=%cs", commit],
                cwd=source,
                capture=True,
            )
            build_env["SLICER_BUILD_DATE"] = build_date
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", build_date):
            raise SystemExit("SLICER_BUILD_DATE must use YYYY-MM-DD")

    with file_lock(f"source-build:{source}"):
        apply_legacy_patches(source, patches)
        source_state_hash = source_state(source, patches)
        deps_fingerprint = dependency_fingerprint(
            manifest, source, arch, image_identity, build_env
        )
        build_tree_hash = build_tree_fingerprint(
            manifest,
            commit,
            arch,
            image_identity,
            deps_fingerprint,
            source_state_hash,
            build_env,
        )
        fingerprint = build_fingerprint(
            manifest,
            commit,
            patches,
            arch,
            image_identity,
            deps_fingerprint,
            build_tree_hash,
            source_state_hash,
            build_env,
        )

        result_pointer = build_result_path(manifest, ref, arch)
        immutable_result = immutable_build_result_path(
            manifest, ref, arch, commit, fingerprint
        )
        if not args.skip_package:
            expected_artifact = immutable_build_artifact_path(
                manifest, ref, arch, commit, fingerprint
            )
            if (
                not immutable_result.exists()
                and not immutable_result.is_symlink()
                and (expected_artifact.exists() or expected_artifact.is_symlink())
            ):
                quarantine_cache_path(
                    expected_artifact, "orphaned fingerprint artifact"
                )
            for cached_result in (immutable_result, result_pointer):
                if not cached_result.exists() and not cached_result.is_symlink():
                    continue
                previous = validated_cached_build_result(
                    manifest,
                    ref,
                    arch,
                    commit,
                    fingerprint,
                    cached_result,
                    authoritative=cached_result == immutable_result,
                )
                if previous is not None and not args.force:
                    previous = publish_build_result(
                        manifest, ref, arch, commit, fingerprint, previous
                    )
                    previous["reused"] = True
                    print(json.dumps(previous, indent=2, sort_keys=True))
                    return previous

        prepare_managed_build_tree(source, build_tree_hash)

        jobs = args.jobs if args.jobs is not None else default_build_jobs()
        family_cache = WORK_ROOT / "cache" / manifest.data["family"]
        node_cache = managed_node_cache(family_cache, build_tree_hash)
        home = WORK_ROOT / "home" / manifest.name
        for directory in (
            family_cache / "ccache",
            family_cache / "conan",
            family_cache / "downloads",
            node_cache,
            family_cache / "sccache",
            home,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        deps_cache = WORK_ROOT / "deps" / arch / deps_fingerprint
        deps_stamp = deps_cache / ".slicer-builds-complete.json"
        deps_cache.mkdir(parents=True, exist_ok=True)
        (source / "deps").mkdir(exist_ok=True)
        (source / "deps" / "DL_CACHE").mkdir(exist_ok=True)
        (source / "deps" / "build").mkdir(exist_ok=True)

        # Docker cannot create a nested bind target below ROOT after ROOT has
        # already been mounted read-only. A clean Git checkout intentionally
        # has no slicer-src directory (it is ignored), so provision the empty
        # overlay target before assembling the container command.
        ensure_container_mountpoint(ROOT / "slicer-src")
        ensure_container_mountpoint(ROOT / "node-cache")

        base_command: list[str | Path] = [
            cli,
            "run",
            "--rm",
            "--init",
            *container_run_identity_options(cli),
            "--workdir",
            "/workspace/repo",
            "--volume",
            f"{ROOT}:/workspace/repo:ro",
            "--volume",
            f"{source}:/workspace/repo/slicer-src:rw",
            "--volume",
            f"{node_cache}:/workspace/repo/node-cache:rw",
            "--volume",
            f"{deps_cache}:/workspace/repo/slicer-src/deps/build:rw",
            "--volume",
            f"{family_cache / 'downloads'}:/workspace/repo/slicer-src/deps/DL_CACHE:rw",
            "--volume",
            f"{home}:/workspace/home:rw",
            "--volume",
            f"{family_cache / 'ccache'}:/cache/ccache:rw",
            "--volume",
            f"{family_cache / 'conan'}:/cache/conan:rw",
            "--volume",
            f"{family_cache / 'sccache'}:/cache/sccache:rw",
            "--env",
            "HOME=/workspace/home",
            "--env",
            f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
            "--env",
            f"MAKEFLAGS=-j{jobs}",
            "--env",
            "CMAKE_C_COMPILER_LAUNCHER=sccache",
            "--env",
            "CMAKE_CXX_COMPILER_LAUNCHER=sccache",
            "--env",
            "CMAKE_CCACHE=sccache",
            "--env",
            "CCACHE_BASEDIR=/workspace/repo/slicer-src",
            "--env",
            "CONAN=/opt/conan/bin/conan",
            "--env",
            f"SLICER_JOBS={jobs}",
            "--env",
            f"ARCH={arch}",
            "--env",
            f"SLICER_BUILD_FINGERPRINT={fingerprint}",
        ]
        for key in FORWARDED_BUILD_ENV:
            if value := build_env.get(key):
                base_command.extend(["--env", f"{key}={value}"])

        phase_seconds: dict[str, float] = {}

        def run_phase(phase: str) -> None:
            started_at = time.monotonic()
            try:
                run(
                    [
                        *base_command,
                        args.image,
                        "bash",
                        f"./slicers/{manifest.name}/steps/{phase}.sh",
                    ]
                )
            finally:
                restore_managed_worktree(source)
                phase_seconds[phase] = round(time.monotonic() - started_at, 6)
            if source_state(source, patches) != source_state_hash:
                raise SystemExit(f"Build phase {phase} modified the managed Git index")

        with file_lock(f"dependency-build:{deps_cache}"):
            marker_value = manifest.data.get("build", {}).get("dependency_marker")
            checkout_dependency_ready = (
                marker_value is None or (source / marker_value).is_file()
            )
            deps_cache_hit = (
                valid_dependency_stamp(deps_stamp, deps_fingerprint, arch)
                and checkout_dependency_ready
            )
            validate_dependency_skip(args.skip_deps, deps_cache_hit, marker_value)
            if not deps_cache_hit:
                # A complete stamp is valid only together with the checkout's
                # dependency marker. Invalidate it before rebuilding so a
                # failed phase that happens to create the marker cannot make
                # the next invocation accept stale shared dependencies.
                deps_stamp.unlink(missing_ok=True)
                # Different dependency fingerprints still share download and
                # Conan metadata. Serialize that mutable family state while
                # leaving compiler caches available to concurrent builds.
                with file_lock(
                    f"family-dependency-build:{manifest.data['family']}:{arch}"
                ):
                    run_phase("build-deps")
                if marker_value is not None and not (source / marker_value).is_file():
                    raise SystemExit(
                        f"Dependency build did not create checkout marker {marker_value}"
                    )
                atomic_json(
                    deps_stamp,
                    {
                        "schema_version": 1,
                        "fingerprint": deps_fingerprint,
                        "family": manifest.data["family"],
                        "arch": arch,
                    },
                )
            elif deps_cache_hit:
                print(
                    f"Reusing dependency build {deps_fingerprint[:12]}",
                    file=sys.stderr,
                )

        run_phase("build")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "slicer": manifest.name,
            "ref": ref,
            "upstream_commit": commit,
            "arch": arch,
            "builder_image": args.image,
            "builder_image_identity": image_identity,
            "fingerprint": fingerprint,
            "dependency_fingerprint": deps_fingerprint,
            "dependency_cache_hit": deps_cache_hit,
            "build_tree_fingerprint": build_tree_hash,
            "source_state": source_state_hash,
            "patches": [str(path.relative_to(ROOT)) for path in patches],
            "source": str(source),
            "packaged": not args.skip_package,
            "phase_seconds": phase_seconds,
            "reused": False,
        }
        if args.skip_package:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return payload

        run_phase("package-binary")
        bundle = bundle_path(source)
        executable = bundle / "bin" / manifest.data["executable"]
        if not executable.is_file():
            raise SystemExit(f"Build did not stage {executable}")
        bundle_bin_bytes = sum(
            path.stat().st_size
            for path in (bundle / "bin").rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        bundle_bytes = sum(
            path.stat().st_size
            for path in bundle.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        bundle_resource_bytes = (
            sum(
                path.stat().st_size
                for path in (bundle / "resources").rglob("*")
                if path.is_file() and not path.is_symlink()
            )
            if (bundle / "resources").is_dir()
            else 0
        )
        bundle_inventory = validated_bundle_inventory(
            source / "build" / "slicer-bundle-report.json",
            arch,
            bundle_bytes,
            bundle_resource_bytes,
        )
        bundle_digest = directory_tree_sha256(bundle)
        artifact_bundle = publish_build_artifact(
            bundle,
            immutable_build_artifact_path(manifest, ref, arch, commit, fingerprint),
            bundle_digest,
        )
        artifact_executable = artifact_bundle / "bin" / manifest.data["executable"]
        payload.update(
            {
                "bundle": str(artifact_bundle),
                "executable_bytes": artifact_executable.stat().st_size,
                "bundle_bin_bytes": bundle_bin_bytes,
                "bundle_bytes": bundle_bytes,
                "bundle_resource_bytes": bundle_resource_bytes,
                "bundle_inventory": bundle_inventory,
                "bundle_tree_sha256": bundle_digest,
                "executable": str(artifact_executable),
            }
        )
        payload = publish_build_result(
            manifest, ref, arch, commit, fingerprint, payload
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload


def latest_build_bundle(manifest: Manifest, ref: str, arch: str) -> Path:
    result = build_result_path(manifest, ref, arch)
    if not result.is_file():
        raise SystemExit(
            f"No build result for {manifest.name} {ref} {arch}; "
            "pass --bundle or build it first"
        )
    value = load_json(result)
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid build result: {result}")
    return recorded_build_bundle(manifest, value, ref=ref, arch=arch)


def git_repository_identity(path: Path) -> dict[str, Any] | None:
    """Return stable source identity without including dirty file names."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD^{commit}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def default_test_input(manifest: Manifest, backend_root: Path) -> Path:
    configured = manifest.data["test"].get("input_fixture")
    if configured is not None:
        candidate = (ROOT / configured).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            raise SystemExit(f"Unsafe test input fixture: {configured!r}") from None
        return candidate
    if manifest.data["test"]["contract"] == "bambu":
        return DEFAULT_BAMBU_INPUT
    return backend_root / "tests/e2e/fixtures/calicat.stl"


def backend_version(ref: str) -> str:
    if ref == "HEAD":
        return "nightly"
    normalized = ref.removeprefix("version_").removeprefix("v")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9_.-]+)?", normalized):
        return normalized
    return "nightly"


def test_one(args: argparse.Namespace) -> dict[str, Any]:
    manifest = get_manifest(args.slicer)
    ref = args.ref or manifest.ref
    arch = args.arch
    if arch not in manifest.data["architectures"]:
        raise SystemExit(f"{manifest.name} does not declare support for {arch}")
    if args.timeout < 1:
        raise SystemExit("--timeout must be a positive integer")
    build_result: dict[str, Any] | None = None
    if args.bundle:
        bundle = args.bundle.resolve()
    else:
        result_path = build_result_path(manifest, ref, arch)
        if not result_path.is_file():
            raise SystemExit(
                f"No build result for {manifest.name} {ref} {arch}; "
                "pass --bundle or build it first"
            )
        loaded_result = load_json(result_path)
        if not isinstance(loaded_result, dict):
            raise SystemExit(f"Invalid build result: {result_path}")
        build_result = loaded_result
        bundle = recorded_build_bundle(manifest, build_result, ref=ref, arch=arch)
    backend_root = args.backend_root.resolve()
    profiles_root = args.profiles_root.resolve()
    input_path = (
        args.input.resolve()
        if args.input
        else default_test_input(manifest, backend_root)
    )
    required_paths: list[Path] = [
        bundle / "bin" / manifest.data["executable"],
        backend_root,
        input_path,
    ]
    profile_source = manifest.data["test"]["profile_source"]
    needs_profiles_mount = profile_source in {"profiles-db", "profiles-direct"}
    if needs_profiles_mount:
        required_paths.append(profiles_root)
    for required in required_paths:
        if not required.exists():
            raise SystemExit(f"Required smoke-test input does not exist: {required}")

    work = WORK_ROOT / "tests" / manifest.name / storage_name(ref) / safe_name(arch)
    with file_lock(f"smoke-test:{work}"):
        if work.exists():
            shutil.rmtree(work)
        (work / "home").mkdir(parents=True)
        spec = dict(manifest.data)
        declared_ref = manifest_ref_spec(manifest, ref)
        spec["backend_version"] = (
            args.backend_version
            or (declared_ref or {}).get("backend_version")
            or manifest.data["test"].get("backend_version")
            or backend_version(ref)
        )
        spec["runtime_image"] = args.image
        if build_result is not None:
            provenance_keys = (
                "ref",
                "upstream_commit",
                "arch",
                "builder_image_identity",
                "fingerprint",
                "dependency_fingerprint",
                "build_tree_fingerprint",
                "source_state",
                "patches",
                "bundle_tree_sha256",
            )
            spec["build_provenance"] = {
                key: build_result[key] for key in provenance_keys if key in build_result
            }
        spec["backend_source_identity"] = git_repository_identity(backend_root)
        if needs_profiles_mount:
            spec["profile_source_identity"] = git_repository_identity(profiles_root)
        cli = container_cli(args.container_cli)
        spec["runtime_image_identity"] = container_image_identity(cli, args.image)
        atomic_json(work / "spec.json", spec)

        name_seed = f"{manifest.name}-{ref}-{os.getpid()}"
        container_name = f"slicer-smoke-{bounded_storage_name(name_seed, 80).lower()}"
        command: list[str | Path] = [
            cli,
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            *container_run_identity_options(cli),
            "--env",
            "HOME=/test/work/home",
            "--volume",
            f"{bundle}:/test/bundle:ro",
            "--volume",
            f"{backend_root}:/test/backend:ro",
            "--volume",
            f"{work}:/test/work:rw",
            "--volume",
            f"{ROOT / 'tests/integration/backend_smoke.py'}:/test/backend_smoke.py:ro",
            "--volume",
            f"{input_path}:/test/input/{input_path.name}:ro",
        ]
        if needs_profiles_mount:
            command.extend(["--volume", f"{profiles_root}:/test/profiles:ro"])
        command.extend(
            [
                args.image,
                "python3",
                "/test/backend_smoke.py",
                "--spec",
                "/test/work/spec.json",
                "--backend-root",
                "/test/backend",
                "--profiles-root",
                "/test/profiles" if needs_profiles_mount else "/test/unused-profiles",
                "--bundle",
                "/test/bundle",
                "--input",
                f"/test/input/{input_path.name}",
                "--work",
                "/test/work/output",
            ]
        )
        try:
            run(command, timeout_seconds=args.timeout)
        except subprocess.TimeoutExpired as error:
            subprocess.run(
                [cli, "rm", "--force", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise SystemExit(
                f"Smoke test exceeded {args.timeout} seconds: {manifest.name} {ref}"
            ) from error
        report_path = work / "output" / "smoke-result.json"
        if not report_path.is_file():
            raise SystemExit(f"Smoke test did not write {report_path}")
        report = load_json(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report


def command_list(args: argparse.Namespace) -> None:
    values = [
        {
            "name": manifest.name,
            "family": manifest.data["family"],
            "ref": manifest.ref,
            "supported_refs": [spec["ref"] for spec in manifest_ref_specs(manifest)],
            "architectures": manifest.data["architectures"],
            "backend_supported": manifest.data["backend_supported"],
            "capabilities": manifest.data["capabilities"],
        }
        for manifest in manifests().values()
    ]
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for value in values:
            print(f"{value['name']:<24} {value['family']:<8} {value['ref']}")


def command_matrix(args: argparse.Namespace) -> None:
    include = []
    for manifest in manifests().values():
        if not manifest.data["capabilities"].get(args.capability, False):
            continue
        refs = (
            manifest_ref_specs(manifest)
            if args.all_refs
            else manifest_ref_specs(manifest)[:1]
        )
        for ref_spec in refs:
            for arch in manifest.data["architectures"]:
                include.append(
                    {
                        "slicer": manifest.name,
                        "arch": arch,
                        "ref": ref_spec["ref"],
                        "expected_commit": ref_spec["expected_commit"],
                        "repository": manifest.data["repository"],
                    }
                )
    print(json.dumps({"include": include}, separators=(",", ":")))


def command_patch_files(args: argparse.Namespace) -> None:
    manifest = get_manifest(args.slicer)
    paths = legacy_patch_files(manifest, args.ref, args.mode)
    rendered = [str(path.relative_to(ROOT)) for path in paths]
    if args.null:
        for path in rendered:
            sys.stdout.buffer.write(os.fsencode(path) + b"\0")
    else:
        print("\n".join(rendered))


def _display_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _render_patch_verification(payload: dict[str, Any]) -> None:
    for result in payload["results"]:
        label = f"{result['slicer']}@{result['ref']} [{result['mode']}]"
        if result["status"] == "passed":
            print(f"PASS {label}: {result['patch_count']} patches")
            continue
        detail = f"{result['stage']}: {result['error']}"
        if result.get("patch"):
            detail = f"{result['patch']}: {detail}"
        print(f"FAIL {label}: {detail}")

    summary = payload["summary"]
    print(
        "Patch verification: "
        f"{summary['passed_stacks']}/{summary['planned_stacks']} stacks passed; "
        f"{summary['failed_stacks']} failed; "
        f"{summary['patches_selected']} patch selections"
    )
    if payload["stopped_early"]:
        print("Stopped after the first failure (--fail-fast).")


def command_verify_patches(args: argparse.Namespace) -> None:
    available = manifests()
    requested_names = list(dict.fromkeys(args.slicers or available))
    unknown_names = sorted(set(requested_names) - set(available))
    if unknown_names:
        raise SystemExit(f"Unknown slicers: {', '.join(unknown_names)}")
    if args.source is not None and len(requested_names) != 1:
        raise SystemExit("--source requires exactly one --slicer")

    requested_refs = list(dict.fromkeys(args.refs or []))
    requested_modes = list(dict.fromkeys(args.modes or ("binary", "dump")))
    capability_for_mode = {"binary": "binary", "dump": "config_dump"}
    matched_refs: set[str] = set()
    matched_modes: set[str] = set()
    plans: list[
        tuple[
            Manifest,
            list[tuple[dict[str, str | None], list[tuple[str, list[Path]]]]],
        ]
    ] = []

    for name in requested_names:
        manifest = available[name]
        ref_specs = [
            spec
            for spec in patch_verification_ref_specs(manifest, args.include_head)
            if not requested_refs or spec["ref"] in requested_refs
        ]
        matched_refs.update(str(spec["ref"]) for spec in ref_specs)
        modes = [
            mode
            for mode in requested_modes
            if manifest.data["capabilities"].get(capability_for_mode[mode], False)
        ]
        matched_modes.update(modes)
        ref_plans = [
            (
                spec,
                [
                    (
                        mode,
                        legacy_patch_files(manifest, str(spec["ref"]), mode),
                    )
                    for mode in modes
                ],
            )
            for spec in ref_specs
        ]
        if ref_plans and modes:
            plans.append((manifest, ref_plans))

    missing_refs = sorted(set(requested_refs) - matched_refs)
    if missing_refs:
        raise SystemExit(
            "Refs are not declared by the selected slicers: " + ", ".join(missing_refs)
        )
    missing_modes = sorted(set(args.modes or ()) - matched_modes)
    if missing_modes:
        raise SystemExit(
            "Patch modes are not enabled by the selected slicers: "
            + ", ".join(missing_modes)
        )

    planned_stacks = sum(
        len(mode_plans)
        for _manifest, ref_plans in plans
        for _spec, mode_plans in ref_plans
    )
    planned_patches = sum(
        len(patch_files)
        for _manifest, ref_plans in plans
        for _spec, mode_plans in ref_plans
        for _mode, patch_files in mode_plans
    )
    results: list[dict[str, Any]] = []
    stopped_early = False

    def append_result(
        manifest: Manifest,
        spec: dict[str, str | None],
        mode: str,
        patch_files: Sequence[Path],
        *,
        commit: str | None,
        failure: dict[str, str] | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "slicer": manifest.name,
            "ref": spec["ref"],
            "expected_commit": spec.get("expected_commit"),
            "commit": commit,
            "mode": mode,
            "status": "failed" if failure else "passed",
            "patch_count": len(patch_files),
            "patches": [_display_repo_path(path) for path in patch_files],
        }
        if failure:
            result.update(failure)
            if failure.get("patch"):
                result["patch"] = _display_repo_path(Path(failure["patch"]))
        results.append(result)

    for manifest, ref_plans in plans:
        try:
            mirror = prepare_patch_verification_mirror(manifest, args.source)
        except (OSError, subprocess.CalledProcessError, SystemExit) as error:
            failure = {
                "stage": "prepare-mirror",
                "error": failure_message(error),
            }
            for spec, mode_plans in ref_plans:
                for mode, patch_files in mode_plans:
                    append_result(
                        manifest,
                        spec,
                        mode,
                        patch_files,
                        commit=None,
                        failure=failure,
                    )
                    if args.fail_fast:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            if stopped_early:
                break
            continue

        for spec, mode_plans in ref_plans:
            original_ref = str(spec["ref"])
            commit, resolve_error = resolve_patch_verification_ref(mirror, original_ref)
            ref_failure: dict[str, str] | None = None
            if commit is None:
                ref_failure = {"stage": "resolve-ref", "error": resolve_error}
            elif expected_commit := spec.get("expected_commit"):
                if commit != expected_commit:
                    ref_failure = {
                        "stage": "expected-commit",
                        "error": (
                            f"resolved to {commit}, expected locked commit "
                            f"{expected_commit}"
                        ),
                    }

            if ref_failure:
                for mode, patch_files in mode_plans:
                    append_result(
                        manifest,
                        spec,
                        mode,
                        patch_files,
                        commit=commit,
                        failure=ref_failure,
                    )
                    if args.fail_fast:
                        stopped_early = True
                        break
                if stopped_early:
                    break
                continue

            assert commit is not None
            for mode, patch_files in mode_plans:
                failure = verify_patch_stack(mirror, commit, patch_files)
                append_result(
                    manifest,
                    spec,
                    mode,
                    patch_files,
                    commit=commit,
                    failure=failure,
                )
                if failure and args.fail_fast:
                    stopped_early = True
                    break
            if stopped_early:
                break
        if stopped_early:
            break

    passed_stacks = sum(result["status"] == "passed" for result in results)
    failed_stacks = sum(result["status"] == "failed" for result in results)
    payload = {
        "schema_version": 1,
        "ok": failed_stacks == 0 and len(results) == planned_stacks,
        "stopped_early": stopped_early,
        "summary": {
            "slicers": len(plans),
            "refs": sum(len(ref_plans) for _manifest, ref_plans in plans),
            "planned_stacks": planned_stacks,
            "attempted_stacks": len(results),
            "passed_stacks": passed_stacks,
            "failed_stacks": failed_stacks,
            "patches_selected": planned_patches,
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_patch_verification(payload)
    if not payload["ok"]:
        raise SystemExit(1)


def selected_manifest_names(requested: list[str] | None, arch: str) -> list[str]:
    available = manifests()
    if requested:
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise SystemExit(f"Unknown slicers: {', '.join(unknown)}")
        names = list(dict.fromkeys(requested))
        unsupported = [
            name for name in names if arch not in available[name].data["architectures"]
        ]
        if unsupported:
            raise SystemExit(f"Slicers do not support {arch}: {', '.join(unsupported)}")
    else:
        names = list(available)
    return [name for name in names if arch in available[name].data["architectures"]]


def selected_manifest_targets(
    requested: list[str] | None, arch: str, all_refs: bool
) -> list[tuple[str, str]]:
    available = manifests()
    names = selected_manifest_names(requested, arch)
    return [
        (name, str(ref_spec["ref"]))
        for name in names
        for ref_spec in (
            manifest_ref_specs(available[name])
            if all_refs
            else manifest_ref_specs(available[name])[:1]
        )
    ]


def failure_message(error: BaseException) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        return f"command exited {error.returncode}: {' '.join(map(str, error.cmd))}"
    if isinstance(error, SystemExit):
        return str(error.code)
    return str(error)


def run_many(
    names: Sequence[str],
    workers: int,
    fail_fast: bool,
    action: Callable[[str], Any],
) -> tuple[list[str], dict[str, str]]:
    if workers < 1:
        raise SystemExit("--workers must be a positive integer")
    outcomes: dict[str, str | None] = {}

    def invoke(name: str) -> str | None:
        try:
            action(name)
            return None
        except (subprocess.CalledProcessError, SystemExit) as error:
            return failure_message(error)

    # Fail-fast has deterministic semantics only before another task starts.
    # Keep it serial; ordinary runs may use bounded slicer-level concurrency.
    if workers == 1 or fail_fast:
        for name in names:
            outcomes[name] = invoke(name)
            if outcomes[name] is not None:
                print(f"{name}: {outcomes[name]}", file=sys.stderr)
            if fail_fast and outcomes[name] is not None:
                break
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(names)))
        futures = {executor.submit(invoke, name): name for name in names}
        try:
            for future in as_completed(futures):
                name = futures[future]
                outcomes[name] = future.result()
                if outcomes[name] is not None:
                    print(f"{name}: {outcomes[name]}", file=sys.stderr)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    successful = [
        name for name in names if outcomes.get(name) is None and name in outcomes
    ]
    failed = {
        name: outcome for name in names if (outcome := outcomes.get(name)) is not None
    }
    return successful, failed


def run_many_collect(
    names: Sequence[str],
    workers: int,
    fail_fast: bool,
    action: Callable[[str], Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run named work while retaining successful return values in input order."""

    if workers < 1:
        raise SystemExit("--workers must be a positive integer")
    values: dict[str, Any] = {}
    failures: dict[str, str] = {}

    def invoke(name: str) -> tuple[Any | None, str | None]:
        try:
            return action(name), None
        except SystemExit as error:
            return None, failure_message(error)
        except Exception as error:
            return None, failure_message(error)

    if workers == 1 or fail_fast:
        for name in names:
            value, error = invoke(name)
            if error is None:
                values[name] = value
                continue
            failures[name] = error
            print(f"{name}: {error}", file=sys.stderr)
            if fail_fast:
                break
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(names)))
        futures = {executor.submit(invoke, name): name for name in names}
        try:
            for future in as_completed(futures):
                name = futures[future]
                value, error = future.result()
                if error is None:
                    values[name] = value
                else:
                    failures[name] = error
                    print(f"{name}: {error}", file=sys.stderr)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    ordered_values = {name: values[name] for name in names if name in values}
    ordered_failures = {name: failures[name] for name in names if name in failures}
    return ordered_values, ordered_failures


def parse_minimum_versions(
    values: Sequence[str] | None, selected_names: Sequence[str]
) -> dict[str, str]:
    """Parse repeatable ``SLICER=VERSION`` thresholds, with one-slicer shorthand."""

    minimums: dict[str, str] = {}
    selected = set(selected_names)
    for value in values or ():
        name, separator, version = value.partition("=")
        if separator != "=":
            if len(selected_names) != 1:
                raise SystemExit(
                    "--minimum-version requires SLICER=VERSION when multiple "
                    "slicers are selected"
                )
            name, version = selected_names[0], value
        if name not in selected:
            raise SystemExit(f"Minimum version names an unselected slicer: {name}")
        if name in minimums:
            raise SystemExit(f"Duplicate minimum version for {name}")
        if release_version_parts(version) is None:
            raise SystemExit(f"Invalid stable minimum version for {name}: {version!r}")
        minimums[name] = clean_release_version(version)
    return minimums


def _release_target_payload(target: ReleaseTarget) -> dict[str, Any]:
    return {
        "slicer": target.slicer,
        "ref": target.ref,
        "version": target.version,
        "version_group": target.version_group,
        "upstream_commit": target.expected_commit,
    }


def compact_build_output(
    target: ReleaseTarget, payload: dict[str, Any]
) -> dict[str, Any]:
    """Return the retained artifact and exact size fields for one build."""

    executable_value = payload.get("executable")
    if not isinstance(executable_value, str) or not executable_value:
        raise SystemExit(f"Build result has no executable for {target.slicer}@{target.ref}")
    executable = Path(executable_value)
    if not executable.is_file():
        raise SystemExit(f"Build executable is missing: {executable}")
    bundle_value = payload.get("bundle")
    if not isinstance(bundle_value, str) or not bundle_value:
        raise SystemExit(f"Build result has no bundle for {target.slicer}@{target.ref}")
    bundle = Path(bundle_value)
    if not bundle.is_dir():
        raise SystemExit(f"Build bundle is missing: {bundle}")

    def tree_bytes(root: Path) -> int:
        if not root.is_dir():
            return 0
        return sum(
            path.stat().st_size
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    # Recompute at report time. This keeps legacy cached records useful and
    # makes the table describe the actual retained bytes rather than trusting
    # optional or stale aggregate metadata.
    size_fields = {
        "bundle_bin_bytes": tree_bytes(bundle / "bin"),
        "bundle_resource_bytes": tree_bytes(bundle / "resources"),
        "bundle_bytes": tree_bytes(bundle),
    }

    return {
        **_release_target_payload(target),
        "status": "reused" if payload.get("reused") else "built",
        "executable": str(executable),
        # Older immutable results predate this field; statting the retained
        # executable upgrades reports without invalidating a costly build.
        "executable_bytes": executable.stat().st_size,
        **size_fields,
        "bundle": str(bundle),
        "bundle_tree_sha256": payload.get("bundle_tree_sha256"),
        "result": payload.get("result"),
        "immutable_result": payload.get("immutable_result"),
        "fingerprint": payload.get("fingerprint"),
        "phase_seconds": payload.get("phase_seconds", {}),
    }


def format_byte_count(value: Any) -> str:
    if not isinstance(value, int) or value < 0:
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if amount < 1024 or candidate == units[-1]:
            break
        amount /= 1024
    return f"{amount:.1f} {unit}" if unit != "B" else f"{value} B"


def render_history_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Historical slicer build sizes",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "| Slicer | Version | Status | Executable | bin/ closure | Resources | Bundle | Artifact |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for build in report["builds"]:
        artifact = str(build.get("bundle") or "—").replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(build["slicer"]),
                    str(build["ref"]),
                    str(build["status"]),
                    format_byte_count(build.get("executable_bytes")),
                    format_byte_count(build.get("bundle_bin_bytes")),
                    format_byte_count(build.get("bundle_resource_bytes")),
                    format_byte_count(build.get("bundle_bytes")),
                    f"`{artifact}`" if artifact != "—" else artifact,
                )
            )
            + " |"
        )
    skipped = report["summary"].get("skipped_slicers", {})
    if skipped:
        lines.extend(("", "## Skipped slicers", ""))
        for name, reason in skipped.items():
            lines.append(f"- **{name}**: {reason}")
    build_issues = [
        build for build in report["builds"] if build["status"] in {"failed", "not-run"}
    ]
    if build_issues:
        lines.extend(("", "## Build issues", ""))
        for build in build_issues:
            lines.append(
                f"- **{build['slicer']}@{build['ref']}**: "
                f"{build.get('error', build['status'])}"
            )
    if report["discovery_failures"]:
        lines.extend(("", "## Discovery failures", ""))
        for name, error in report["discovery_failures"].items():
            lines.append(f"- **{name}**: {error}")
    return "\n".join(lines) + "\n"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def command_backfill(args: argparse.Namespace) -> None:
    """Discover, pin, build, and size-report a bounded stable release history."""

    available = manifests()
    names = selected_manifest_names(args.slicers, args.arch)
    report_path = args.report or (
        WORK_ROOT / "reports" / f"historical-builds-{safe_name(args.arch)}.json"
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if report_path.suffix.lower() != ".json":
        raise SystemExit("--report must name a .json file")
    markdown_path = report_path.with_suffix(".md")
    if report_path.exists() and not report_path.is_file():
        raise SystemExit(f"--report is not a file path: {report_path}")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, probe_name = tempfile.mkstemp(
            prefix=".slicerctl-report-probe.", dir=report_path.parent
        )
        os.close(descriptor)
        Path(probe_name).unlink()
    except OSError as error:
        raise SystemExit(
            f"Cannot write historical build reports under {report_path.parent}: {error}"
        ) from error
    minimums = parse_minimum_versions(args.minimum_versions, names)
    maximum = None if args.all_versions else args.max_versions
    workers = args.workers

    def discover(name: str) -> list[ReleaseTarget]:
        return historical_release_targets(
            available[name],
            maximum=maximum,
            minimum=minimums.get(name),
            source=args.versions_from,
        )

    discovered, discovery_failures = run_many_collect(
        names, workers, args.fail_fast, discover
    )
    targets = [target for name in names for target in discovered.get(name, [])]
    skipped: dict[str, str] = {}
    for name in names:
        if name in discovery_failures:
            continue
        if name not in discovered:
            skipped[name] = "not attempted after an earlier fail-fast failure"
        elif not discovered[name]:
            index_path = available[name].directory / "out" / "_index.json"
            skipped[name] = (
                f"no release index at {index_path}"
                if args.versions_from == "index" and not index_path.is_file()
                else "no stable releases matched the requested policy"
            )
    labels = [f"{target.slicer}@{target.ref}" for target in targets]
    if len(labels) != len(set(labels)):
        raise SystemExit("Historical release planning produced duplicate build labels")
    by_label = dict(zip(labels, targets, strict=True))

    build_values: dict[str, Any] = {}
    build_failures: dict[str, str] = {}
    if (
        not args.plan_only
        and labels
        and not (args.fail_fast and discovery_failures)
    ):
        per_build_jobs = args.jobs
        effective_workers = 1 if args.fail_fast else workers
        if per_build_jobs is None and effective_workers > 1:
            per_build_jobs = max(
                1, default_build_jobs() // min(effective_workers, len(labels))
            )

        def build(label: str) -> dict[str, Any]:
            target = by_label[label]
            child = argparse.Namespace(**vars(args))
            child.slicer = target.slicer
            child.ref = target.ref
            child.expected_commit = target.expected_commit
            child.source = None
            child.jobs = per_build_jobs
            child.skip_package = False
            return compact_build_output(target, build_one(child))

        build_values, build_failures = run_many_collect(
            labels, workers, args.fail_fast, build
        )

    builds: list[dict[str, Any]] = []
    for target in targets:
        label = f"{target.slicer}@{target.ref}"
        if args.plan_only:
            builds.append({**_release_target_payload(target), "status": "planned"})
        elif label in build_values:
            builds.append(build_values[label])
        elif label in build_failures:
            builds.append(
                {
                    **_release_target_payload(target),
                    "status": "failed",
                    "error": build_failures[label],
                }
            )
        else:
            builds.append(
                {
                    **_release_target_payload(target),
                    "status": "not-run",
                    "error": "not started after an earlier fail-fast failure",
                }
            )

    successful_builds = sum(
        build["status"] in {"built", "reused"} for build in builds
    )
    failed_builds = sum(build["status"] == "failed" for build in builds)
    not_run_builds = sum(build["status"] == "not-run" for build in builds)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "architecture": args.arch,
        "plan_only": args.plan_only,
        "policy": {
            "source": args.versions_from,
            "group_components": 3,
            "max_versions": maximum,
            "minimum_versions": minimums,
        },
        "summary": {
            "selected_slicers": len(names),
            "selected_versions": len(targets),
            "successful_builds": successful_builds,
            "failed_builds": failed_builds,
            "not_run_builds": not_run_builds,
            "skipped_slicers": skipped,
        },
        "discovery_failures": discovery_failures,
        "build_failures": build_failures,
        "builds": builds,
    }
    atomic_json(report_path, report)
    atomic_text(markdown_path, render_history_markdown(report))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "size_report": str(markdown_path),
                **report["summary"],
                "discovery_failures": discovery_failures,
                "build_failures": build_failures,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if not targets and not discovery_failures:
        raise SystemExit("No stable releases matched the requested history policy")
    if discovery_failures or build_failures:
        raise SystemExit(
            f"Historical build completed with {len(discovery_failures)} discovery "
            f"and {len(build_failures)} build failure(s)"
        )


def command_build_all(args: argparse.Namespace) -> None:
    targets = selected_manifest_targets(args.slicers, args.arch, args.all_refs)
    labels = [f"{name}@{ref}" if args.all_refs else name for name, ref in targets]
    by_label = dict(zip(labels, targets, strict=True))
    workers = getattr(args, "workers", 1)
    per_build_jobs = getattr(args, "jobs", None)
    effective_workers = 1 if args.fail_fast else workers
    if per_build_jobs is None and effective_workers > 1:
        per_build_jobs = max(1, default_build_jobs() // min(workers, len(labels)))

    def build(label: str) -> None:
        name, ref = by_label[label]
        child = argparse.Namespace(**vars(args))
        child.slicer = name
        child.ref = ref
        child.source = None
        child.jobs = per_build_jobs
        build_one(child)

    successful, failed = run_many(labels, workers, args.fail_fast, build)
    print(
        json.dumps(
            {"successful": successful, "failed": failed}, indent=2, sort_keys=True
        )
    )
    if failed:
        raise SystemExit(f"{len(failed)} slicer build(s) failed")


def command_test_all(args: argparse.Namespace) -> None:
    targets = selected_manifest_targets(args.slicers, args.arch, args.all_refs)
    labels = [f"{name}@{ref}" if args.all_refs else name for name, ref in targets]
    by_label = dict(zip(labels, targets, strict=True))

    def test(label: str) -> None:
        name, ref = by_label[label]
        child = argparse.Namespace(**vars(args))
        child.slicer = name
        child.ref = ref
        child.bundle = None
        test_one(child)

    successful, failed = run_many(
        labels, getattr(args, "workers", 1), args.fail_fast, test
    )
    print(
        json.dumps(
            {"successful": successful, "failed": failed}, indent=2, sort_keys=True
        )
    )
    if failed:
        raise SystemExit(f"{len(failed)} slicer smoke test(s) failed")


def add_common_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("slicer")
    parser.add_argument("--ref")
    parser.add_argument(
        "--source", type=Path, help="seed/fetch the mirror from a local clone"
    )
    parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    parser.add_argument(
        "--patches", choices=("none", "binary", "dump"), default="binary"
    )
    parser.add_argument("--container-cli")
    parser.add_argument("--image", default=DEFAULT_BUILDER_IMAGE)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="ignore a matching build result"
    )


def add_common_test_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=(ROOT / "../../simplyprint-web/cloud-slicer/backend").resolve(),
    )
    parser.add_argument(
        "--profiles-root", type=Path, default=(ROOT / "../slicer-profiles-db").resolve()
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--backend-version",
        help="adapter version override for a branch/commit build (default: nightly)",
    )
    parser.add_argument("--image", default=DEFAULT_SMOKE_IMAGE)
    parser.add_argument("--container-cli")
    parser.add_argument("--timeout", type=int, default=1800)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="slicerctl")
    commands = root.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=command_list)

    matrix_parser = commands.add_parser("matrix")
    matrix_parser.add_argument(
        "--capability", choices=("binary", "config_dump", "thumbnail"), default="binary"
    )
    matrix_parser.add_argument(
        "--all-refs",
        action="store_true",
        help="expand each default and declared supported ref",
    )
    matrix_parser.set_defaults(handler=command_matrix)

    patch_files_parser = commands.add_parser("patch-files")
    patch_files_parser.add_argument("slicer")
    patch_files_parser.add_argument("ref")
    patch_files_parser.add_argument("mode", choices=("binary", "dump"))
    patch_files_parser.add_argument("--null", action="store_true")
    patch_files_parser.set_defaults(handler=command_patch_files)

    verify_patches_parser = commands.add_parser(
        "verify-patches",
        help="check every selected patch stack without compiling slicers",
    )
    verify_patches_parser.add_argument(
        "--slicer",
        dest="slicers",
        action="append",
        help="manifest name to check (repeatable; default: all)",
    )
    verify_patches_parser.add_argument(
        "--ref",
        dest="refs",
        action="append",
        help="declared manifest ref to check (repeatable; default: all)",
    )
    verify_patches_parser.add_argument(
        "--mode",
        dest="modes",
        choices=("binary", "dump"),
        action="append",
        help="patch mode to check (repeatable; default: both enabled modes)",
    )
    verify_patches_parser.add_argument(
        "--source",
        type=Path,
        help="local Git transport seed (requires exactly one --slicer)",
    )
    verify_patches_parser.add_argument(
        "--include-head",
        action="store_true",
        help="also check HEAD for slicers with an enabled nightly CI lane",
    )
    verify_patches_parser.add_argument("--fail-fast", action="store_true")
    verify_patches_parser.add_argument("--json", action="store_true")
    verify_patches_parser.set_defaults(handler=command_verify_patches)

    image_parser = commands.add_parser("image")
    image_parser.add_argument(
        "--target", choices=("builder", "smoke"), default="builder"
    )
    image_parser.add_argument("--image")
    image_parser.add_argument("--cloud-slicer-image", default=DEFAULT_SMOKE_IMAGE)
    image_parser.add_argument("--container-cli")
    image_parser.set_defaults(handler=builder_image)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("slicer")
    prepare_parser.add_argument("--ref")
    prepare_parser.add_argument("--source", type=Path)
    prepare_parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    prepare_parser.add_argument(
        "--patches", choices=("none", "binary", "dump"), default="none"
    )

    def prepare_handler(args: argparse.Namespace) -> None:
        manifest = get_manifest(args.slicer)
        ref = args.ref or manifest.ref
        patches = legacy_patch_files(manifest, ref, args.patches)
        source, commit = prepare_source(
            manifest,
            ref,
            args.source,
            variant=f"{patch_variant(args.patches, patches)}-{args.arch}",
        )
        with file_lock(f"source-build:{source}"):
            apply_legacy_patches(source, patches)
        print(
            json.dumps(
                {
                    "source": str(source),
                    "commit": commit,
                    "arch": args.arch,
                    "patches": [str(path.relative_to(ROOT)) for path in patches],
                },
                indent=2,
            )
        )

    prepare_parser.set_defaults(handler=prepare_handler)

    build_parser = commands.add_parser("build")
    add_common_build_options(build_parser)
    build_parser.set_defaults(handler=build_one)

    build_all_parser = commands.add_parser("build-all")
    build_all_parser.add_argument("--slicer", dest="slicers", action="append")
    build_all_parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    build_all_parser.add_argument(
        "--patches", choices=("none", "binary", "dump"), default="binary"
    )
    build_all_parser.add_argument("--container-cli")
    build_all_parser.add_argument("--image", default=DEFAULT_BUILDER_IMAGE)
    build_all_parser.add_argument("--jobs", type=int)
    build_all_parser.add_argument("--skip-deps", action="store_true")
    build_all_parser.add_argument("--skip-package", action="store_true")
    build_all_parser.add_argument("--force", action="store_true")
    build_all_parser.add_argument("--fail-fast", action="store_true")
    build_all_parser.add_argument(
        "--all-refs", action="store_true", help="build every supported ref"
    )
    build_all_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="build this many slicers concurrently; --jobs is per slicer",
    )
    build_all_parser.set_defaults(handler=command_build_all)

    backfill_parser = commands.add_parser(
        "backfill",
        aliases=("build-history",),
        help="build a bounded history of stable upstream releases",
    )
    backfill_parser.add_argument("--slicer", dest="slicers", action="append")
    backfill_parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    backfill_parser.add_argument(
        "--versions-from",
        choices=("index", "git-tags", "github-releases"),
        default="index",
        help=(
            "select from the downloader index (default), all stable Git tags, "
            "or published stable GitHub releases"
        ),
    )
    history_limit = backfill_parser.add_mutually_exclusive_group()
    history_limit.add_argument(
        "--max-versions",
        type=int,
        default=3,
        help="newest downloader-compatible version groups to build (default: 3)",
    )
    history_limit.add_argument(
        "--all-versions",
        action="store_true",
        help="build every matching version group, optionally down to a minimum",
    )
    backfill_parser.add_argument(
        "--minimum-version",
        dest="minimum_versions",
        action="append",
        metavar="SLICER=VERSION",
        help=(
            "inclusive per-slicer threshold (repeatable; VERSION alone is accepted "
            "when one slicer is selected)"
        ),
    )
    backfill_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="resolve and record exact tags/commits without compiling",
    )
    backfill_parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (a Markdown size table is written beside it)",
    )
    backfill_parser.add_argument(
        "--patches", choices=("none", "binary", "dump"), default="binary"
    )
    backfill_parser.add_argument("--container-cli")
    backfill_parser.add_argument("--image", default=DEFAULT_BUILDER_IMAGE)
    backfill_parser.add_argument("--jobs", type=int)
    backfill_parser.add_argument("--skip-deps", action="store_true")
    backfill_parser.add_argument("--force", action="store_true")
    backfill_parser.add_argument("--fail-fast", action="store_true")
    backfill_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="discover/build this many slicers or versions concurrently",
    )
    backfill_parser.set_defaults(handler=command_backfill)

    test_parser = commands.add_parser("test")
    test_parser.add_argument("slicer")
    test_parser.add_argument("--ref")
    test_parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    test_parser.add_argument("--bundle", type=Path)
    add_common_test_options(test_parser)
    test_parser.set_defaults(handler=test_one)

    test_all_parser = commands.add_parser("test-all")
    test_all_parser.add_argument("--slicer", dest="slicers", action="append")
    test_all_parser.add_argument(
        "--arch", choices=sorted(SUPPORTED_ARCHITECTURES), default=native_architecture()
    )
    test_all_parser.add_argument("--fail-fast", action="store_true")
    test_all_parser.add_argument(
        "--all-refs", action="store_true", help="test every supported ref"
    )
    test_all_parser.add_argument("--workers", type=int, default=1)
    add_common_test_options(test_all_parser)
    test_all_parser.set_defaults(handler=command_test_all)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except subprocess.CalledProcessError as error:
        print(
            f"Command failed with exit status {error.returncode}: "
            + " ".join(map(str, error.cmd)),
            file=sys.stderr,
        )
        return error.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
