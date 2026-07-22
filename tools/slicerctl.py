#!/usr/bin/env python3
"""List slicer manifests and validate their patch stacks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = ROOT / ".work"
COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")
RETAINED_RELEASE_LIMIT = 3


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


def fail(message: str) -> None:
    raise SystemExit(message)


def safe_ref(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or SAFE_REF_RE.fullmatch(value) is None
        or ".." in value
        or value.endswith((".", "/"))
    ):
        fail(f"{label} is not a safe Git ref")
    return value


def ref_specs(manifest: Manifest) -> list[dict[str, str | None]]:
    """Read retained release locks from the successful-artifact index."""

    index_path = manifest.directory / "out" / "_index.json"
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Cannot read {index_path}: {error}")
    versions = index.get("versions") if isinstance(index, dict) else None
    if not isinstance(versions, dict):
        fail(f"{index_path}: versions must be an object")

    def version_parts(ref: str) -> tuple[int, ...]:
        clean = ref.removeprefix("version_").removeprefix("v")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", clean) is None:
            fail(f"{index_path}: unsupported release version {ref!r}")
        return tuple(int(part) for part in clean.split("."))

    grouped: dict[str, tuple[str, tuple[int, ...]]] = {}
    for ref in versions:
        if ref == "nightly":
            continue
        parts = version_parts(ref)
        clean = ref.removeprefix("version_").removeprefix("v")
        clean_parts = clean.split(".")
        group = ".".join(clean_parts[:3]) if len(parts) >= 3 else clean
        previous = grouped.get(group)
        if previous is None or parts > previous[1]:
            grouped[group] = (ref, parts)

    selected = sorted(grouped.values(), key=lambda item: item[1], reverse=True)
    specs: list[dict[str, str | None]] = []
    for ref, _parts in selected:
        version = versions[ref]
        commit = version.get("upstream_ref") if isinstance(version, dict) else None
        # Old generated entries may predate upstream SHA metadata. They remain
        # downloadable but are not safe inputs for compatibility validation.
        if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
            continue
        specs.append({"ref": safe_ref(ref, f"{index_path}: ref"), "expected_commit": commit})
        if len(specs) == RETAINED_RELEASE_LIMIT:
            break
    return specs


def validate_manifest(manifest: Manifest) -> None:
    data = manifest.data
    name = data.get("name")
    if not isinstance(name, str) or SAFE_NAME_RE.fullmatch(name) is None:
        fail(f"{manifest.path}: invalid name")
    if name != manifest.directory.name:
        fail(f"{manifest.path}: name must match its directory")
    repository = data.get("repository")
    if not isinstance(repository, str) or not repository.startswith(
        "https://github.com/"
    ):
        fail(f"{manifest.path}: repository must be an HTTPS GitHub URL")
    if not isinstance(data.get("capabilities"), dict):
        fail(f"{manifest.path}: capabilities must be a table")
    if not isinstance(data.get("ci"), dict):
        fail(f"{manifest.path}: ci must be a table")

    seen: set[str] = set()
    for spec in ref_specs(manifest):
        ref = safe_ref(spec["ref"], f"{manifest.path}: ref")
        if ref in seen:
            fail(f"{manifest.path}: duplicate ref {ref!r}")
        seen.add(ref)
        commit = spec.get("expected_commit")
        if commit is not None and (
            not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None
        ):
            fail(f"{manifest.path}: invalid commit lock for {ref!r}")

    patch_config = data.get("patches", {})
    if not isinstance(patch_config, dict):
        fail(f"{manifest.path}: patches must be a table")
    shared = patch_config.get("shared", [])
    if not isinstance(shared, list) or len(shared) != len(set(shared)):
        fail(f"{manifest.path}: patches.shared must contain unique names")
    for name in shared:
        if not isinstance(name, str) or SAFE_NAME_RE.fullmatch(name) is None:
            fail(f"{manifest.path}: invalid shared patch-set name")
        if not (ROOT / "patches" / name).is_dir():
            fail(f"{manifest.path}: unknown shared patch set {name!r}")


def manifests() -> dict[str, Manifest]:
    result: dict[str, Manifest] = {}
    for path in sorted((ROOT / "slicers").glob("*/slicer.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            fail(f"Cannot read {path}: {error}")
        manifest = Manifest(path, data)
        validate_manifest(manifest)
        if manifest.name in result:
            fail(f"Duplicate slicer manifest: {manifest.name}")
        result[manifest.name] = manifest
    if not result:
        fail("No slicer manifests found")
    return result


def get_manifest(name: str) -> Manifest:
    available = manifests()
    if name not in available:
        fail(f"Unknown slicer: {name}")
    return available[name]


def version_directory(root: Path, ref: str) -> Path:
    safe_ref(ref, "patch ref")
    exact = root / ref
    if ref != "HEAD" or any(exact.glob("*.patch")):
        return exact
    return root / "nightly"


def patch_roots(manifest: Manifest) -> list[Path]:
    shared = manifest.data.get("patches", {}).get("shared", [])
    return [
        *(ROOT / "patches" / name for name in shared),
        manifest.directory / "patches",
    ]


def patches_in(*directories: Path) -> list[Path]:
    return [
        patch
        for directory in directories
        if directory.is_dir()
        for patch in sorted(directory.glob("*.patch"))
    ]


def patch_files(manifest: Manifest, ref: str, mode: str) -> list[Path]:
    if mode not in {"binary", "dump"}:
        fail(f"Unknown patch mode: {mode}")
    selected: list[Path] = []
    aliases = manifest.data.get("patches", {}).get("binary_ref_aliases", {})
    patch_ref = aliases.get(ref, ref) if mode == "binary" else ref
    safe_ref(patch_ref, "patch ref alias")

    for root in patch_roots(manifest):
        common = root / "common"
        selected.extend(
            patches_in(common / "all", version_directory(common, ref))
        )
        if mode == "binary":
            binary = root / "binary"
            selected.extend(
                patches_in(binary / "all", version_directory(binary, patch_ref))
            )
            continue

        version = version_directory(root, ref)
        selected.extend(patches_in(root / "all", version))
        if not (version / "dump_configs.patch").is_file():
            fallback = root / "dump_configs.patch"
            if fallback.is_file():
                selected.append(fallback)
    return selected


def run(
    command: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> str:
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def prepare_mirror(manifest: Manifest, seed: Path | None) -> tuple[Path, str]:
    mirror = WORK_ROOT / "git" / f"{manifest.name}.git"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    transport = str(seed.resolve()) if seed else manifest.data["repository"]

    if not mirror.exists():
        run(["git", "clone", "--mirror", transport, mirror])
    elif not mirror.is_dir():
        fail(f"Git mirror path is not a directory: {mirror}")
    else:
        run(["git", "fetch", "--prune", "--force", "--tags", transport], cwd=mirror)
    run(
        ["git", "remote", "set-url", "origin", manifest.data["repository"]],
        cwd=mirror,
    )
    return mirror, transport


def resolve_ref(mirror: Path, transport: str, ref: str) -> tuple[str | None, str]:
    if ref == "HEAD":
        completed = subprocess.run(
            ["git", "ls-remote", transport, "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            return None, git_error(completed)
        fields = completed.stdout.split()
        if not fields or COMMIT_RE.fullmatch(fields[0]) is None:
            return None, "transport returned an invalid HEAD commit"
        commit = fields[0]
        subprocess.run(
            ["git", "fetch", "--quiet", transport, commit],
            cwd=mirror,
            check=False,
        )
        return commit, ""

    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=mirror,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None, git_error(completed)
    commit = completed.stdout.strip()
    if COMMIT_RE.fullmatch(commit) is None:
        return None, f"Git returned an invalid commit for {ref!r}"
    return commit, ""


def git_error(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    return (detail or f"git exited with {completed.returncode}")[-4000:]


def verify_stack(
    mirror: Path, commit: str, selected: Sequence[Path]
) -> dict[str, str] | None:
    with tempfile.TemporaryDirectory(prefix="slicer-patch-index-") as temporary:
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        loaded = subprocess.run(
            ["git", "read-tree", commit],
            cwd=mirror,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if loaded.returncode != 0:
            return {"stage": "read-tree", "error": git_error(loaded)}

        for patch in selected:
            applied = subprocess.run(
                ["git", "apply", "--cached", "--whitespace=fix", patch],
                cwd=mirror,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if applied.returncode != 0:
                return {
                    "stage": "apply",
                    "patch": str(patch.relative_to(ROOT)),
                    "error": git_error(applied),
                }
    return None


def verification_refs(manifest: Manifest, include_head: bool) -> list[dict[str, str | None]]:
    specs = ref_specs(manifest)
    ci = manifest.data["ci"]
    nightly = any(
        isinstance(lane, dict)
        and lane.get("enabled", False)
        and lane.get("nightly", False)
        and lane.get("repository", manifest.data["repository"])
        == manifest.data["repository"]
        for lane in (ci.get("binary", {}), ci.get("config", {}))
    )
    if include_head and nightly and all(spec["ref"] != "HEAD" for spec in specs):
        specs.append({"ref": "HEAD", "expected_commit": None})
    return specs


def command_list(args: argparse.Namespace) -> None:
    values = []
    for manifest in manifests().values():
        specs = ref_specs(manifest)
        values.append(
            {
                "name": manifest.name,
                "family": manifest.data["family"],
                "repository": manifest.data["repository"],
                "ref": specs[0]["ref"] if specs else "HEAD",
                "refs": specs,
                "architectures": manifest.data["architectures"],
                "capabilities": manifest.data["capabilities"],
            }
        )
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for value in values:
            print(f"{value['name']:<24} {value['family']:<8} {value['ref']}")


def command_patch_files(args: argparse.Namespace) -> None:
    selected = patch_files(get_manifest(args.slicer), args.ref, args.mode)
    rendered = [str(path.relative_to(ROOT)) for path in selected]
    if args.null:
        for path in rendered:
            sys.stdout.buffer.write(os.fsencode(path) + b"\0")
    else:
        print("\n".join(rendered))


def command_verify(args: argparse.Namespace) -> None:
    available = manifests()
    names = list(dict.fromkeys(args.slicers or available))
    unknown = sorted(set(names) - set(available))
    if unknown:
        fail(f"Unknown slicers: {', '.join(unknown)}")
    if args.source and len(names) != 1:
        fail("--source requires exactly one --slicer")

    requested_refs = set(args.refs or [])
    requested_modes = list(dict.fromkeys(args.modes or ("binary", "dump")))
    capability = {"binary": "binary", "dump": "config_dump"}
    plans: list[tuple[Manifest, dict[str, str | None], str, list[Path]]] = []
    matched_refs: set[str] = set()
    matched_modes: set[str] = set()
    for name in names:
        manifest = available[name]
        specs = [
            spec
            for spec in verification_refs(manifest, args.include_head)
            if not requested_refs or spec["ref"] in requested_refs
        ]
        matched_refs.update(str(spec["ref"]) for spec in specs)
        modes = [
            mode
            for mode in requested_modes
            if manifest.data["capabilities"].get(capability[mode], False)
        ]
        matched_modes.update(modes)
        for spec in specs:
            for mode in modes:
                plans.append(
                    (
                        manifest,
                        spec,
                        mode,
                        patch_files(manifest, str(spec["ref"]), mode),
                    )
                )

    missing_refs = sorted(requested_refs - matched_refs)
    if missing_refs:
        fail(f"Refs are not declared: {', '.join(missing_refs)}")
    missing_modes = sorted(set(args.modes or ()) - matched_modes)
    if missing_modes:
        fail(f"Patch modes are not enabled: {', '.join(missing_modes)}")

    results: list[dict[str, Any]] = []
    mirrors: dict[str, tuple[Path, str] | BaseException] = {}
    for manifest, spec, mode, selected in plans:
        if manifest.name not in mirrors:
            try:
                mirrors[manifest.name] = prepare_mirror(manifest, args.source)
            except (OSError, subprocess.CalledProcessError, SystemExit) as error:
                mirrors[manifest.name] = error

        failure: dict[str, str] | None = None
        commit: str | None = None
        mirror_result = mirrors[manifest.name]
        if isinstance(mirror_result, BaseException):
            failure = {"stage": "prepare-mirror", "error": str(mirror_result)}
        else:
            mirror, transport = mirror_result
            commit, error = resolve_ref(mirror, transport, str(spec["ref"]))
            if commit is None:
                failure = {"stage": "resolve-ref", "error": error}
            elif spec.get("expected_commit") and commit != spec["expected_commit"]:
                failure = {
                    "stage": "expected-commit",
                    "error": f"resolved to {commit}, expected {spec['expected_commit']}",
                }
            else:
                failure = verify_stack(mirror, commit, selected)

        result: dict[str, Any] = {
            "slicer": manifest.name,
            "ref": spec["ref"],
            "expected_commit": spec.get("expected_commit"),
            "commit": commit,
            "mode": mode,
            "status": "failed" if failure else "passed",
            "patch_count": len(selected),
            "patches": [str(path.relative_to(ROOT)) for path in selected],
        }
        if failure:
            result.update(failure)
        results.append(result)
        if failure and args.fail_fast:
            break

    failed = sum(result["status"] == "failed" for result in results)
    payload = {
        "ok": failed == 0 and len(results) == len(plans),
        "summary": {
            "planned_stacks": len(plans),
            "attempted_stacks": len(results),
            "passed_stacks": len(results) - failed,
            "failed_stacks": failed,
            "patches_selected": sum(len(selected) for *_, selected in plans),
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            label = f"{result['slicer']}@{result['ref']} [{result['mode']}]"
            print(f"{result['status'].upper()} {label}: {result['patch_count']} patches")
    if not payload["ok"]:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="slicerctl")
    commands = root.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=command_list)

    patch_parser = commands.add_parser("patch-files")
    patch_parser.add_argument("slicer")
    patch_parser.add_argument("ref")
    patch_parser.add_argument("mode", choices=("binary", "dump"))
    patch_parser.add_argument("--null", action="store_true")
    patch_parser.set_defaults(handler=command_patch_files)

    verify = commands.add_parser("verify-patches")
    verify.add_argument("--slicer", dest="slicers", action="append")
    verify.add_argument("--ref", dest="refs", action="append")
    verify.add_argument(
        "--mode", dest="modes", choices=("binary", "dump"), action="append"
    )
    verify.add_argument("--source", type=Path)
    verify.add_argument("--include-head", action="store_true")
    verify.add_argument("--fail-fast", action="store_true")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(handler=command_verify)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit status {error.returncode}", file=sys.stderr)
        return error.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
