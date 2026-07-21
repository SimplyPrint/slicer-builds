#!/usr/bin/env python3
"""Generate validated include-only GitHub Actions matrices from slicer.toml."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SUPPORTED_ARCHITECTURES = ("x86-64", "arm64")
SUPPORTED_BUILD_TYPES = ("nightly", "latest_release")
LANE_CAPABILITY = {"binary": "binary", "config": "config_dump"}
LANE_KEYS = {
    "binary": {"enabled", "publish", "release", "nightly", "architectures"},
    "config": {
        "enabled",
        "publish",
        "release",
        "nightly",
        "repository",
        "expected_commit",
        "generator_only",
        "gui",
    },
}


class MatrixError(ValueError):
    """Raised when CI policy or a requested selection is invalid."""


@dataclass(frozen=True)
class ReleasePolicy:
    mode: str
    tag: str = ""


@dataclass(frozen=True)
class LanePolicy:
    slicer: str
    family: str
    repository: str
    repo_slug: str
    manifest_repo_slug: str
    ref_locks: tuple[tuple[str, str], ...]
    enabled: bool
    publish: bool
    release: ReleasePolicy
    nightly: bool
    architectures: tuple[str, ...]
    release_expected_commit: str = ""
    generator_only: bool = False
    gui: bool = False


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_bool(table: dict[str, Any], key: str, default: bool, label: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise MatrixError(f"{label}.{key} must be a boolean")
    return value


def _github_slug(repository: str, label: str) -> str:
    parsed = urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise MatrixError(f"{label} must be an https://github.com/<owner>/<repo> URL")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parsed.query or parsed.fragment:
        raise MatrixError(f"{label} must identify exactly one GitHub repository")
    owner, repo = parts
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo or not re.fullmatch(r"[A-Za-z0-9_.-]+", owner + repo):
        raise MatrixError(f"{label} contains an invalid GitHub owner or repository")
    return f"{owner}/{repo}"


def _tag(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*", value):
        raise MatrixError(f"{label} contains characters unsafe for a Git tag")
    if value.endswith((".", "/")) or ".." in value or "//" in value:
        raise MatrixError(f"{label} is not a valid Git tag shape")
    return value


def _release_policy(value: Any, label: str) -> ReleasePolicy:
    release = _require_string(value, label)
    if release == "latest":
        return ReleasePolicy("latest")
    if release == "none":
        return ReleasePolicy("none")
    if release.startswith("tag:") and release[4:].strip():
        return ReleasePolicy("tag", _tag(release[4:].strip(), label))
    raise MatrixError(f'{label} must be "latest", "none", or "tag:<exact-ref>"')


def _architectures(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{label} must be a non-empty array")
    result: list[str] = []
    for architecture in value:
        architecture = _require_string(architecture, label)
        if architecture not in SUPPORTED_ARCHITECTURES:
            supported = ", ".join(SUPPORTED_ARCHITECTURES)
            raise MatrixError(
                f"{label} contains {architecture!r}; supported values: {supported}"
            )
        if architecture in result:
            raise MatrixError(
                f"{label} contains duplicate architecture {architecture!r}"
            )
        result.append(architecture)
    return tuple(result)


def _ref_locks(data: dict[str, Any], label: str) -> tuple[tuple[str, str], ...]:
    """Return commit locks declared by the build manifest.

    The CI matrix generator intentionally validates the small subset it
    consumes instead of relying on a separate validation command having run
    first. This keeps fixed-tag anti-retag checks fail-closed in CI.
    """

    locks: dict[str, str] = {}
    default_ref = _require_string(data.get("default_ref"), f"{label}.default_ref")
    expected_commit = data.get("expected_commit")
    if expected_commit is not None:
        if not isinstance(expected_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40,64}", expected_commit
        ):
            raise MatrixError(
                f"{label}.expected_commit must be a lowercase full Git object ID"
            )
        locks[default_ref] = expected_commit

    supported_refs = data.get("supported_refs", [])
    if not isinstance(supported_refs, list):
        raise MatrixError(f"{label}.supported_refs must be an array of tables")
    for index, item in enumerate(supported_refs):
        item_label = f"{label}.supported_refs[{index}]"
        if not isinstance(item, dict):
            raise MatrixError(f"{item_label} must be a table")
        ref = _require_string(item.get("ref"), f"{item_label}.ref")
        commit = _require_string(
            item.get("expected_commit"), f"{item_label}.expected_commit"
        )
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise MatrixError(
                f"{item_label}.expected_commit must be a lowercase full Git object ID"
            )
        if ref in locks:
            raise MatrixError(f"{item_label}.ref duplicates declared ref {ref!r}")
        locks[ref] = commit
    return tuple(sorted(locks.items()))


def load_policies(root: Path, lane: str) -> list[LanePolicy]:
    if lane not in LANE_CAPABILITY:
        raise MatrixError(f"unknown lane: {lane}")

    manifest_paths = sorted((root / "slicers").glob("*/slicer.toml"))
    if not manifest_paths:
        raise MatrixError(f"no slicer manifests found under {root / 'slicers'}")

    policies: list[LanePolicy] = []
    names: set[str] = set()
    for path in manifest_paths:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise MatrixError(f"cannot read {path}: {exc}") from exc

        manifest_label = path.relative_to(root).as_posix()
        name = _require_string(data.get("name"), f"{manifest_label}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise MatrixError(f"{manifest_label}.name is unsafe for workflow paths")
        if name != path.parent.name:
            raise MatrixError(
                f"{manifest_label}.name must match directory {path.parent.name!r}"
            )
        if name in names:
            raise MatrixError(f"duplicate slicer name: {name}")
        names.add(name)

        family = _require_string(data.get("family"), f"{manifest_label}.family")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", family):
            raise MatrixError(
                f"{manifest_label}.family must be a safe lowercase identifier"
            )

        capabilities = data.get("capabilities")
        if not isinstance(capabilities, dict):
            raise MatrixError(f"{manifest_label}.capabilities must be a table")

        ci = data.get("ci")
        if not isinstance(ci, dict):
            raise MatrixError(f"{manifest_label}.ci must be a table")
        lane_table = ci.get(lane)
        if not isinstance(lane_table, dict):
            raise MatrixError(f"{manifest_label}.ci.{lane} must be a table")
        unknown = sorted(set(lane_table) - LANE_KEYS[lane])
        if unknown:
            raise MatrixError(
                f"{manifest_label}.ci.{lane} has unknown keys: {', '.join(unknown)}"
            )

        enabled = _optional_bool(
            lane_table, "enabled", False, f"{manifest_label}.ci.{lane}"
        )
        publish = _optional_bool(
            lane_table, "publish", False, f"{manifest_label}.ci.{lane}"
        )
        nightly = _optional_bool(
            lane_table, "nightly", False, f"{manifest_label}.ci.{lane}"
        )
        if publish and not enabled:
            raise MatrixError(
                f"{manifest_label}.ci.{lane} cannot publish while disabled"
            )

        capability = capabilities.get(LANE_CAPABILITY[lane])
        if not isinstance(capability, bool):
            raise MatrixError(
                f"{manifest_label}.capabilities.{LANE_CAPABILITY[lane]} must be a boolean"
            )
        if enabled and not capability:
            raise MatrixError(
                f"{manifest_label}.ci.{lane} is enabled but its capability is false"
            )

        release = _release_policy(
            lane_table.get("release", "none"), f"{manifest_label}.ci.{lane}.release"
        )
        if enabled and release.mode == "none" and not nightly:
            raise MatrixError(
                f"{manifest_label}.ci.{lane} is enabled but has no release or nightly build"
            )
        if enabled and publish and release.mode == "none":
            raise MatrixError(
                f"{manifest_label}.ci.{lane} cannot publish without a release policy"
            )

        manifest_repository = _require_string(
            data.get("repository"), f"{manifest_label}.repository"
        )
        manifest_repo_slug = _github_slug(
            manifest_repository, f"{manifest_label}.repository"
        )
        repository = _require_string(
            lane_table.get("repository", data.get("repository")),
            f"{manifest_label}.ci.{lane}.repository",
        )
        repo_slug = _github_slug(repository, f"{manifest_label}.ci.{lane}.repository")
        release_expected_commit = lane_table.get("expected_commit", "")
        if release_expected_commit:
            if not isinstance(release_expected_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40,64}", release_expected_commit
            ):
                raise MatrixError(
                    f"{manifest_label}.ci.{lane}.expected_commit must be a "
                    "lowercase full Git object ID"
                )
            if release.mode != "tag":
                raise MatrixError(
                    f"{manifest_label}.ci.{lane}.expected_commit requires "
                    "release = \"tag:<exact-ref>\""
                )
        ref_locks = _ref_locks(data, manifest_label)
        root_expected_commit = dict(ref_locks).get(release.tag, "")
        if (
            release_expected_commit
            and repo_slug == manifest_repo_slug
            and root_expected_commit
            and release_expected_commit != root_expected_commit
        ):
            raise MatrixError(
                f"{manifest_label}.ci.{lane}.expected_commit conflicts with "
                "the manifest ref lock"
            )

        root_architectures = data.get("architectures")
        architectures = (
            _architectures(
                lane_table.get("architectures", root_architectures),
                f"{manifest_label}.ci.binary.architectures",
            )
            if lane == "binary"
            else ("x86-64",)
        )
        generator_only = (
            _optional_bool(
                lane_table, "generator_only", False, f"{manifest_label}.ci.config"
            )
            if lane == "config"
            else False
        )
        gui = (
            _optional_bool(lane_table, "gui", False, f"{manifest_label}.ci.config")
            if lane == "config"
            else False
        )

        policies.append(
            LanePolicy(
                slicer=name,
                family=family,
                repository=repository,
                repo_slug=repo_slug,
                manifest_repo_slug=manifest_repo_slug,
                ref_locks=ref_locks,
                enabled=enabled,
                publish=publish,
                release=release,
                nightly=nightly,
                architectures=architectures,
                release_expected_commit=release_expected_commit,
                generator_only=generator_only,
                gui=gui,
            )
        )
    return policies


def generate_matrix(
    root: Path,
    lane: str,
    selected_slicer: str = "all",
    selected_build_type: str = "all",
    selected_arch: str = "all",
    release_tag_override: str = "",
) -> dict[str, list[dict[str, Any]]]:
    policies = load_policies(root, lane)
    by_name = {policy.slicer: policy for policy in policies}

    if selected_slicer != "all" and selected_slicer not in by_name:
        available = ", ".join(sorted(by_name))
        raise MatrixError(
            f"unknown slicer {selected_slicer!r}; available slicers: {available}"
        )
    selected_policies = (
        policies if selected_slicer == "all" else [by_name[selected_slicer]]
    )
    if selected_slicer != "all" and not selected_policies[0].enabled:
        raise MatrixError(f"{selected_slicer} has ci.{lane}.enabled=false")

    if selected_build_type not in ("all", *SUPPORTED_BUILD_TYPES):
        raise MatrixError(f"unknown build type: {selected_build_type}")
    if lane == "binary" and selected_arch not in ("all", *SUPPORTED_ARCHITECTURES):
        raise MatrixError(f"unknown architecture: {selected_arch}")
    if release_tag_override:
        _tag(release_tag_override, "--release-tag")
        if selected_slicer == "all":
            raise MatrixError("--release-tag requires selecting exactly one slicer")
        if selected_policies[0].release.mode == "none":
            raise MatrixError(f"{selected_slicer} has no release policy to override")

    rows: list[dict[str, Any]] = []
    for policy in selected_policies:
        if not policy.enabled:
            continue

        build_types: list[str] = []
        if policy.nightly:
            build_types.append("nightly")
        if policy.release.mode != "none":
            build_types.append("latest_release")
        if selected_build_type != "all":
            build_types = [item for item in build_types if item == selected_build_type]

        architectures = policy.architectures
        if lane == "binary" and selected_arch != "all":
            architectures = tuple(
                item for item in architectures if item == selected_arch
            )

        for build_type in build_types:
            for architecture in architectures:
                row: dict[str, Any] = {
                    "build-type": build_type,
                    "family": policy.family,
                    "nightly_enabled": policy.nightly,
                    "publish": policy.publish,
                    "release_mode": policy.release.mode,
                    "release_tag": policy.release.tag,
                    "repo": policy.repo_slug,
                    "slicer": policy.slicer,
                }
                selected_release_tag = release_tag_override or policy.release.tag
                expected_commit = ""
                if (
                    build_type == "latest_release"
                    and selected_release_tag
                    and policy.repo_slug == policy.manifest_repo_slug
                ):
                    expected_commit = dict(policy.ref_locks).get(
                        selected_release_tag, ""
                    )
                if (
                    build_type == "latest_release"
                    and selected_release_tag == policy.release.tag
                    and policy.release_expected_commit
                ):
                    expected_commit = policy.release_expected_commit
                row["expected_commit"] = expected_commit
                if lane == "binary":
                    row["arch"] = architecture
                else:
                    row["config_generator_only"] = policy.generator_only
                    row["gui"] = policy.gui
                rows.append(row)

    if not rows:
        requested = f"slicer={selected_slicer}, build-type={selected_build_type}"
        if lane == "binary":
            requested += f", arch={selected_arch}"
        raise MatrixError(f"no enabled ci.{lane} combinations match {requested}")

    rows.sort(
        key=lambda row: (
            row["slicer"],
            SUPPORTED_BUILD_TYPES.index(row["build-type"]),
            SUPPORTED_ARCHITECTURES.index(row.get("arch", "x86-64")),
        )
    )
    return {"include": rows}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--lane", choices=tuple(LANE_CAPABILITY), required=True)
    parser.add_argument("--slicer", default="all")
    parser.add_argument("--build-type", default="all")
    parser.add_argument("--arch", default="all")
    parser.add_argument("--release-tag", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        matrix = generate_matrix(
            args.root.resolve(),
            args.lane,
            args.slicer,
            args.build_type,
            args.arch,
            args.release_tag.strip(),
        )
    except MatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(matrix, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
