#!/usr/bin/env python3
"""Generate validated include-only GitHub Actions matrices from slicer.toml."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SUPPORTED_ARCHITECTURES = ("x86-64", "arm64")
SUPPORTED_BUILD_TYPES = ("nightly", "latest_release")
CONSUMER_RELEASE_LIMIT = 3
LANE_CAPABILITY = {"binary": "binary", "config": "config_dump"}
LANE_KEYS = {
    "binary": {
        "enabled",
        "publish",
        "release",
        "release_repository",
        "nightly",
        "architectures",
    },
    "config": {
        "enabled",
        "publish",
        "release",
        "release_repository",
        "nightly",
        "repository",
        "generator_only",
        "gui",
    },
}


class MatrixError(ValueError):
    """Raised when CI policy or a requested selection is invalid."""


@dataclass(frozen=True)
class ReleasePolicy:
    mode: str


@dataclass(frozen=True)
class LanePolicy:
    slicer: str
    family: str
    repo_slug: str
    release_repo_slug: str
    enabled: bool
    publish: bool
    release: ReleasePolicy
    nightly: bool
    architectures: tuple[str, ...]
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
    raise MatrixError(f'{label} must be "latest" or "none"')


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


def _clean_version(ref: str) -> str:
    if ref.startswith("version_"):
        return ref.removeprefix("version_")
    return ref.removeprefix("v") if ref.startswith("v") else ref


def _version_parts(ref: str) -> tuple[int, ...]:
    clean = _clean_version(ref)
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", clean) is None:
        raise MatrixError(f"indexed release {ref!r} is not a stable numeric version")
    return tuple(int(part) for part in clean.split("."))


def _load_index(root: Path, policy: LanePolicy) -> dict[str, Any]:
    index_path = root / "slicers" / policy.slicer / "out" / "_index.json"
    if not index_path.is_file():
        return {"latest": None, "versions": {}}

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read consumer index {index_path}: {exc}") from exc
    versions = index.get("versions") if isinstance(index, dict) else None
    if not isinstance(versions, dict):
        raise MatrixError(f"invalid consumer index: {index_path}")
    return index


def _consumer_releases(root: Path, policy: LanePolicy) -> list[dict[str, str]]:
    """Return retained releases using the successful-build index as the lock."""

    index = _load_index(root, policy)
    versions = index["versions"]

    grouped: dict[str, tuple[str, tuple[int, ...]]] = {}
    for ref in versions:
        if ref == "nightly":
            continue
        parts = _version_parts(ref)
        clean_parts = _clean_version(ref).split(".")
        group = ".".join(clean_parts[:3]) if len(parts) >= 3 else _clean_version(ref)
        previous = grouped.get(group)
        if previous is None or parts > previous[1]:
            grouped[group] = (ref, parts)

    selected = sorted(grouped.values(), key=lambda item: item[1], reverse=True)
    result: list[dict[str, str]] = []
    for ref, _parts in selected:
        version = versions[ref]
        commit = version.get("upstream_ref") if isinstance(version, dict) else None
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
            continue
        published_at = version.get("upstream_published_at", "")
        result.append(
            {
                "tag": ref,
                "commit": commit,
                "published_at": published_at if isinstance(published_at, str) else "",
            }
        )
        if len(result) == CONSUMER_RELEASE_LIMIT:
            break
    return result


class GitHub:
    """Small cached GitHub API client used while resolving the build matrix."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.token = os.environ.get("GITHUB_TOKEN", "").strip()

    def get(self, path: str, *, missing_ok: bool = False) -> Any:
        if path in self.cache:
            return self.cache[path]
        request = urllib.request.Request(
            f"https://api.github.com/{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "slicer-builds-matrix",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        detail = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    value = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if missing_ok and exc.code == 404:
                    return {}
                detail = f"HTTP {exc.code}"
                if exc.code != 429 and exc.code < 500:
                    raise MatrixError(f"GitHub API failed for {path}: {detail}") from exc
            except (OSError, json.JSONDecodeError) as exc:
                detail = str(exc)
            if attempt < 2:
                time.sleep(2**attempt)
        else:
            raise MatrixError(f"GitHub API failed for {path}: {detail}")
        self.cache[path] = value
        return value

    def release(self, repository: str, override: str = "") -> tuple[str, str]:
        if override:
            encoded = urllib.parse.quote(override, safe="")
            value = self.get(
                f"repos/{repository}/releases/tags/{encoded}", missing_ok=True
            )
            tag = override
        else:
            value = self.get(f"repos/{repository}/releases/latest")
            if not isinstance(value, dict):
                raise MatrixError(f"GitHub returned an invalid release for {repository}")
            tag = value.get("tag_name")
        if not isinstance(value, dict):
            raise MatrixError(f"GitHub returned an invalid release for {repository}")
        tag = _tag(
            _require_string(tag, f"latest release of {repository}"),
            f"latest release of {repository}",
        )
        published_at = value.get("published_at", "")
        return tag, published_at if isinstance(published_at, str) else ""

    def commit(self, repository: str, ref: str) -> str:
        encoded = urllib.parse.quote(ref, safe="")
        value = self.get(f"repos/{repository}/commits/{encoded}")
        commit = value.get("sha") if isinstance(value, dict) else None
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
            raise MatrixError(f"GitHub returned an invalid commit for {repository}@{ref}")
        return commit

    def commit_before(self, repository: str, ref: str, timestamp: str) -> str:
        query = urllib.parse.urlencode(
            {"sha": ref, "until": timestamp, "per_page": "1"}
        )
        value = self.get(f"repos/{repository}/commits?{query}")
        commit = value[0].get("sha") if isinstance(value, list) and value else None
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
            raise MatrixError(
                f"GitHub returned no {repository}@{ref} commit before {timestamp}"
            )
        return commit


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
        _github_slug(manifest_repository, f"{manifest_label}.repository")
        repository = _require_string(
            lane_table.get("repository", data.get("repository")),
            f"{manifest_label}.ci.{lane}.repository",
        )
        repo_slug = _github_slug(repository, f"{manifest_label}.ci.{lane}.repository")
        release_repository = _require_string(
            lane_table.get("release_repository", repository),
            f"{manifest_label}.ci.{lane}.release_repository",
        )
        release_repo_slug = _github_slug(
            release_repository, f"{manifest_label}.ci.{lane}.release_repository"
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
                repo_slug=repo_slug,
                release_repo_slug=release_repo_slug,
                enabled=enabled,
                publish=publish,
                release=release,
                nightly=nightly,
                architectures=architectures,
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
    force_rebuild: bool = False,
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

    if selected_build_type not in ("all", "consumer", *SUPPORTED_BUILD_TYPES):
        raise MatrixError(f"unknown build type: {selected_build_type}")
    if selected_build_type == "consumer" and lane != "binary":
        raise MatrixError("consumer builds are only available for the binary lane")
    if lane == "binary" and selected_arch not in ("all", *SUPPORTED_ARCHITECTURES):
        raise MatrixError(f"unknown architecture: {selected_arch}")
    if release_tag_override:
        _tag(release_tag_override, "--release-tag")
        if selected_build_type == "consumer":
            raise MatrixError("--release-tag cannot be combined with consumer builds")
        if selected_slicer == "all":
            raise MatrixError("--release-tag requires selecting exactly one slicer")
        if selected_policies[0].release.mode == "none":
            raise MatrixError(f"{selected_slicer} has no release policy to override")

    github = GitHub()
    rows: list[dict[str, Any]] = []
    for policy in selected_policies:
        if not policy.enabled:
            continue

        architectures = policy.architectures
        if lane == "binary" and selected_arch != "all":
            architectures = tuple(
                item for item in architectures if item == selected_arch
            )

        index = _load_index(root, policy)
        latest_tag = ""
        latest_commit = ""
        latest_published_at = ""
        if policy.release.mode != "none":
            override = release_tag_override
            latest_tag, latest_published_at = github.release(
                policy.release_repo_slug, override
            )
            latest_commit = github.commit(
                policy.repo_slug, f"refs/tags/{latest_tag}"
            )

        needs_nightly = policy.nightly and selected_build_type in (
            "all",
            "consumer",
            "nightly",
        )
        head_commit = github.commit(policy.repo_slug, "HEAD") if needs_nightly else ""
        release_at_head = bool(latest_commit and head_commit == latest_commit)

        stable: dict[str, dict[str, str]] = {}
        if selected_build_type == "consumer":
            stable = {item["tag"]: item for item in _consumer_releases(root, policy)}
            if latest_tag:
                stable[latest_tag] = {
                    "tag": latest_tag,
                    "commit": latest_commit,
                    "published_at": latest_published_at,
                }
            stable = dict(
                sorted(
                    stable.items(),
                    key=lambda item: _version_parts(item[0]),
                    reverse=True,
                )[:CONSUMER_RELEASE_LIMIT]
            )
        elif selected_build_type in ("all", "latest_release") and latest_tag:
            stable[latest_tag] = {
                "tag": latest_tag,
                "commit": latest_commit,
                "published_at": latest_published_at,
            }

        for architecture in architectures:
            candidates: list[dict[str, Any]] = []
            for item in stable.values():
                version = index["versions"].get(item["tag"], {})
                same_source = isinstance(version, dict) and version.get(
                    "upstream_ref"
                ) == item["commit"]
                if lane == "binary":
                    present = same_source and any(
                        isinstance(build, dict)
                        and build.get("platform") == "linux"
                        and build.get("arch") == architecture
                        for build in version.get("builds", [])
                    ) if isinstance(version, dict) else False
                else:
                    present = same_source and isinstance(version.get("config"), dict)
                if present and not (force_rebuild or selected_build_type == "consumer"):
                    continue
                candidates.append(
                    {
                        "build-type": "latest_release",
                        "build_ref": item["commit"],
                        "patch_version": item["tag"],
                        "publish_nightly": False,
                        "publish_release": policy.publish,
                        "release_tag": item["tag"],
                        "release_published_at": item["published_at"],
                        "mark_latest": item["tag"] == latest_tag,
                    }
                )

            if needs_nightly:
                latest_version = index["versions"].get(latest_tag, {})
                same_latest_source = isinstance(
                    latest_version, dict
                ) and latest_version.get("upstream_ref") == latest_commit
                if lane == "binary":
                    latest_present = same_latest_source and any(
                        isinstance(build, dict)
                        and build.get("platform") == "linux"
                        and build.get("arch") == architecture
                        for build in latest_version.get("builds", [])
                    ) if isinstance(latest_version, dict) else False
                else:
                    latest_present = same_latest_source and isinstance(
                        latest_version.get("config"), dict
                    )
                candidates.append(
                    {
                        "build-type": "nightly",
                        "build_ref": head_commit,
                        # Match the old one-build/two-publish behavior: a release
                        # at HEAD uses the release patch stack for both aliases.
                        "patch_version": latest_tag if release_at_head else "nightly",
                        "publish_nightly": policy.publish,
                        "publish_release": (
                            policy.publish
                            and release_at_head
                            and (force_rebuild or not latest_present)
                        ),
                        "release_tag": latest_tag,
                        "release_published_at": latest_published_at,
                        "mark_latest": release_at_head,
                    }
                )

            # Collapse logical channels into physical builds. The release-at-HEAD
            # rule above gives nightly and release the same patch context, so the
            # deduplication key is both source- and recipe-safe.
            physical: dict[tuple[str, str], dict[str, Any]] = {}
            for candidate in candidates:
                key = (candidate["build_ref"], candidate["patch_version"])
                previous = physical.get(key)
                if previous is None:
                    physical[key] = candidate
                    continue
                previous["publish_nightly"] |= candidate["publish_nightly"]
                previous["publish_release"] |= candidate["publish_release"]
                previous["mark_latest"] |= candidate["mark_latest"]
                if candidate["build-type"] == "nightly":
                    previous["build-type"] = "nightly"

            for candidate in physical.values():
                cura_conan_config_ref = ""
                if policy.slicer == "Cura" and candidate["release_published_at"]:
                    cura_conan_config_ref = github.commit_before(
                        "Ultimaker/conan-config",
                        "master",
                        candidate["release_published_at"],
                    )
                row = {
                    **candidate,
                    "arch": architecture,
                    "artifact_key": (
                        f"{candidate['release_tag']}-nightly"
                        if candidate["publish_nightly"] and candidate["publish_release"]
                        else "nightly"
                        if candidate["publish_nightly"]
                        else candidate["release_tag"]
                    ),
                    "consumer_rebuild": selected_build_type == "consumer",
                    "cura_conan_config_ref": cura_conan_config_ref,
                    "family": policy.family,
                    "nightly_enabled": policy.nightly,
                    "publish": policy.publish,
                    "repo": policy.repo_slug,
                    "release_repo": policy.release_repo_slug,
                    "skip": False,
                    "slicer": policy.slicer,
                }
                if lane == "config":
                    row["config_generator_only"] = policy.generator_only
                    row["gui"] = policy.gui
                rows.append(row)

    if not rows:
        # GitHub Actions rejects an empty matrix. One explicitly skipped row lets
        # an up-to-date workflow finish successfully without renting a runner.
        rows.append(
            {
                "arch": "x86-64",
                "artifact_key": "skip",
                "build-type": "latest_release",
                "publish": False,
                "skip": True,
                "slicer": selected_slicer,
            }
        )

    rows.sort(
        key=lambda row: (
            row.get("slicer", ""),
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
    parser.add_argument("--force-rebuild", action="store_true")
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
            args.force_rebuild,
        )
    except MatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(matrix, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
