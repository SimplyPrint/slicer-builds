#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def compact_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value not in (None, "")}


def parse_metadata(items: list[str]) -> dict:
    metadata = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid metadata item '{item}', expected key=value")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def version_sort_key(version: str) -> tuple:
    normalized = version.removeprefix("version_").removeprefix("v")
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", normalized)
        if part
    )


def should_promote_latest(index: dict, version: str) -> bool:
    current = index.get("latest")
    if not isinstance(current, str) or not current:
        return True

    if current not in index.get("versions", {}):
        return True

    return version_sort_key(version) > version_sort_key(current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a slicer out/_index.json file.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--kind", choices=("config", "visibility", "build"), required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--upstream-ref")
    parser.add_argument("--upstream-published-at")
    parser.add_argument("--source-repo")
    parser.add_argument("--workflow-run-url")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--platform")
    parser.add_argument("--arch")
    parser.add_argument("--sha256")
    parser.add_argument("--url")
    parser.add_argument("--asset-name")
    args = parser.parse_args()

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    index = load_json(index_path)
    index.setdefault("latest", None)
    versions = index.setdefault("versions", {})
    version = versions.setdefault(args.version, {})

    if args.latest and should_promote_latest(index, args.version):
        index["latest"] = args.version

    metadata = version.setdefault("metadata", {})
    metadata.update(parse_metadata(args.metadata))

    version.update(compact_none({
        "upstream_ref": args.upstream_ref,
        "upstream_published_at": args.upstream_published_at,
        "source_repo": args.source_repo,
    }))

    stamp = compact_none({
        "created_at": args.created_at,
        "workflow_run_url": args.workflow_run_url,
    })

    if args.kind in ("config", "visibility"):
        version[args.kind] = stamp
    else:
        build = compact_none({
            "platform": args.platform,
            "arch": args.arch,
            "sha256": args.sha256,
            "url": args.url,
            "asset_name": args.asset_name,
            "created_at": args.created_at,
            "workflow_run_url": args.workflow_run_url,
        })
        builds = version.setdefault("builds", [])
        builds[:] = [
            existing for existing in builds
            if not (existing.get("platform") == build.get("platform") and existing.get("arch") == build.get("arch"))
        ]
        builds.append(build)
        builds.sort(key=lambda item: (item.get("platform", ""), item.get("arch", "")))

    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
