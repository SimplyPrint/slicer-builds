#!/usr/bin/env python3
"""Run one production-shaped slice through the cloud-slicer engine adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import struct
import sys
import time
import zlib
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_one(
    records: list[dict[str, Any]],
    key: str,
    value: str,
    *,
    machine_name: str | None = None,
    machine_variant: str | None = None,
) -> dict[str, Any]:
    matches = [record for record in records if record.get(key) == value]
    if key == "native_id":
        exact_matches = [
            record
            for record in records
            if isinstance(record.get("context"), dict)
            and record["context"].get("native_id") == value
        ]
        # Resolved Cura material profiles have a deterministic native ID suffix
        # for each machine/nozzle. Selectors intentionally use the stable
        # upstream source ID, then narrow it to the requested machine variant.
        matches = exact_matches or [
            record
            for record in records
            if isinstance(record.get("context"), dict)
            and record["context"].get("source_native_id") == value
        ]

    if machine_name is not None and machine_variant is not None:
        compatible_matches = []
        records_with_compatibility = []
        for record in matches:
            compatible_printers = record.get("compatible_printers")
            if not isinstance(compatible_printers, dict):
                continue
            records_with_compatibility.append(record)
            variants = compatible_printers.get(machine_name)
            if isinstance(variants, list) and machine_variant in variants:
                compatible_matches.append(record)
        if records_with_compatibility:
            matches = compatible_matches

    if len(matches) != 1:
        compatibility = (
            f" compatible with {machine_name!r} variant {machine_variant!r}"
            if machine_name is not None and machine_variant is not None
            else ""
        )
        raise AssertionError(
            f"Expected one profile with {key}={value!r}{compatibility}, "
            f"found {len(matches)}"
        )
    return matches[0]


def profile_data(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    if not isinstance(data, dict):
        name = record.get("name", "<unnamed>")
        raise AssertionError(f"Profile {name!r} has no data object")
    return data


def select_machine(
    records: list[dict[str, Any]], machine_name: str, variant: str
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if isinstance(record.get("machine_model"), dict)
        and record["machine_model"].get("name") == machine_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected machine model {machine_name!r}, found {len(matches)}"
        )
    variants = matches[0].get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise AssertionError(f"Machine {machine_name!r} has no variant {variant!r}")
    return variants[variant]


def runtime_wrapper(record: dict[str, Any]) -> dict[str, Any]:
    required = ("data", "context", "setting_scopes")
    result = {key: record[key] for key in required if key in record}
    if set(result) != set(required):
        raise AssertionError(
            f"Cura profile is missing runtime wrapper keys: {record.get('name')}"
        )
    result["transport"] = "envelope.v1"
    return result


def version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return (0,)


def direct_profile(path: Path, settings_id: str) -> dict[str, Any]:
    record = load_json(path)
    settings = record.get("settings")
    if not isinstance(settings, dict):
        raise AssertionError(f"Direct profile has no settings object: {path}")

    resolved: dict[str, Any] = {}
    for key, versions in settings.items():
        if not isinstance(versions, dict) or not versions:
            continue
        selected_version = max(versions, key=version_tuple)
        resolved[key] = versions[selected_version]
    resolved[settings_id] = record["name"]
    return resolved


def sibling_profile_names(
    directory: Path, cache: dict[Path, dict[str, Path]]
) -> dict[str, Path]:
    directory = directory.resolve()
    if directory in cache:
        return cache[directory]

    names: dict[str, Path] = {}
    for candidate in sorted(directory.glob("*.json")):
        record = load_json(candidate)
        if not isinstance(record, dict):
            raise AssertionError(f"Profile is not a JSON object: {candidate}")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in names:
            raise AssertionError(
                f"Duplicate sibling profile name {name!r}: "
                f"{names[name].name}, {candidate.name}"
            )
        names[name] = candidate.resolve()
    cache[directory] = names
    return names


def inherited_profile(
    path: Path,
    stack: tuple[Path, ...] = (),
    name_indexes: dict[Path, dict[str, Path]] | None = None,
) -> dict[str, Any]:
    if name_indexes is None:
        name_indexes = {}
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(item.name for item in (*stack, path))
        raise AssertionError(f"Profile inheritance cycle: {chain}")
    if not path.is_file():
        raise AssertionError(f"Inherited profile does not exist: {path}")

    record = load_json(path)
    parent_name = record.get("inherits")
    if not parent_name:
        return record
    if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
        raise AssertionError(f"Invalid inherits value in {path}: {parent_name!r}")

    parent = path.parent / f"{parent_name}.json"
    if not parent.is_file():
        parent = sibling_profile_names(path.parent, name_indexes).get(parent_name)
        if parent is None:
            raise AssertionError(
                f"Inherited profile {parent_name!r} does not exist beside {path}"
            )
    resolved = inherited_profile(parent, (*stack, path), name_indexes)
    resolved.update(record)
    resolved.pop("inherits", None)
    return resolved


def load_profiles(
    spec: dict[str, Any], backend_root: Path, profiles_root: Path, bundle: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    test = spec["test"]
    source = test["profile_source"]

    if source == "backend-example":
        example = backend_root / "example"
        return (
            load_json(example / "machine.json"),
            load_json(example / "filament.json"),
            load_json(example / "process.json"),
        )

    if source == "profiles-direct":
        root = profiles_root / "profiles" / test["profile_engine"]
        return (
            direct_profile(root / test["machine_profile"], "printer_settings_id"),
            direct_profile(root / test["filament_profile"], "filament_settings_id"),
            direct_profile(root / test["process_profile"], "print_settings_id"),
        )

    if source == "bundle-resources":
        root = bundle / "resources" / test["profile_root"]
        return (
            inherited_profile(root / test["machine_profile"]),
            inherited_profile(root / test["filament_profile"]),
            inherited_profile(root / test["process_profile"]),
        )

    if source != "profiles-db":
        raise AssertionError(f"Unsupported profile source: {source}")

    root = (
        profiles_root
        / "out"
        / "models"
        / str(test["model_id"])
        / test["profile_engine"]
    )
    machines = load_json(root / "machine_profiles.json")
    filaments = load_json(root / "filament_profiles.json")
    processes = load_json(root / "print_profiles.json")

    machine_record = select_machine(
        machines, test["machine_name"], test["machine_variant"]
    )
    process_key = "native_id" if "process_native_id" in test else "name"
    process_value = test.get("process_native_id", test.get("process_name"))
    filament_key = "native_id" if "filament_native_id" in test else "name"
    filament_value = test.get("filament_native_id", test.get("filament_name"))
    selector_context = {
        "machine_name": test["machine_name"],
        "machine_variant": test["machine_variant"],
    }
    process_record = require_one(
        processes, process_key, process_value, **selector_context
    )
    filament_record = require_one(
        filaments, filament_key, filament_value, **selector_context
    )

    if test["contract"] == "cura":
        return (
            runtime_wrapper(machine_record),
            runtime_wrapper(filament_record),
            runtime_wrapper(process_record),
        )
    return (
        profile_data(machine_record),
        profile_data(filament_record),
        profile_data(process_record),
    )


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    return above if above_distance <= upper_left_distance else upper_left


def validate_png(
    data: bytes, member: str, *, require_visual_detail: bool = True
) -> dict[str, int]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"Thumbnail is not a PNG: {member}")

    position = 8
    idat = bytearray()
    width = height = bit_depth = color_type = 0
    saw_header = saw_end = False
    while position + 12 <= len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        if position + 12 + length > len(data):
            raise AssertionError(f"Truncated PNG chunk in thumbnail: {member}")
        payload = data[position + 8 : position + 8 + length]
        expected_crc = struct.unpack(
            ">I", data[position + 8 + length : position + 12 + length]
        )[0]
        actual_crc = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise AssertionError(f"Invalid PNG checksum in thumbnail: {member}")
        position += 12 + length
        if chunk_type == b"IHDR":
            if saw_header or len(payload) != 13:
                raise AssertionError(f"Malformed PNG header in thumbnail: {member}")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise AssertionError(f"Unsupported PNG encoding in thumbnail: {member}")
            saw_header = True
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            saw_end = True
            break

    if not saw_header or not saw_end or not idat:
        raise AssertionError(f"Incomplete PNG thumbnail: {member}")
    if not (64 <= width <= 2048 and 64 <= height <= 2048):
        raise AssertionError(
            f"Unexpected thumbnail dimensions {width}x{height}: {member}"
        )
    if bit_depth != 8 or color_type not in (2, 6):
        raise AssertionError(
            f"Unsupported thumbnail PNG format depth={bit_depth} type={color_type}"
        )

    bytes_per_pixel = 4 if color_type == 6 else 3
    stride = width * bytes_per_pixel
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise AssertionError(f"Cannot decompress thumbnail PNG: {member}") from error
    if len(raw) != height * (stride + 1):
        raise AssertionError(f"Malformed thumbnail pixel data: {member}")

    previous = bytearray(stride)
    pixels = bytearray()
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        source = raw[offset + 1 : offset + stride + 1]
        offset += stride + 1
        row = bytearray(stride)
        for index, value in enumerate(source):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = (
                previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            )
            if filter_type == 0:
                filtered = value
            elif filter_type == 1:
                filtered = value + left
            elif filter_type == 2:
                filtered = value + above
            elif filter_type == 3:
                filtered = value + ((left + above) // 2)
            elif filter_type == 4:
                filtered = value + paeth(left, above, upper_left)
            else:
                raise AssertionError(f"Unsupported PNG filter {filter_type}: {member}")
            row[index] = filtered & 0xFF
        pixels.extend(row)
        previous = row

    if color_type == 6:
        visible_pixels = [
            bytes(pixels[index : index + bytes_per_pixel])
            for index in range(0, len(pixels), bytes_per_pixel)
            if pixels[index + 3] >= 16
        ]
        minimum_visible = max(64, (width * height) // 100)
        if require_visual_detail and len(visible_pixels) < minimum_visible:
            raise AssertionError(
                f"Thumbnail has too few visible pixels "
                f"({len(visible_pixels)} < {minimum_visible}): {member}"
            )
    else:
        visible_pixels = [
            bytes(pixels[index : index + bytes_per_pixel])
            for index in range(0, len(pixels), bytes_per_pixel)
        ]
    pixel_values = set(visible_pixels)
    if require_visual_detail and len(pixel_values) < 8:
        raise AssertionError(f"Thumbnail is blank or nearly uniform: {member}")
    return {"width": width, "height": height, "bytes": len(data)}


def resolve_archive_member(names: set[str], value: str) -> str:
    candidates = (value, value.lstrip("/"), f"Metadata/{value.lstrip('/')}")
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise AssertionError(f"3MF references missing thumbnail {value!r}")


def validate_thumbnails(path: Path) -> dict[str, dict[str, int]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        config_name = "Metadata/model_settings.config"
        if config_name not in names:
            raise AssertionError("Sliced 3MF has no Metadata/model_settings.config")
        root = ElementTree.fromstring(archive.read(config_name))
        references: dict[str, str] = {}
        for metadata in root.iter():
            if metadata.tag.rsplit("}", 1)[-1] != "metadata":
                continue
            key = metadata.attrib.get("key", "")
            value = metadata.attrib.get("value", "")
            if (
                key
                in {
                    "thumbnail_file",
                    "thumbnail_no_light_file",
                    "no_light_thumbnail_file",
                    "top_file",
                    "pick_file",
                }
                and value
            ):
                references[key] = resolve_archive_member(names, value)
        if "thumbnail_file" not in references:
            raise AssertionError("Sliced 3MF does not reference a plate thumbnail")
        return {
            key: validate_png(
                archive.read(member),
                member,
                require_visual_detail=key == "thumbnail_file",
            )
            for key, member in references.items()
        }


def validate_gcode(path: Path) -> int:
    size = path.stat().st_size
    if size < 10_000:
        raise AssertionError(f"G-code is unexpectedly small: {size} bytes")
    with path.open("rb") as source:
        sample = source.read(min(size, 2_000_000))
    if b"\nG0 " not in sample and b"\nG1 " not in sample:
        raise AssertionError("G-code contains no movement commands")
    return size


def validate_slicer_return_code(return_code: int | None) -> int:
    if return_code != 0:
        raise AssertionError(
            f"Slicer exited with status {return_code!r} despite producing artifacts"
        )
    return return_code


def validate_compatibility_profile_identity(
    work: Path, expected_machine: dict[str, Any]
) -> dict[str, str]:
    expected_name = expected_machine.get("name")
    if not isinstance(expected_name, str) or not expected_name:
        raise AssertionError("Compatibility machine profile has no name")

    machine_files = sorted(work.glob("machine-*.json"))
    process_files = sorted(work.glob("process-*.json"))
    if len(machine_files) != 1 or len(process_files) != 1:
        raise AssertionError(
            "Compatibility adapter did not leave exactly one generated "
            "machine and process profile"
        )
    generated_machine = load_json(machine_files[0])
    generated_process = load_json(process_files[0])
    generated_name = generated_machine.get("name")
    if generated_name != expected_name:
        raise AssertionError(
            f"Compatibility adapter changed machine identity "
            f"{expected_name!r} to {generated_name!r}"
        )
    compatible_printers = generated_process.get("compatible_printers", [])
    if (
        not isinstance(compatible_printers, list)
        or expected_name not in compatible_printers
    ):
        raise AssertionError(
            f"Generated process profile is not compatible with {expected_name!r}"
        )
    return {
        "expected_machine_name": expected_name,
        "generated_machine_name": generated_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    spec = load_json(args.spec)
    sys.path.insert(0, str(args.backend_root))

    from src.slicers.bambu import BambuStudioEngine
    from src.slicers.creality import CrealityPrintEngine
    from src.slicers.cura import CuraEngine
    from src.slicers.elegoo import ElegooSlicerEngine
    from src.slicers.orca import OrcaSlicerEngine
    from src.slicers.prusa import PrusaSlicerEngine
    from src.slicers.schemas import SlicerSettings
    from src.slicers.superslicer import SuperSlicerEngine

    engines = {
        "BambuStudio": BambuStudioEngine,
        "CrealityPrint": CrealityPrintEngine,
        "Cura": CuraEngine,
        "ElegooSlicer": ElegooSlicerEngine,
        "OrcaSlicer": OrcaSlicerEngine,
        "PrusaSlicer": PrusaSlicerEngine,
        "SuperSlicer": SuperSlicerEngine,
    }
    engine_class = engines[spec["backend_engine"]]
    executable = args.bundle / "bin" / spec["executable"]
    if not executable.is_file():
        raise AssertionError(f"Bundle executable is missing: {executable}")

    # Several adapters use self.version while producing settings and argv. Give
    # them an exact installed-layout view so they cannot fall back to a stale
    # slicer version already present in the runtime image.
    installed_root = args.work / "installed"
    expected_executable = (
        installed_root
        / engine_class.SLICER_NAME
        / spec["backend_version"]
        / "bin"
        / engine_class.EXECUTABLE_NAME
    )
    expected_executable.parent.mkdir(parents=True, exist_ok=True)
    expected_executable.symlink_to(executable)
    engine_class.BASE_PATH = installed_root
    os.environ[engine_class.ENV_VAR] = str(executable)

    machine, filament, process = load_profiles(
        spec, args.backend_root, args.profiles_root, args.bundle
    )
    settings = SlicerSettings(
        machine=machine,
        filament=[filament],
        process=process,
        # The production request path supplies positioned geometry and leaves
        # the adapter's optional re-arrangement disabled by default.
    )

    args.work.mkdir(parents=True, exist_ok=True)
    input_path = args.work / args.input.name
    shutil.copy2(args.input, input_path)
    engine = engine_class(
        version=spec.get("backend_version"),
        working_dir=args.work,
        progress_cb=None,
    )
    if spec["capabilities"]["thumbnail"]:
        display = os.environ.get("DISPLAY", "")
        display_number = display.removeprefix(":").split(".", 1)[0]
        if display_number.isdigit():
            display_socket = Path("/tmp/.X11-unix") / f"X{display_number}"
            deadline = time.monotonic() + 5
            while not display_socket.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not display_socket.exists():
                raise AssertionError(
                    f"X11 display socket did not become ready: {display_socket}"
                )
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_at = time.perf_counter()
    try:
        result = engine.slice(settings, input_path)
    except Exception:
        print(
            f"slicer return code: {engine.return_code}\n"
            + "\n".join(engine.output),
            file=sys.stderr,
        )
        raise
    wall_seconds = time.perf_counter() - started_at
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    test_spec = spec["test"]
    profile_selectors = {
        key: value
        for key, value in test_spec.items()
        if key not in {"contract", "profile_source", "backend_version"}
    }

    report: dict[str, Any] = {
        "slicer": spec["name"],
        "backend_adapter": spec["backend_engine"],
        "backend_supported": spec["backend_supported"],
        "backend_version": engine.version,
        "runtime_image": spec.get("runtime_image"),
        "runtime_image_identity": spec.get("runtime_image_identity"),
        "build_provenance": spec.get("build_provenance"),
        "backend_source_identity": spec.get("backend_source_identity"),
        "contract": spec["test"]["contract"],
        "return_code": engine.return_code,
        "arrange": settings.arrange,
        "executable_sha256": file_sha256(executable),
        "input_sha256": file_sha256(args.input),
        "profile_sha256": {
            "machine": json_sha256(machine),
            "filament": json_sha256(filament),
            "process": json_sha256(process),
        },
        "profile_provenance": {
            "coverage": "native" if spec["backend_supported"] else "compatibility",
            "source": test_spec["profile_source"],
            "source_identity": spec.get("profile_source_identity"),
            "selectors": profile_selectors,
        },
        "performance": {
            "wall_seconds": round(wall_seconds, 6),
            "child_user_cpu_seconds": round(
                usage_after.ru_utime - usage_before.ru_utime, 6
            ),
            "child_system_cpu_seconds": round(
                usage_after.ru_stime - usage_before.ru_stime, 6
            ),
            # The smoke runtime is Linux, where ru_maxrss is reported in KiB.
            "peak_child_rss_kib": usage_after.ru_maxrss,
        },
        "result_file": str(result.result_file),
        "gcode_file": str(result.gcode_file),
        "warnings": result.warnings or [],
    }
    try:
        report["return_code"] = validate_slicer_return_code(engine.return_code)
        report["gcode_bytes"] = validate_gcode(result.gcode_file)
        if test_spec["contract"] == "bambu" and not spec["backend_supported"]:
            report["compatibility_profile_identity"] = (
                validate_compatibility_profile_identity(args.work, machine)
            )
        if spec["capabilities"]["thumbnail"]:
            report["thumbnails"] = validate_thumbnails(result.result_file)
    except AssertionError as error:
        report["status"] = "failed"
        report["validation_error"] = str(error)
        failure_path = args.work / "smoke-failure.json"
        failure_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    report["status"] = "passed"

    report_path = args.work / "smoke-result.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
