#!/usr/bin/env python3
"""Run bounded, support-heavy slicer workloads for GCC PGO."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def binary_stl(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) < 84 + count * 50:
        raise ValueError(f"PGO model is not a binary STL: {path}")
    vertices: list[tuple[float, float, float]] = []
    indices: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []
    for offset in range(84, 84 + count * 50, 50):
        values = struct.unpack_from("<12f", data, offset)
        triangle = []
        for vertex in (values[3:6], values[6:9], values[9:12]):
            if vertex not in indices:
                indices[vertex] = len(vertices)
                vertices.append(vertex)
            triangle.append(indices[vertex])
        triangles.append(tuple(triangle))
    return vertices, triangles


def obj_mesh(
    path: Path,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in path.read_text(errors="strict").splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append(tuple(map(float, fields[1:4])))
        elif fields[:1] == ["f"]:
            face = [int(field.split("/", 1)[0]) - 1 for field in fields[1:]]
            triangles.extend(
                (face[0], face[index], face[index + 1])
                for index in range(1, len(face) - 1)
            )
    return vertices, triangles


def model_xml(model: Path) -> bytes:
    vertices, triangles = (
        obj_mesh(model) if model.suffix.lower() == ".obj" else binary_stl(model)
    )
    xs, ys, zs = zip(*vertices)
    dx = 90 - (min(xs) + max(xs)) / 2
    dy = 90 - (min(ys) + max(ys)) / 2
    dz = -min(zs)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        '<metadata name="Application">BambuStudio-01.00.00.00</metadata>',
        "<resources><object id=\"2\" type=\"model\"><mesh><vertices>",
    ]
    lines.extend(
        f'<vertex x="{x + dx:.6g}" y="{y + dy:.6g}" z="{z + dz:.6g}"/>'
        for x, y, z in vertices
    )
    lines.append("</vertices><triangles>")
    lines.extend(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in triangles)
    lines.extend(
        [
            "</triangles></mesh></object></resources>",
            '<build><item objectid="2" printable="1"/></build>',
            "</model>",
        ]
    )
    return "".join(lines).encode()


def center_3mf_model(data: bytes) -> bytes:
    root = ET.fromstring(data)
    vertices = [
        (float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib["z"]))
        for node in root.iter()
        if node.tag.endswith("}vertex")
    ]
    if not vertices:
        raise ValueError("PGO 3MF contains no vertices")
    xs, ys, zs = zip(*vertices)
    transform = (
        f"1 0 0 0 1 0 0 0 1 "
        f"{90 - (min(xs) + max(xs)) / 2:.6g} "
        f"{90 - (min(ys) + max(ys)) / 2:.6g} {-min(zs):.6g}"
    )
    encoded = transform.encode()
    if re.search(rb"<item\b[^>]*\btransform=\"", data):
        return re.sub(
            rb"(<item\b[^>]*\btransform=\")[^\"]*(\")",
            lambda match: match.group(1) + encoded + match.group(2),
            data,
        )
    return re.sub(rb"(<item\b[^>]*)/>", rb'\1 transform="' + encoded + rb'"/>', data)


def make_orca_project(source: Path, output: Path) -> None:
    seeds = list(source.glob("resources/calib/**/pa_pattern.3mf"))
    handy_models = [
        source / "resources/handy_models/3DBenchy.3mf",
        source / "resources/handy_models/Stanford_Bunny.3mf",
    ]
    models = [
        source / "resources/model/3DBenchy.stl",
        source / "resources/creality_models/3DBenchy.stl",
        source / "resources/handy_models/OrcaToleranceTest.stl",
        source / "resources/model/ksr_FDMTest.stl",
        source / "tests/data/frog_legs.obj",
        source / "resources/calib/temperature_tower/temperature_tower.stl",
    ]
    seed = next((path for path in seeds if path.is_file()), None)
    handy_model = next((path for path in handy_models if path.is_file()), None)
    model = next((path for path in models if path.is_file()), None)
    if seed is None or (handy_model is None and model is None):
        raise FileNotFoundError(
            "PGO training needs pa_pattern.3mf and a representative model"
        )

    if handy_model is not None:
        with ZipFile(handy_model) as model_zip:
            replacements = {
                name: model_zip.read(name)
                for name in model_zip.namelist()
                if name in ("3D/3dmodel.model", "Metadata/model_settings.config")
                or name.startswith("3D/Objects/")
                or name == "3D/_rels/3dmodel.model.rels"
            }
        for name in replacements:
            if name.endswith((".model", ".config")):
                replacements[name] = re.sub(
                    rb"BambuStudio-[0-9A-Za-z._-]+",
                    b"BambuStudio-01.00.00.00",
                    replacements[name],
                )
        replacements["3D/3dmodel.model"] = center_3mf_model(
            replacements["3D/3dmodel.model"]
        )
    else:
        replacements = {
            "3D/3dmodel.model": model_xml(model),
            "Metadata/model_settings.config": (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<config><object id="2"><metadata key="name" value="PGO workload"/>'
                b"</object></config>"
            ),
        }
    with ZipFile(seed) as source_zip, ZipFile(output, "w", ZIP_DEFLATED) as target_zip:
        for info in source_zip.infolist():
            if (
                info.filename in replacements
                or info.filename.startswith("3D/Objects/")
                or info.filename == "3D/_rels/3dmodel.model.rels"
            ):
                continue
            target_zip.writestr(info, source_zip.read(info))
        for name, data in replacements.items():
            target_zip.writestr(name, data)


def run(command: list[str], bundle: Path) -> dict[str, float]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(bundle / "bin") + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    print("+", " ".join(command), flush=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+t") as log:
        try:
            subprocess.run(
                command,
                cwd=bundle,
                env=env,
                timeout=180,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            log.seek(0)
            sys.stderr.write(log.read()[-20_000:])
            raise
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_seconds": time.perf_counter() - started,
        "user_seconds": after.ru_utime - before.ru_utime,
        "system_seconds": after.ru_stime - before.ru_stime,
        "max_rss_kib": after.ru_maxrss,
    }


def require_gcode_markers(work: Path, markers: tuple[str, ...]) -> None:
    gcode = next(work.glob("*.gcode"), None)
    if gcode is None:
        raise RuntimeError("PGO training produced no G-code")
    contents = gcode.read_text(errors="replace")
    missing = [marker for marker in markers if marker not in contents]
    if missing:
        raise RuntimeError(f"PGO workload did not exercise: {', '.join(missing)}")


def option_supports(help_text: str, option: str, value: str) -> bool:
    match = re.search(
        rf"(?ms)^ {re.escape(option)}(?:[^\n]*\n)(?:(?!^ --).*\n)*",
        help_text,
    )
    return match is not None and re.search(rf"\b{re.escape(value)}\b", match[0]) is not None


def train_orca(
    executable: Path,
    source: Path,
    bundle: Path,
    work: Path,
    family: str,
    repeat: int,
) -> list[dict[str, float]]:
    project = work / "support-heavy.3mf"
    make_orca_project(source, project)
    # Bambu-family forks resolve "default" + tree(auto) to their organic tree
    # implementation; the explicit enum spelling differs between older forks.
    support_style = "default" if family == "bambu" else "organic"
    metrics = []
    for iteration in range(repeat):
        run_work = work / f"run-{iteration + 1}"
        run_work.mkdir()
        metrics.append(
            run(
                [
                    str(executable),
                    "--slice",
                    "0",
                    "--outputdir",
                    str(run_work),
                    "--enable-support",
                    "--support-type",
                    "tree(auto)",
                    "--support-style",
                    support_style,
                    "--support-interface-top-layers",
                    "3",
                    "--sparse-infill-density",
                    "25%",
                    "--sparse-infill-pattern",
                    "gyroid",
                    "--wall-loops",
                    "3",
                    "--wall-generator",
                    "arachne",
                    "--top-shell-layers",
                    "5",
                    "--enable-overhang-speed",
                    "--detect-overhang-wall",
                    str(project),
                ],
                bundle,
            )
        )
        require_gcode_markers(
            run_work,
            (
                "; enable_support = 1",
                "; sparse_infill_pattern = gyroid",
                "; wall_generator = arachne",
                f"; support_style = {support_style}",
                "; FEATURE: Support",
            ),
        )
    return metrics


def train_prusa(
    executable: Path,
    source: Path,
    bundle: Path,
    work: Path,
    repeat: int,
) -> list[dict[str, float]]:
    model = next(
        path
        for path in (
            source / "resources/shapes/3DBenchy.stl",
            source / "resources/shapes/bunny.stl",
            source / "resources/model/3DBenchy.stl",
        )
        if path.is_file()
    )
    help_text = subprocess.run(
        [str(executable), "--help-fff"],
        cwd=bundle,
        env={**os.environ, "LD_LIBRARY_PATH": str(bundle / "bin")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=True,
    ).stdout
    markers = ["; fill_pattern = gyroid", ";TYPE:Support material"]
    support_style = next(
        (
            value
            for value in ("organic", "snug")
            if option_supports(help_text, "--support-material-style", value)
        ),
        None,
    )
    if support_style:
        markers.append(f"; support_material_style = {support_style}")
    if option_supports(help_text, "--perimeter-generator", "arachne"):
        markers.append("; perimeter_generator = arachne")
    metrics = []
    for iteration in range(repeat):
        run_work = work / f"run-{iteration + 1}"
        run_work.mkdir()
        command = [
            str(executable),
            "--export-gcode",
            "--output",
            str(run_work / "support-heavy.gcode"),
            "--support-material",
            "--support-material-auto",
            "--support-material-interface-layers",
            "3",
            "--fill-density",
            "25%",
            "--fill-pattern",
            "gyroid",
            "--perimeters",
            "3",
            "--top-solid-layers",
            "5",
            "--avoid-crossing-perimeters",
            "--ensure-on-bed",
        ]
        if support_style:
            command.extend(["--support-material-style", support_style])
        if option_supports(help_text, "--perimeter-generator", "arachne"):
            command.extend(["--perimeter-generator", "arachne"])
        command.append(str(model))
        metrics.append(run(command, bundle))
        require_gcode_markers(run_work, tuple(markers))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("bambu", "orca", "prusa"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    source = args.source.resolve()
    bundle = args.bundle.resolve()
    executable = bundle / "bin" / args.executable
    if not executable.is_file():
        raise FileNotFoundError(executable)

    with tempfile.TemporaryDirectory(prefix="slicer-pgo-") as temp:
        work = Path(temp)
        if args.family in ("bambu", "orca"):
            metrics = train_orca(
                executable, source, bundle, work, args.family, args.repeat
            )
        else:
            metrics = train_prusa(
                executable, source, bundle, work, args.repeat
            )
    if args.metrics:
        args.metrics.write_text(
            json.dumps({"family": args.family, "runs": metrics}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
