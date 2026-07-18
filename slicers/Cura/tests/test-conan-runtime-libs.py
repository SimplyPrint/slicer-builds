#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
TOOL = TOOLS / "collect_conan_runtime_libs.py"
SPEC = importlib.util.spec_from_file_location("collect_conan_runtime_libs", TOOL)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def dependency(node_id: str, ref: str, package_folder: Path) -> dict:
    return {
        "context": "host",
        "dependencies": {},
        "id": node_id,
        "package_folder": str(package_folder),
        "package_id": f"package-{node_id}",
        "package_type": "shared-library",
        "recipe": "Cache",
        "ref": ref,
    }


def fake_elf(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.ELF_MAGIC + content)


class ConanRuntimeLibraryCollectionTests(unittest.TestCase):
    def test_collects_payload_and_soname_names_from_distribution_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_package = root / "first-package"
            first_payload = first_package / "lib" / "libalpha.so.4.7"
            fake_elf(first_payload, b"alpha")
            (first_payload.parent / "libalpha.so.4").symlink_to(first_payload.name)
            (first_payload.parent / "libalpha.so").symlink_to("libalpha.so.4")
            (first_payload.parent / "linker-script.so").write_text("INPUT(-lalpha)\n")

            transitive_package = root / "transitive-package"
            plugin = transitive_package / "plugins" / "filter.so"
            fake_elf(plugin, b"plugin")

            build_package = root / "build-package"
            build_payload = build_package / "lib" / "excluded.so.1"
            fake_elf(build_payload, b"build-only")

            nodes = {
                "0": {
                    "context": "host",
                    "dependencies": {
                        "1": {"libs": True},
                        "3": {"build": True, "run": True},
                    },
                    "recipe": "Consumer",
                },
                "1": dependency("1", "alpha/1.0", first_package),
                "2": dependency("2", "plugin/1.0", transitive_package),
                "3": dependency("3", "builder/1.0", build_package),
            }
            nodes["1"]["dependencies"] = {"2": {"run": True}}
            nodes["3"]["context"] = "build"
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))

            def soname(path: Path) -> str | None:
                if path.name == "libalpha.so.4.7":
                    return "libalpha.so.4"
                return None

            first = MODULE.collect(graph, root / "bundle", soname_reader=soname)
            repeated = MODULE.collect(graph, root / "bundle", soname_reader=soname)

            self.assertEqual(first, repeated)
            self.assertEqual(
                [item["name"] for item in first["libraries"]],
                ["filter.so", "libalpha.so.4", "libalpha.so.4.7"],
            )
            self.assertEqual(
                (root / "bundle" / "libalpha.so.4").read_bytes(),
                first_payload.read_bytes(),
            )
            self.assertFalse((root / "bundle" / "excluded.so.1").exists())
            self.assertFalse((root / "bundle" / "linker-script.so").exists())
            for path in (root / "bundle").iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o755)

    def test_rejects_conflicting_names_between_conan_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = [root / "first", root / "second"]
            fake_elf(packages[0] / "lib" / "libcollision.so.1.1", b"first")
            fake_elf(packages[1] / "lib" / "libcollision.so.1.2", b"second")
            nodes = {
                "0": {
                    "dependencies": {"1": {"libs": True}, "2": {"run": True}},
                    "recipe": "Consumer",
                },
                "1": dependency("1", "first/1.0", packages[0]),
                "2": dependency("2", "second/1.0", packages[1]),
            }
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))

            with self.assertRaisesRegex(
                MODULE.RuntimeLibraryCollectionError,
                r"Conflicting Conan runtime library libcollision\.so\.1",
            ):
                MODULE.collect(
                    graph,
                    root / "bundle",
                    soname_reader=lambda _path: "libcollision.so.1",
                )

    def test_rejects_conflict_with_direct_linker_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            payload = package / "lib" / "libruntime.so.3.2"
            fake_elf(payload, b"conan")
            nodes = {
                "0": {"dependencies": {"1": {"run": True}}, "recipe": "Consumer"},
                "1": dependency("1", "runtime/1.0", package),
            }
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))
            output = root / "bundle"
            output.mkdir()
            fake_elf(output / "libruntime.so.3", b"different-ldd-payload")

            with self.assertRaisesRegex(
                MODULE.RuntimeLibraryCollectionError,
                r"Conflicting existing runtime library libruntime\.so\.3",
            ):
                MODULE.collect(
                    graph,
                    output,
                    soname_reader=lambda _path: "libruntime.so.3",
                )


if __name__ == "__main__":
    unittest.main()
