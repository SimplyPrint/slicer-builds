#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TOOL = Path(__file__).parents[1] / "tools" / "collect_conan_licenses.py"
SPEC = importlib.util.spec_from_file_location("collect_conan_licenses", TOOL)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def dependency(node_id: str, ref: str, package_type: str, **paths: str) -> dict:
    return {
        "context": "host",
        "dependencies": {},
        "id": node_id,
        "license": "MIT",
        "package_folder": paths.get("package_folder"),
        "package_id": f"package-{node_id}",
        "package_type": package_type,
        "recipe": "Cache",
        "recipe_folder": paths.get("recipe_folder"),
        "ref": ref,
    }


class ConanLicenseCollectionTests(unittest.TestCase):
    def test_collects_host_code_and_filters_build_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packaged = root / "packaged"
            (packaged / "licenses").mkdir(parents=True)
            (packaged / "licenses" / "LICENSE").write_text("package license\n")
            packaged_recipe = root / "packaged-recipe"
            packaged_recipe.mkdir()
            (packaged_recipe / "NOTICE").write_text("must not replace package license\n")

            fallback_package = root / "fallback-package"
            fallback_package.mkdir()
            fallback_recipe = root / "fallback-recipe"
            (fallback_recipe / "legal").mkdir(parents=True)
            (fallback_recipe / "legal" / "LICENSE.md").write_text("recipe license\n")
            (fallback_recipe / "conanfile.py").write_text("# not a notice\n")

            build_package = root / "build-package"
            (build_package / "licenses").mkdir(parents=True)
            (build_package / "licenses" / "LICENSE").write_text("build only\n")

            nodes = {
                "0": {
                    "context": "host",
                    "dependencies": {
                        "1": {"headers": True},
                        "2": {"libs": True},
                        "3": {"build": True, "run": True},
                        "4": {"run": True},
                        "5": {"run": True, "test": True},
                    },
                    "id": "0",
                    "recipe": "Consumer",
                    "ref": "consumer/1",
                },
                "1": dependency(
                    "1",
                    "headers/1.0@owner/stable#revision",
                    "header-library",
                    package_folder=str(packaged),
                    recipe_folder=str(packaged_recipe),
                ),
                "2": dependency(
                    "2",
                    "static/1.0",
                    "static-library",
                    package_folder=str(fallback_package),
                    recipe_folder=str(fallback_recipe),
                ),
                "3": dependency(
                    "3", "builder/1.0", "application", package_folder=str(build_package)
                ),
                "4": dependency(
                    "4", "cmake-helpers/1.0", "build-scripts", package_folder=str(build_package)
                ),
                "5": dependency(
                    "5", "test-runtime/1.0", "shared-library", package_folder=str(build_package)
                ),
            }
            nodes["3"]["context"] = "build"
            nodes["3"]["dependencies"] = {"6": {"libs": True}}
            nodes["4"]["dependencies"] = {"6": {"libs": True}}
            nodes["6"] = dependency(
                "6", "build-only-transitive/1.0", "static-library", package_folder=str(build_package)
            )
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))

            first = MODULE.collect(graph, root / "first")
            second = MODULE.collect(graph, root / "second")
            repeated = MODULE.collect(graph, root / "first")

            self.assertEqual(first, second)
            self.assertEqual(first, repeated)
            self.assertEqual(first["dependency_count"], 2)
            self.assertEqual(first["file_count"], 2)
            self.assertEqual(first["generator"], MODULE.INVENTORY_GENERATOR)
            self.assertEqual(
                [item["source"] for item in first["dependencies"]],
                ["package_licenses", "recipe_export"],
            )
            self.assertRegex(first["dependencies"][0]["destination"], r"^[A-Za-z0-9._+-]+--[0-9a-f]{12}$")
            self.assertEqual(
                (root / "first" / first["dependencies"][0]["files"][0]["path"]).read_text(),
                "package license\n",
            )
            self.assertEqual(
                (root / "first" / first["dependencies"][1]["files"][0]["path"]).read_text(),
                "recipe license\n",
            )
            for path in (root / "first").rglob("*"):
                self.assertEqual(int(path.stat().st_mtime), MODULE.ZIP_EPOCH)

    def test_fails_when_a_distribution_dependency_has_no_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            recipe = root / "recipe"
            package.mkdir()
            recipe.mkdir()
            nodes = {
                "0": {
                    "dependencies": {"1": {"libs": True}},
                    "recipe": "Consumer",
                },
                "1": dependency(
                    "1",
                    "missing-notice/1.0",
                    "static-library",
                    package_folder=str(package),
                    recipe_folder=str(recipe),
                ),
            }
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))

            with self.assertRaisesRegex(MODULE.LicenseCollectionError, "missing-notice/1.0"):
                MODULE.collect(graph, root / "output")

    def test_refuses_to_delete_an_unrecognized_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            (package / "licenses").mkdir(parents=True)
            (package / "licenses" / "LICENSE").write_text("license\n")
            nodes = {
                "0": {"dependencies": {"1": {"headers": True}}, "recipe": "Consumer"},
                "1": dependency(
                    "1", "headers/1.0", "header-library", package_folder=str(package)
                ),
            }
            graph = root / "graph.json"
            graph.write_text(json.dumps({"graph": {"nodes": nodes}}))
            output = root / "output"
            output.mkdir()
            protected = output / "keep.txt"
            protected.write_text("keep\n")

            with self.assertRaisesRegex(MODULE.LicenseCollectionError, "Refusing to replace"):
                MODULE.collect(graph, output)
            self.assertEqual(protected.read_text(), "keep\n")


if __name__ == "__main__":
    unittest.main()
