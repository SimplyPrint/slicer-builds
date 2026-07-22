from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_stage_module():
    path = ROOT / "tools" / "stage_bundle.py"
    spec = importlib.util.spec_from_file_location("stage_bundle_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StageBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_stage_module()

    def args(
        self,
        executable: Path,
        output: Path,
        library_root: Path,
        resources: list[Path] | None = None,
        resource_include: list[str] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            executable=[executable],
            name="fixture",
            arch={"x86_64": "x86-64", "aarch64": "arm64"}[platform.machine()],
            output=output,
            resources=resources or [],
            resource_include=resource_include or [],
            library_root=[library_root],
            strip=False,
            strip_program="strip",
            json=False,
        )

    def system_executable(self, root: Path) -> Path:
        compiler = os.environ.get("CC") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable for hermetic ELF fixture")
        source = root / "system-fixture.c"
        executable = root / "system-fixture"
        source.write_text("int main(void) { return 0; }\n")
        subprocess.run([compiler, source, "-o", executable], check=True)
        return executable

    def test_missing_executable_fails_without_replacing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("preserved")
            libraries = root / "libraries"
            libraries.mkdir()
            with self.assertRaises(self.module.BundleError):
                self.module.stage_bundle(self.args(root / "missing", output, libraries))
            self.assertEqual(marker.read_text(), "preserved")

    def test_symlinked_output_never_deletes_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            marker = target / "keep"
            marker.write_text("preserved")
            output = root / "bundle"
            output.symlink_to(target, target_is_directory=True)
            libraries = root / "libraries"
            libraries.mkdir()

            with self.assertRaisesRegex(
                self.module.BundleError, "symlinked output path"
            ):
                self.module.stage_bundle(
                    self.args(Path(sys.executable), output, libraries)
                )

            self.assertTrue(output.is_symlink())
            self.assertEqual(marker.read_text(), "preserved")

    def test_ambiguous_fallback_library_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "libfixture.so.1").write_bytes(b"first")
            (second / "libfixture.so.1").write_bytes(b"second")

            with self.assertRaisesRegex(
                self.module.BundleError, "ambiguous shared-library candidates"
            ):
                self.module._resolve_missing_search_dirs(
                    [first, second], ["libfixture.so.1"]
                )

    def test_identical_fallback_library_copies_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "libfixture.so.1").write_bytes(b"identical")
            (second / "libfixture.so.1").write_bytes(b"identical")

            self.assertEqual(
                self.module._resolve_missing_search_dirs(
                    [second, first], ["libfixture.so.1"]
                ),
                [first],
            )

    def test_dependency_closure_resolves_sonames_revealed_in_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fixture"
            executable.write_bytes(b"executable")
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "libfirst.so.1"
            second = second_dir / "libsecond.so.1"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            dependency_passes = [
                self.module.MissingDependencies(executable, [first.name]),
                self.module.MissingDependencies(executable, [second.name]),
                [
                    self.module.Dependency(first.name, first),
                    self.module.Dependency(second.name, second),
                ],
                [],
                [],
            ]
            with (
                mock.patch.object(
                    self.module,
                    "_hermetic_dependencies",
                    side_effect=dependency_passes,
                ) as dependencies,
                mock.patch.object(
                    self.module,
                    "_resolve_missing_search_dirs",
                    side_effect=[[first_dir], [second_dir]],
                ),
            ):
                closure = self.module._dependency_closure(
                    executable, [first_dir, second_dir]
                )

            self.assertEqual(
                closure,
                {first: {first.name}, second: {second.name}},
            )
            self.assertEqual(dependencies.call_args_list[0].args[1], [])
            self.assertEqual(dependencies.call_args_list[1].args[1], [first_dir])
            self.assertEqual(
                dependencies.call_args_list[2].args[1],
                [first_dir, second_dir],
            )

    def test_elf_architecture_must_match_declared_bundle_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "foreign-arch"
            header = bytearray(20)
            header[:4] = b"\x7fELF"
            header[4] = 2
            header[5] = 1
            foreign_arch = "arm64" if platform.machine() == "x86_64" else "x86-64"
            machine = 183 if foreign_arch == "arm64" else 62
            header[18:20] = machine.to_bytes(2, "little")
            executable.write_bytes(header)

            expected_arch = "x86-64" if foreign_arch == "arm64" else "arm64"
            with self.assertRaisesRegex(
                self.module.BundleError,
                f"architecture is {foreign_arch}, expected {expected_arch}",
            ):
                self.module._validate_elf(
                    executable, "fixture executable", expected_arch
                )

    @unittest.skipUnless(
        os.environ.get("CC") or shutil.which("cc"), "C compiler unavailable"
    )
    def test_stages_private_dependency_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            (root / "helper.c").write_text("int helper(void) { return 7; }\n")
            (root / "main.c").write_text(
                "int helper(void); int main(void) { return helper() != 7; }\n"
            )
            compiler = os.environ.get("CC", "cc")
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    root / "helper.c",
                    "-Wl,-soname,libhelper.so.1",
                    "-o",
                    libraries / "libhelper.so.1",
                ],
                check=True,
            )
            (libraries / "libhelper.so").symlink_to("libhelper.so.1")
            executable = root / "fixture-input"
            subprocess.run(
                [
                    compiler,
                    root / "main.c",
                    f"-L{libraries}",
                    "-lhelper",
                    f"-Wl,-rpath,{libraries}",
                    "-o",
                    executable,
                ],
                check=True,
            )
            resources = root / "resources"
            resources.mkdir()
            (resources / "profile.json").write_text("{}\n")
            output = root / "bundle"
            result = self.module.stage_bundle(
                self.args(executable, output, libraries, [resources])
            )
            _executable, library_count, _size = result
            environment = os.environ.copy()
            environment["LD_LIBRARY_PATH"] = str(output / "bin")
            subprocess.run([output / "bin" / "fixture"], check=True, env=environment)
            self.assertEqual(library_count, 1)
            self.assertEqual(result.resources.mode, "copy-all")
            self.assertEqual(result.resources.source_files, 1)
            self.assertEqual(result.resources.staged_files, 1)
            self.assertEqual(result.resources.staged_bytes, 3)
            self.assertEqual(result.resources.omitted_bytes, 0)
            self.assertTrue((output / "bin" / "libhelper.so.1").is_file())
            self.assertEqual(
                (output / "resources" / "profile.json").read_text(), "{}\n"
            )

    @unittest.skipUnless(
        os.environ.get("CC") or shutil.which("cc"), "C compiler unavailable"
    )
    def test_staged_validation_ignores_original_runpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            (root / "helper.c").write_text("int helper(void) { return 7; }\n")
            (root / "main.c").write_text(
                "int helper(void); int main(void) { return helper() != 7; }\n"
            )
            compiler = os.environ.get("CC", "cc")
            private_library = libraries / "libhelper.so.1"
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    root / "helper.c",
                    "-Wl,-soname,libhelper.so.1",
                    "-o",
                    private_library,
                ],
                check=True,
            )
            (libraries / "libhelper.so").symlink_to(private_library.name)
            source_executable = root / "fixture-input"
            subprocess.run(
                [
                    compiler,
                    root / "main.c",
                    f"-L{libraries}",
                    "-lhelper",
                    f"-Wl,-rpath,{libraries}",
                    "-o",
                    source_executable,
                ],
                check=True,
            )

            bin_dir = root / "bundle" / "bin"
            bin_dir.mkdir(parents=True)
            staged_executable = bin_dir / "fixture"
            shutil.copy2(source_executable, staged_executable)
            architecture = {"x86_64": "x86-64", "aarch64": "arm64"}[platform.machine()]

            # Ordinary ldd succeeds through the original absolute RUNPATH,
            # which is exactly the false-positive deployment check this guards.
            self.assertTrue(self.module._ldd(staged_executable, []))
            with self.assertRaisesRegex(
                self.module.MissingDependencies, "libhelper.so.1"
            ):
                self.module._validate_staged(
                    staged_executable, bin_dir, architecture, [libraries]
                )

            shutil.copy2(private_library, bin_dir / private_library.name)
            self.module._validate_staged(
                staged_executable, bin_dir, architecture, [libraries]
            )

    @unittest.skipUnless(
        os.environ.get("CC") or shutil.which("cc"), "C compiler unavailable"
    )
    def test_dependency_discovery_ignores_original_private_runpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            allowed = root / "allowed"
            private.mkdir()
            allowed.mkdir()
            (root / "helper.c").write_text("int helper(void) { return 7; }\n")
            (root / "main.c").write_text(
                "int helper(void); int main(void) { return helper() != 7; }\n"
            )
            compiler = os.environ.get("CC", "cc")
            for directory in (private, allowed):
                subprocess.run(
                    [
                        compiler,
                        "-shared",
                        "-fPIC",
                        root / "helper.c",
                        "-Wl,-soname,libhelper.so.1",
                        "-o",
                        directory / "libhelper.so.1",
                    ],
                    check=True,
                )
                (directory / "libhelper.so").symlink_to("libhelper.so.1")
            executable = root / "fixture-input"
            subprocess.run(
                [
                    compiler,
                    root / "main.c",
                    f"-L{private}",
                    "-lhelper",
                    f"-Wl,-rpath,{private}",
                    "-o",
                    executable,
                ],
                check=True,
            )

            output = root / "bundle"
            result = self.module.stage_bundle(
                self.args(executable, output, allowed)
            )

            self.assertEqual(result.library_count, 1)
            self.assertTrue((output / "bin" / "libhelper.so.1").is_file())

    @unittest.skipUnless(
        os.environ.get("CC") or shutil.which("cc"), "C compiler unavailable"
    )
    def test_undeclared_private_runpath_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "omitted-private-root"
            private.mkdir()
            declared_root = root / "declared-root"
            declared_root.mkdir()
            (root / "helper.c").write_text("int helper(void) { return 7; }\n")
            (root / "main.c").write_text(
                "int helper(void); int main(void) { return helper() != 7; }\n"
            )
            compiler = os.environ.get("CC", "cc")
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    root / "helper.c",
                    "-Wl,-soname,libhelper.so.1",
                    "-o",
                    private / "libhelper.so.1",
                ],
                check=True,
            )
            (private / "libhelper.so").symlink_to("libhelper.so.1")
            executable = root / "fixture-input"
            subprocess.run(
                [
                    compiler,
                    root / "main.c",
                    f"-L{private}",
                    "-lhelper",
                    f"-Wl,-rpath,{private}",
                    "-o",
                    executable,
                ],
                check=True,
            )
            output = root / "bundle"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("preserved\n")

            with self.assertRaisesRegex(
                self.module.BundleError,
                "no eligible candidate: libhelper.so.1",
            ):
                self.module.stage_bundle(self.args(executable, output, declared_root))

            self.assertEqual(marker.read_text(), "preserved\n")
            self.assertFalse((output / "bin").exists())

    def test_resource_allowlist_stages_only_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            resources = root / "resources"
            (resources / "profiles" / "vendor").mkdir(parents=True)
            (resources / "profiles" / "vendor" / "machine.json").write_text("machine\n")
            (resources / "profiles" / "unused.json").write_text("unused\n")
            (resources / "shaders").mkdir()
            (resources / "shaders" / "preview.frag").write_text("shader\n")
            output = root / "bundle"

            result = self.module.stage_bundle(
                self.args(
                    self.system_executable(root),
                    output,
                    libraries,
                    [resources],
                    ["profiles/vendor", "shaders/*.frag"],
                )
            )

            self.assertTrue(
                (
                    output / "resources" / "profiles" / "vendor" / "machine.json"
                ).is_file()
            )
            self.assertTrue(
                (output / "resources" / "shaders" / "preview.frag").is_file()
            )
            self.assertFalse(
                (output / "resources" / "profiles" / "unused.json").exists()
            )
            inventory = result.resources
            self.assertEqual(inventory.mode, "allowlist")
            self.assertEqual(inventory.includes, ("profiles/vendor", "shaders/*.frag"))
            self.assertEqual(inventory.source_files, 3)
            self.assertEqual(inventory.selected_files, 2)
            self.assertEqual(inventory.staged_files, 2)
            self.assertEqual(inventory.omitted_files, 1)
            self.assertEqual(inventory.omitted_bytes, len("unused\n"))
            report = result.as_dict()
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["bundle_resource_bytes"], inventory.staged_bytes)
            self.assertEqual(report["resources"]["omitted_files"], 1)

    def test_unsafe_or_unmatched_resource_patterns_preserve_output(self) -> None:
        unsafe_patterns = [
            "",
            "/absolute",
            "../secret",
            "profiles/../../secret",
            "./profiles",
            "profiles//vendor",
            "profiles\\vendor",
            "profiles/**vendor",
            "profiles/[vendor",
            "not-present/**",
        ]
        for pattern in unsafe_patterns:
            with (
                self.subTest(pattern=pattern),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                libraries = root / "libraries"
                libraries.mkdir()
                resources = root / "resources"
                resources.mkdir()
                (resources / "profile.json").write_text("{}\n")
                output = root / "bundle"
                output.mkdir()
                marker = output / "keep"
                marker.write_text("preserved")

                with self.assertRaises(self.module.BundleError):
                    self.module.stage_bundle(
                        self.args(
                            Path(sys.executable),
                            output,
                            libraries,
                            [resources],
                            [pattern],
                        )
                    )

                self.assertEqual(marker.read_text(), "preserved")

    def test_multiple_resource_roots_reject_file_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserved\n")
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "profile.json").write_text("first\n")
            (second / "profile.json").write_text("second\n")

            with self.assertRaisesRegex(
                self.module.BundleError, "selected path collision"
            ):
                self.module.stage_bundle(
                    self.args(
                        Path(sys.executable),
                        output,
                        Path(sys.executable).parent,
                        [first, second],
                    )
                )

            self.assertEqual(sentinel.read_text(), "preserved\n")

    def test_resource_symlink_escape_is_rejected_before_output_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside\n")
            resources = root / "resources"
            resources.mkdir()
            (resources / "escape").symlink_to(outside)
            output = root / "bundle"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("preserved")

            with self.assertRaisesRegex(
                self.module.BundleError, "unsafe resource symlink|symlink escapes"
            ):
                self.module.stage_bundle(
                    self.args(Path(sys.executable), output, libraries, [resources])
                )

            self.assertEqual(marker.read_text(), "preserved")

    def test_allowlisted_internal_symlink_requires_and_stages_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "actual.txt").write_text("actual\n")
            (resources / "alias.txt").symlink_to("actual.txt")
            output = root / "bundle"

            with self.assertRaisesRegex(self.module.BundleError, "omits its target"):
                self.module.stage_bundle(
                    self.args(
                        Path(sys.executable),
                        output,
                        libraries,
                        [resources],
                        ["alias.txt"],
                    )
                )

            result = self.module.stage_bundle(
                self.args(
                    self.system_executable(root),
                    output,
                    libraries,
                    [resources],
                    ["alias.txt", "actual.txt"],
                )
            )
            alias = output / "resources" / "alias.txt"
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.read_text(), "actual\n")
            self.assertEqual(result.resources.source_symlinks, 1)
            self.assertEqual(result.resources.selected_symlinks, 1)
            self.assertEqual(result.resources.staged_symlinks, 1)

    def test_json_cli_report_uses_controller_bundle_size_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "keep.txt").write_text("keep\n")
            output = root / "bundle"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                return_code = self.module.main(
                    [
                        "--executable",
                        str(self.system_executable(root)),
                        "--name",
                        "fixture",
                        "--arch",
                        {"x86_64": "x86-64", "aarch64": "arm64"}[platform.machine()],
                        "--output",
                        str(output),
                        "--resources",
                        str(resources),
                        "--resource-include",
                        "keep.txt",
                        "--library-root",
                        str(libraries),
                        "--json",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["bundle_resource_bytes"], len("keep\n".encode()))
            self.assertEqual(report["resources"]["mode"], "allowlist")

    def test_resource_policy_can_come_from_build_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "keep.txt").write_text("keep\n")
            (resources / "omit.txt").write_text("omit\n")
            output = root / "bundle"
            args = self.args(
                self.system_executable(root), output, libraries, [resources]
            )
            args.resource_include = None

            with mock.patch.dict(os.environ, {"SLICER_RESOURCE_INCLUDES": "keep.txt"}):
                result = self.module.stage_bundle(args)

            self.assertTrue((output / "resources" / "keep.txt").is_file())
            self.assertFalse((output / "resources" / "omit.txt").exists())
            self.assertEqual(result.resources.includes, ("keep.txt",))


if __name__ == "__main__":
    unittest.main()
