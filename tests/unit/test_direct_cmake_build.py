from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DirectCMakeBuildTests(unittest.TestCase):
    def run_build_step(self, slicer: str) -> tuple[list[dict[str, object]], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        source = workspace / "slicer-src"
        source.mkdir()
        (source / "version.inc").write_text("SLICER_VERSION=+UNKNOWN\n")
        (source / "scripts").mkdir()

        gettext_body = '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'%s\' "$PWD" > "$GETTEXT_MARKER"\n'
        (source / "scripts" / "run_gettext.sh").write_text(gettext_body)
        (source / "run_gettext.sh").write_text(gettext_body)

        (workspace / "tools").mkdir()
        for name in ("build_cmake_target.sh", "stamp_version_date.sh"):
            shutil.copy2(ROOT / "tools" / name, workspace / "tools" / name)
        step_dir = workspace / "slicers" / slicer / "steps"
        step_dir.mkdir(parents=True)
        shutil.copy2(
            ROOT / "slicers" / slicer / "steps" / "build.sh",
            step_dir / "build.sh",
        )

        fake_bin = workspace / "fake-bin"
        fake_bin.mkdir()
        command_log = workspace / "cmake.jsonl"
        fake_cmake = fake_bin / "cmake"
        fake_cmake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "keys = ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', "
            "'CMAKE_BUILD_PARALLEL_LEVEL', 'CMAKE_POLICY_VERSION_MINIMUM')\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as output:\n"
            "    output.write(json.dumps({'argv': sys.argv[1:], "
            "'env': {key: os.environ.get(key) for key in keys}}) + '\\n')\n"
        )
        fake_cmake.chmod(0o755)
        fake_sccache = fake_bin / "sccache"
        fake_sccache.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_sccache.chmod(0o755)

        gettext_marker = workspace / "gettext-ran"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "COMMAND_LOG": str(command_log),
                "GETTEXT_MARKER": str(gettext_marker),
                "SLICER_BUILD_DATE": "2026-07-21",
                "SLICER_PCH": "OFF",
                "CMAKE_BUILD_PARALLEL_LEVEL": "7",
                "CC": "custom-cc",
                "CXX": "custom-cxx",
                "CFLAGS": "-O2 -pipe",
                "CXXFLAGS": "-O2 -pipe",
                "LDFLAGS": "-fuse-ld=lld",
            }
        )
        env.pop("CMAKE_C_COMPILER_LAUNCHER", None)
        env.pop("CMAKE_CXX_COMPILER_LAUNCHER", None)
        env.pop("CMAKE_POLICY_VERSION_MINIMUM", None)
        env.pop("CMAKE_CCACHE", None)

        subprocess.run(
            ["bash", f"./slicers/{slicer}/steps/build.sh"],
            cwd=workspace,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        calls = [json.loads(line) for line in command_log.read_text().splitlines()]
        return calls, gettext_marker

    def test_fork_build_steps_preserve_upstream_configure_contracts(self) -> None:
        contracts = {
            "AnycubicSlicerNext": {
                # Anycubic's wrapper names a nonexistent logical target. Its
                # generated Ninja graph exposes the output target instead.
                "target": "orca-slicer",
                "generator": "Ninja Multi-Config",
                "config": "Release",
                "gettext": True,
                "policy": "3.5",
                "flags": {
                    "-DSLIC3R_PCH=OFF",
                    "-DSLIC3R_STATIC=1",
                    "-DSLIC3R_GTK=3",
                    "-DBBL_RELEASE_TO_PUBLIC=1",
                    "-DBBL_INTERNAL_TESTING=0",
                    "-DORCA_TOOLS=OFF",
                },
            },
            "ElegooSlicer": {
                "target": "ElegooSlicer",
                "generator": "Ninja Multi-Config",
                "config": "Release",
                "gettext": True,
                "policy": "3.5",
                "flags": {
                    "-DSLIC3R_PCH=OFF",
                    "-DSLIC3R_STATIC=1",
                    "-DSLIC3R_GTK=3",
                    "-DBBL_RELEASE_TO_PUBLIC=1",
                    "-DBBL_INTERNAL_TESTING=0",
                    "-DELEGOO_INTERNAL_TESTING=0",
                    "-DORCA_TOOLS=OFF",
                },
            },
            "OrcaSlicer": {
                "target": "OrcaSlicer",
                "generator": "Ninja Multi-Config",
                "config": "Release",
                "gettext": True,
                "policy": "3.5",
                "flags": {
                    "-DSLIC3R_PCH=OFF",
                    "-DORCA_TOOLS=OFF",
                },
                "forbidden": {"-DSLIC3R_STATIC=1"},
                "compiler_cache": True,
            },
            "CrealityPrint": {
                "target": "CrealityPrint",
                "generator": "Ninja",
                "config": None,
                "gettext": True,
                "policy": None,
                "flags": {
                    "-DSLIC3R_STATIC=1",
                    "-DSLIC3R_GTK=3",
                    "-DBBL_RELEASE_TO_PUBLIC=1",
                    "-DBBL_INTERNAL_TESTING=0",
                    "-DUPDATE_ONLINE_MACHINES=1",
                    "-DORCA_TOOLS=ON",
                    "-DGENERATE_ORCA_HEADER=0",
                    "-DENABLE_BREAKPAD=ON",
                },
            },
            "QIDIStudio": {
                "target": "QIDIStudio",
                "generator": "Ninja",
                "config": None,
                "gettext": False,
                "policy": None,
                "flags": {
                    "-DSLIC3R_STATIC=1",
                    "-DSLIC3R_GTK=3",
                    "-DQDT_RELEASE_TO_PUBLIC=0",
                    "-DQDT_INTERNAL_TESTING=0",
                },
            },
        }

        for slicer, contract in contracts.items():
            with self.subTest(slicer=slicer):
                calls, gettext_marker = self.run_build_step(slicer)
                self.assertEqual(len(calls), 2)
                configure = calls[0]
                build = calls[1]
                configure_args = configure["argv"]
                build_args = build["argv"]

                self.assertEqual(
                    configure_args[configure_args.index("-G") + 1],
                    contract["generator"],
                )
                self.assertTrue(contract["flags"].issubset(set(configure_args)))
                self.assertTrue(
                    any(
                        argument.startswith("-DCMAKE_PREFIX_PATH=")
                        for argument in configure_args
                    )
                    if slicer != "OrcaSlicer"
                    else True
                )
                self.assertTrue(
                    set(contract.get("forbidden", set())).isdisjoint(configure_args)
                )
                self.assertEqual(build_args[-2:], ["--target", contract["target"]])
                if contract["config"] is None:
                    self.assertNotIn("--config", build_args)
                else:
                    self.assertEqual(
                        build_args[build_args.index("--config") + 1],
                        contract["config"],
                    )
                self.assertEqual(gettext_marker.exists(), contract["gettext"])
                self.assertEqual(
                    configure["env"]["CMAKE_POLICY_VERSION_MINIMUM"],
                    contract["policy"],
                )
                self.assertEqual(configure["env"]["CC"], "custom-cc")
                self.assertEqual(configure["env"]["CXX"], "custom-cxx")
                self.assertEqual(configure["env"]["CFLAGS"], "-O2 -pipe")
                self.assertEqual(configure["env"]["CXXFLAGS"], "-O2 -pipe")
                self.assertEqual(configure["env"]["LDFLAGS"], "-fuse-ld=lld")
                self.assertEqual(configure["env"]["CMAKE_BUILD_PARALLEL_LEVEL"], "7")
                if contract.get("compiler_cache"):
                    cache_flags = [
                        argument
                        for argument in configure_args
                        if argument.startswith("-DCMAKE_C_COMPILER_LAUNCHER=")
                        or argument.startswith("-DCMAKE_CXX_COMPILER_LAUNCHER=")
                    ]
                    self.assertEqual(len(cache_flags), 2)
                    self.assertTrue(
                        all(flag.endswith("/sccache") for flag in cache_flags)
                    )
                self.assertNotIn("profile_validator", " ".join(build_args))

    def test_direct_target_bundles_stage_upstream_source_resources(self) -> None:
        for slicer in (
            "AnycubicSlicerNext",
            "ElegooSlicer",
            "OrcaSlicer",
            "CrealityPrint",
            "QIDIStudio",
        ):
            with self.subTest(slicer=slicer):
                package_step = (
                    ROOT / "slicers" / slicer / "steps" / "package-binary.sh"
                ).read_text()
                self.assertIn("--resources slicer-src/resources", package_step)
                self.assertNotIn("--resources slicer-src/build/resources", package_step)

    def test_config_dump_runner_links_resources_and_collects_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "slicer-src"
            run_dir = source / "build" / "src" / "Release"
            images = source / "resources" / "images"
            output = workspace / "slicer-out"
            fake_bin = workspace / "fake-bin"
            run_dir.mkdir(parents=True)
            images.mkdir(parents=True)
            output.mkdir()
            fake_bin.mkdir()
            (images / "expected.svg").write_text("<svg/>\n")
            (output / "obsolete.json").write_text("stale\n")

            executable = run_dir / "orca-slicer"
            executable.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "[[ -f ../resources/images/expected.svg ]]\n"
                "printf '{}\\n' > print_config_def.json\n"
            )
            executable.chmod(0o755)
            xvfb_run = fake_bin / "xvfb-run"
            xvfb_run.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "[[ ${1:-} == -- ]] && shift\n"
                'exec "$@"\n'
            )
            xvfb_run.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "tools" / "run_config_dump.sh"),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--executable",
                    "build/src/Release/orca-slicer",
                ],
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual((output / "print_config_def.json").read_text(), "{}\n")
            self.assertFalse((output / "obsolete.json").exists())
            resource_link = source / "build" / "src" / "resources"
            self.assertTrue(resource_link.is_symlink())
            self.assertEqual(resource_link.resolve(), (source / "resources").resolve())

    def test_direct_config_steps_use_build_tree_executables(self) -> None:
        expected_targets = {
            "AnycubicSlicerNext": "build/src/Release/orca-slicer",
            "ElegooSlicer": "build/src/Release/elegoo-slicer",
            "OrcaSlicer": "build/src/Release/orca-slicer",
            "CrealityPrint": "build/src/CrealityPrint",
            "QIDIStudio": "build/src/qidi-studio",
        }
        for slicer, expected in expected_targets.items():
            with self.subTest(slicer=slicer):
                run_step = (ROOT / "slicers" / slicer / "steps" / "run.sh").read_text()
                self.assertIn("tools/run_config_dump.sh", run_step)
                self.assertIn(expected, run_step)
                self.assertNotIn("build/package", run_step)

    def test_config_dump_rejects_stale_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            run_dir = source / "build" / "src"
            output = root / "output"
            resources = source / "resources"
            fake_bin = root / "bin"
            run_dir.mkdir(parents=True)
            resources.mkdir()
            fake_bin.mkdir()
            (run_dir / "stale.json").write_text("{}\n")

            executable = run_dir / "orca-slicer"
            executable.write_text("#!/usr/bin/env bash\nexit 0\n")
            executable.chmod(0o755)
            xvfb_run = fake_bin / "xvfb-run"
            xvfb_run.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "[[ ${1:-} == -- ]] && shift\n"
                'exec "$@"\n'
            )
            xvfb_run.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "tools" / "run_config_dump.sh"),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--executable",
                    "build/src/orca-slicer",
                ],
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not create JSON files", result.stderr)
            self.assertFalse((output / "stale.json").exists())


if __name__ == "__main__":
    unittest.main()
