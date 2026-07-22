from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_controller_module():
    path = ROOT / "tools" / "slicerctl.py"
    spec = importlib.util.spec_from_file_location("slicerctl_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SlicerctlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_controller_module()

    def publish_fixture_variant(
        self,
        manifest,
        work: Path,
        commit: str,
        fingerprint: str,
        contents: str,
    ):
        staging_root = work / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(dir=staging_root))
        (staged / "bin").mkdir()
        (staged / "bin" / "fixture").write_text(contents)
        digest = self.module.directory_tree_sha256(staged)
        destination = self.module.immutable_build_artifact_path(
            manifest, "HEAD", "x86-64", commit, fingerprint
        )
        artifact = self.module.publish_build_artifact(staged, destination, digest)
        return self.module.publish_build_result(
            manifest,
            "HEAD",
            "x86-64",
            commit,
            fingerprint,
            {
                "slicer": "Fixture",
                "ref": "HEAD",
                "arch": "x86-64",
                "upstream_commit": commit,
                "fingerprint": fingerprint,
                "bundle": str(artifact),
                "bundle_tree_sha256": digest,
            },
        )

    def test_all_manifests_validate(self) -> None:
        found = self.module.manifests()
        self.assertEqual(len(found), 10)
        self.assertEqual(
            set(found),
            {path.parent.name for path in (ROOT / "slicers").glob("*/slicer.toml")},
        )

    def test_container_mountpoint_is_created_and_must_be_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mountpoint = Path(temporary) / "nested" / "slicer-src"
            self.module.ensure_container_mountpoint(mountpoint)
            self.assertTrue(mountpoint.is_dir())

            invalid = Path(temporary) / "file"
            invalid.write_text("not a directory")
            with self.assertRaisesRegex(SystemExit, "not a directory"):
                self.module.ensure_container_mountpoint(invalid)

    def test_node_cache_is_isolated_by_build_tree(self) -> None:
        family_cache = Path("/managed/cache/bambu")
        first = self.module.managed_node_cache(family_cache, "source-a")
        second = self.module.managed_node_cache(family_cache, "source-b")
        self.assertEqual(first, family_cache / "node" / "source-a")
        self.assertNotEqual(first, second)

    def test_manifest_rejects_backend_contract_mismatch(self) -> None:
        source = self.module.get_manifest("OrcaSlicer")
        data = deepcopy(source.data)
        data["test"]["contract"] = "prusa"
        with self.assertRaisesRegex(SystemExit, "requires test contract 'bambu'"):
            self.module.validate_manifest(self.module.Manifest(source.path, data))

    def test_manifest_rejects_duplicate_or_unlocked_supported_refs(self) -> None:
        source = self.module.get_manifest("OrcaSlicer")
        data = deepcopy(source.data)
        data["supported_refs"] = [{"ref": source.ref, "expected_commit": "a" * 40}]
        with self.assertRaisesRegex(SystemExit, "duplicate supported ref"):
            self.module.validate_manifest(self.module.Manifest(source.path, data))
        data["supported_refs"] = [
            {"ref": "v-unlocked", "expected_commit": "not-a-commit"}
        ]
        with self.assertRaisesRegex(SystemExit, "lowercase full Git object ID"):
            self.module.validate_manifest(self.module.Manifest(source.path, data))

    def test_manifest_rejects_unsafe_or_unlocked_default_ref(self) -> None:
        source = self.module.get_manifest("OrcaSlicer")
        data = deepcopy(source.data)
        data["default_ref"] = "../unsafe"
        with self.assertRaisesRegex(SystemExit, "default_ref is not a safe Git ref"):
            self.module.validate_manifest(self.module.Manifest(source.path, data))

        data["default_ref"] = "release"
        data.pop("expected_commit")
        with self.assertRaisesRegex(SystemExit, "requires expected_commit"):
            self.module.validate_manifest(self.module.Manifest(source.path, data))

    def test_vendor_compatibility_adapters_preserve_names_and_avoid_logfile(
        self,
    ) -> None:
        for slicer in ("AnycubicSlicerNext", "QIDIStudio"):
            with self.subTest(slicer=slicer):
                manifest = self.module.get_manifest(slicer)
                self.assertEqual(manifest.data["backend_engine"], "OrcaSlicer")
                self.assertEqual(manifest.data["test"]["backend_version"], "2.3.2")
                self.assertEqual(
                    manifest.data["test"]["profile_source"], "bundle-resources"
                )

        qidi = self.module.get_manifest("QIDIStudio")
        self.assertEqual(qidi.ref, "v2.05.01.52")
        self.assertEqual(
            qidi.data["expected_commit"],
            "59caa5b27d3ce571ed46761edb624c3b1ab8a099",
        )
        self.assertTrue(qidi.data["backend_supported"])

    def test_backend_version_normalizes_common_tag_prefixes(self) -> None:
        self.assertEqual(self.module.backend_version("v2.4.2"), "2.4.2")
        self.assertEqual(self.module.backend_version("version_2.9.6"), "2.9.6")
        self.assertEqual(self.module.backend_version("HEAD"), "nightly")
        self.assertEqual(self.module.backend_version("feature/thumbnail"), "nightly")
        self.assertEqual(self.module.backend_version("205734e0"), "nightly")

    def test_default_input_matches_production_contract(self) -> None:
        backend_root = Path("/backend")
        bambu = self.module.get_manifest("OrcaSlicer")
        cura = self.module.get_manifest("Cura")
        prusa = self.module.get_manifest("PrusaSlicer")

        self.assertEqual(
            self.module.default_test_input(bambu, backend_root),
            ROOT / "tests/integration/fixtures/calicat-bambu-v1.3mf",
        )
        self.assertEqual(
            self.module.default_test_input(cura, backend_root),
            backend_root / "tests/e2e/fixtures/calicat.stl",
        )
        self.assertEqual(
            self.module.default_test_input(prusa, backend_root),
            backend_root / "tests/e2e/fixtures/calicat.stl",
        )

    def test_default_build_jobs_respects_cpu_and_memory_limits(self) -> None:
        gib = 1024**3
        self.assertEqual(self.module.default_build_jobs(64, 10 * gib), 4)
        self.assertEqual(self.module.default_build_jobs(64, 2 * gib), 1)
        self.assertEqual(self.module.default_build_jobs(3, 100 * gib), 3)

    def test_build_defaults_to_binary_patches(self) -> None:
        args = self.module.parser().parse_args(["build", "OrcaSlicer"])
        self.assertEqual(args.patches, "binary")

    def test_all_commands_expose_bounded_slicer_concurrency(self) -> None:
        build = self.module.parser().parse_args(["build-all", "--workers", "3"])
        test = self.module.parser().parse_args(["test-all", "--workers", "2"])
        backfill = self.module.parser().parse_args(
            ["backfill", "--workers", "4", "--plan-only"]
        )
        self.assertEqual(build.workers, 3)
        self.assertEqual(test.workers, 2)
        self.assertEqual(backfill.workers, 4)
        self.assertEqual(backfill.max_versions, 3)
        self.assertEqual(backfill.versions_from, "index")
        self.assertTrue(backfill.plan_only)

    def test_release_selection_matches_downloader_grouping_then_caps(self) -> None:
        selected = self.module.select_release_tags(
            [
                "v02.06.01.55",
                "v02.07.00.55",
                "v02.07.01.57",
                "v02.07.01.62",
                "v02.08.00.01-beta",
            ]
        )
        self.assertEqual(
            [(ref, group) for ref, group, _parts in selected],
            [
                ("v02.07.01.62", "02.07.01"),
                ("v02.07.00.55", "02.07.00"),
                ("v02.06.01.55", "02.06.01"),
            ],
        )

    def test_three_part_releases_remain_distinct_and_support_a_minimum(self) -> None:
        selected = self.module.select_release_tags(
            ["v2.3.2", "v2.4.0", "v2.4.1", "v2.4.2", "v2.5.0-rc"],
            maximum=None,
            minimum="2.4.0",
        )
        self.assertEqual(
            [ref for ref, _group, _parts in selected],
            ["v2.4.2", "v2.4.1", "v2.4.0"],
        )

    def test_release_alias_and_zero_padding_match_downloader_text_groups(self) -> None:
        aliases = self.module.select_release_tags(
            ["v1.2.3", "1.2.3"], maximum=None
        )
        self.assertEqual([item[0] for item in aliases], ["v1.2.3"])
        padded = self.module.select_release_tags(
            ["v02.07.01.57", "v2.7.1.62"], maximum=None
        )
        self.assertEqual(
            [item[0] for item in padded], ["v2.7.1.62", "v02.07.01.57"]
        )

    def test_minimum_version_requires_names_for_a_multi_slicer_plan(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires SLICER=VERSION"):
            self.module.parse_minimum_versions(
                ["2.4.0"], ["OrcaSlicer", "PrusaSlicer"]
            )
        self.assertEqual(
            self.module.parse_minimum_versions(
                ["OrcaSlicer=v2.4.0"], ["OrcaSlicer", "PrusaSlicer"]
            ),
            {"OrcaSlicer": "2.4.0"},
        )

    def test_annotated_release_tags_are_peeled_to_commits(self) -> None:
        manifest = self.module.Manifest(
            Path("/tmp/Fixture/slicer.toml"),
            {"name": "Fixture", "repository": "https://github.com/acme/fixture.git"},
        )
        tag_object = "1" * 40
        peeled_commit = "2" * 40
        lightweight_commit = "3" * 40
        advertisement = "\n".join(
            (
                f"{tag_object}\trefs/tags/v2.0.0",
                f"{peeled_commit}\trefs/tags/v2.0.0^{{}}",
                f"{lightweight_commit}\trefs/tags/v1.0.0",
            )
        )
        completed = subprocess.CompletedProcess(
            args=["git", "ls-remote"], returncode=0, stdout=advertisement, stderr=""
        )
        with mock.patch.object(self.module.subprocess, "run", return_value=completed):
            resolved = self.module.resolve_release_tag_commits(
                manifest, ["v2.0.0", "v1.0.0"]
            )
        self.assertEqual(
            resolved,
            {"v2.0.0": peeled_commit, "v1.0.0": lightweight_commit},
        )

    def test_git_tag_discovery_is_host_independent_and_excludes_prereleases(
        self,
    ) -> None:
        manifest = self.module.Manifest(
            Path("/tmp/Fixture/slicer.toml"),
            {"name": "Fixture", "repository": "ssh://git.example/acme/fixture.git"},
        )
        tag_object = "1" * 40
        peeled_commit = "2" * 40
        prerelease_commit = "3" * 40
        advertisement = "\n".join(
            (
                f"{tag_object}\trefs/tags/v2.0.0",
                f"{peeled_commit}\trefs/tags/v2.0.0^{{}}",
                f"{prerelease_commit}\trefs/tags/v3.0.0-rc",
            )
        )
        completed = subprocess.CompletedProcess(
            args=["git", "ls-remote"], returncode=0, stdout=advertisement, stderr=""
        )
        with mock.patch.object(self.module.subprocess, "run", return_value=completed):
            tags, commits = self.module.git_stable_release_tags(manifest)
        self.assertEqual(tags, ["v2.0.0"])
        self.assertEqual(commits, {"v2.0.0": peeled_commit})

    def test_git_tag_history_prefers_manifest_alias_for_the_same_release(
        self,
    ) -> None:
        commit = "1" * 40
        manifest = self.module.Manifest(
            Path("/tmp/Fixture/slicer.toml"),
            {
                "name": "Fixture",
                "repository": "https://github.com/acme/fixture.git",
                "default_ref": "v2.3.3",
                "expected_commit": commit,
                "test": {},
            },
        )
        with mock.patch.object(
            self.module,
            "git_stable_release_tags",
            return_value=(["2.3.3", "v2.3.3"], {"2.3.3": commit, "v2.3.3": commit}),
        ), mock.patch.object(
            self.module,
            "resolve_release_tag_commits",
            side_effect=AssertionError("advertised commits should be reused"),
        ):
            targets = self.module.historical_release_targets(
                manifest, source="git-tags"
            )
        self.assertEqual([target.ref for target in targets], ["v2.3.3"])
        self.assertEqual(targets[0].expected_commit, commit)

    def test_index_history_uses_recorded_commits_without_remote_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "Fixture"
            output = directory / "out"
            output.mkdir(parents=True)
            commits = {
                "v02.06.01.55": "1" * 40,
                "v02.07.00.55": "2" * 40,
                "v02.07.01.57": "3" * 40,
                "v02.07.01.62": "4" * 40,
            }
            self.module.atomic_json(
                output / "_index.json",
                {
                    "versions": {
                        ref: {"upstream_ref": commit}
                        for ref, commit in commits.items()
                    }
                },
            )
            manifest = self.module.Manifest(
                directory / "slicer.toml",
                {
                    "name": "Fixture",
                    "repository": "https://github.com/acme/fixture.git",
                    "default_ref": "v02.07.01.62",
                    "expected_commit": commits["v02.07.01.62"],
                    "test": {},
                },
            )
            with mock.patch.object(
                self.module,
                "resolve_release_tag_commits",
                side_effect=AssertionError("remote resolution should not run"),
            ):
                targets = self.module.historical_release_targets(manifest)
        self.assertEqual(
            [target.ref for target in targets],
            ["v02.07.01.62", "v02.07.00.55", "v02.06.01.55"],
        )
        self.assertEqual(targets[0].expected_commit, "4" * 40)

    def test_index_history_rejects_a_conflicting_manifest_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "Fixture"
            output = directory / "out"
            output.mkdir(parents=True)
            self.module.atomic_json(
                output / "_index.json",
                {
                    "versions": {
                        "v1.2.3": {"upstream_ref": "1" * 40},
                    }
                },
            )
            manifest = self.module.Manifest(
                directory / "slicer.toml",
                {
                    "name": "Fixture",
                    "repository": "https://github.com/acme/fixture.git",
                    "default_ref": "v1.2.3",
                    "expected_commit": "2" * 40,
                    "test": {},
                },
            )
            with self.assertRaisesRegex(SystemExit, "index and manifest disagree"):
                self.module.historical_release_targets(manifest)

    def test_compact_build_output_stats_legacy_executable_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "bundle" / "bin" / "fixture"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"binary")
            (executable.parent / "library.so").write_bytes(b"l" * 14)
            resources = executable.parents[1] / "resources"
            resources.mkdir()
            (resources / "profiles.json").write_bytes(b"r" * 30)
            target = self.module.ReleaseTarget(
                slicer="Fixture",
                ref="v1.2.3",
                version="1.2.3",
                version_parts=(1, 2, 3),
                version_group="1.2.3",
                expected_commit="a" * 40,
            )
            compact = self.module.compact_build_output(
                target,
                {
                    "executable": str(executable),
                    "bundle": str(executable.parents[1]),
                    "bundle_bin_bytes": 20,
                    "bundle_resource_bytes": 30,
                    "bundle_bytes": 50,
                    "reused": True,
                },
            )
        self.assertEqual(compact["executable_bytes"], 6)
        self.assertEqual(compact["bundle_bin_bytes"], 20)
        self.assertEqual(compact["bundle_resource_bytes"], 30)
        self.assertEqual(compact["bundle_bytes"], 50)
        self.assertEqual(compact["status"], "reused")

    def test_backfill_preserves_plan_order_and_writes_binary_size_reports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "reports" / "history.json"
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {
                    "name": "Fixture",
                    "architectures": ["x86-64"],
                },
            )
            targets = [
                self.module.ReleaseTarget(
                    slicer="Fixture",
                    ref=ref,
                    version=ref.removeprefix("v"),
                    version_parts=parts,
                    version_group=ref.removeprefix("v"),
                    expected_commit=str(index) * 40,
                )
                for index, ref, parts in (
                    (3, "v3.0.0", (3, 0, 0)),
                    (2, "v2.0.0", (2, 0, 0)),
                    (1, "v1.0.0", (1, 0, 0)),
                )
            ]
            release_one_done = threading.Event()
            release_two_done = threading.Event()
            completion_order: list[str] = []
            completion_lock = threading.Lock()

            def fake_build(child: argparse.Namespace) -> dict[str, object]:
                # Force completion in the reverse of the selected release order.
                if child.ref == "v3.0.0":
                    self.assertTrue(release_two_done.wait(5))
                elif child.ref == "v2.0.0":
                    self.assertTrue(release_one_done.wait(5))

                number = int(child.ref[1])
                bundle = root / "artifacts" / child.ref
                executable = bundle / "bin" / "fixture"
                executable.parent.mkdir(parents=True)
                executable.write_bytes(b"x" * (number * 11))
                (executable.parent / "library.so").write_bytes(b"l" * (number * 90))
                resources = bundle / "resources"
                resources.mkdir()
                (resources / "profiles.json").write_bytes(b"r" * (number * 201))
                with completion_lock:
                    completion_order.append(child.ref)
                if child.ref == "v1.0.0":
                    release_one_done.set()
                elif child.ref == "v2.0.0":
                    release_two_done.set()
                return {
                    "executable": str(executable),
                    "bundle": str(bundle),
                    "bundle_bin_bytes": number * 101,
                    "bundle_resource_bytes": number * 201,
                    "bundle_bytes": number * 302,
                    "bundle_tree_sha256": str(number) * 64,
                    "result": str(root / "results" / f"{child.ref}.json"),
                    "immutable_result": str(
                        root / "results" / "immutable" / f"{child.ref}.json"
                    ),
                    "fingerprint": f"fingerprint-{number}",
                    "phase_seconds": {"build": float(number)},
                    "reused": child.ref == "v2.0.0",
                }

            args = argparse.Namespace(
                slicers=["Fixture"],
                arch="x86-64",
                minimum_versions=None,
                all_versions=False,
                max_versions=3,
                versions_from="index",
                workers=3,
                fail_fast=False,
                plan_only=False,
                jobs=1,
                report=report_path,
            )
            with (
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "historical_release_targets",
                    return_value=targets,
                ),
                mock.patch.object(self.module, "build_one", side_effect=fake_build),
                redirect_stdout(io.StringIO()),
            ):
                self.module.command_backfill(args)

            report = self.module.load_json(report_path)
            markdown_path = report_path.with_suffix(".md")
            markdown = markdown_path.read_text()

        self.assertEqual(completion_order, ["v1.0.0", "v2.0.0", "v3.0.0"])
        self.assertEqual(
            [build["ref"] for build in report["builds"]],
            ["v3.0.0", "v2.0.0", "v1.0.0"],
        )
        for number, build in zip((3, 2, 1), report["builds"], strict=True):
            expected_bundle = root / "artifacts" / f"v{number}.0.0"
            self.assertEqual(build["executable_bytes"], number * 11)
            self.assertEqual(build["bundle_bin_bytes"], number * 101)
            self.assertEqual(build["bundle_resource_bytes"], number * 201)
            self.assertEqual(build["bundle_bytes"], number * 302)
            self.assertEqual(build["bundle"], str(expected_bundle))
            self.assertEqual(
                build["executable"], str(expected_bundle / "bin" / "fixture")
            )
        self.assertIn(
            "| Executable | bin/ closure | Resources | Bundle | Artifact |", markdown
        )
        self.assertIn("33 B", markdown)
        self.assertIn("303 B", markdown)
        self.assertIn("603 B", markdown)
        self.assertIn("906 B", markdown)
        self.assertIn(str(root / "artifacts" / "v3.0.0"), markdown)

    def test_backfill_rejects_an_invalid_report_before_discovery_or_build(
        self,
    ) -> None:
        manifest = self.module.Manifest(
            Path("/tmp/Fixture/slicer.toml"),
            {"name": "Fixture", "architectures": ["x86-64"]},
        )
        args = self.module.parser().parse_args(
            [
                "backfill",
                "--slicer",
                "Fixture",
                "--report",
                "sizes.md",
            ]
        )
        with (
            mock.patch.object(
                self.module, "manifests", return_value={"Fixture": manifest}
            ),
            mock.patch.object(
                self.module,
                "historical_release_targets",
                side_effect=AssertionError("discovery must not run"),
            ),
            mock.patch.object(
                self.module,
                "build_one",
                side_effect=AssertionError("build must not run"),
            ),
            self.assertRaisesRegex(SystemExit, "must name a .json file"),
        ):
            self.module.command_backfill(args)

    def test_backfill_writes_partial_success_before_raising_for_a_build_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "partial.json"
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {
                    "name": "Fixture",
                    "architectures": ["x86-64"],
                },
            )
            targets = [
                self.module.ReleaseTarget(
                    slicer="Fixture",
                    ref=ref,
                    version=ref.removeprefix("v"),
                    version_parts=parts,
                    version_group=ref.removeprefix("v"),
                    expected_commit=str(index) * 40,
                )
                for index, ref, parts in (
                    (2, "v2.0.0", (2, 0, 0)),
                    (1, "v1.0.0", (1, 0, 0)),
                )
            ]
            bundle = root / "artifacts" / "v2.0.0"
            executable = bundle / "bin" / "fixture"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"successful-binary")
            (executable.parent / "library.so").write_bytes(b"l" * 84)
            resources = bundle / "resources"
            resources.mkdir()
            (resources / "profiles.json").write_bytes(b"r" * 202)

            def fake_build(child: argparse.Namespace) -> dict[str, object]:
                if child.ref == "v1.0.0":
                    raise subprocess.CalledProcessError(7, ["fixture-build", child.ref])
                return {
                    "executable": str(executable),
                    "bundle": str(bundle),
                    "bundle_bin_bytes": 101,
                    "bundle_resource_bytes": 202,
                    "bundle_bytes": 303,
                    "reused": False,
                }

            args = argparse.Namespace(
                slicers=["Fixture"],
                arch="x86-64",
                minimum_versions=None,
                all_versions=False,
                max_versions=3,
                versions_from="index",
                workers=2,
                fail_fast=False,
                plan_only=False,
                jobs=1,
                report=report_path,
            )
            with (
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "historical_release_targets",
                    return_value=targets,
                ),
                mock.patch.object(self.module, "build_one", side_effect=fake_build),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaisesRegex(
                    SystemExit,
                    "Historical build completed with 0 discovery and 1 build",
                ),
            ):
                self.module.command_backfill(args)

            self.assertTrue(report_path.is_file())
            markdown_path = report_path.with_suffix(".md")
            self.assertTrue(markdown_path.is_file())
            report = self.module.load_json(report_path)
            markdown = markdown_path.read_text()

        self.assertEqual(
            [(build["ref"], build["status"]) for build in report["builds"]],
            [("v2.0.0", "built"), ("v1.0.0", "failed")],
        )
        self.assertEqual(report["summary"]["successful_builds"], 1)
        self.assertEqual(report["summary"]["failed_builds"], 1)
        self.assertEqual(report["builds"][0]["bundle"], str(bundle))
        self.assertEqual(report["builds"][0]["executable_bytes"], 17)
        self.assertIn("Fixture@v1.0.0", report["build_failures"])
        self.assertIn("command exited 7", report["builds"][1]["error"])
        self.assertIn("v2.0.0", markdown)
        self.assertIn(str(bundle), markdown)

    def test_supported_ref_matrix_expands_only_commit_locked_refs(self) -> None:
        targets = self.module.selected_manifest_targets(
            ["OrcaSlicer"], "x86-64", all_refs=True
        )
        self.assertEqual(
            targets,
            [
                ("OrcaSlicer", "v2.4.2"),
                ("OrcaSlicer", "v2.4.1"),
                ("OrcaSlicer", "v2.4.0"),
            ],
        )
        manifest = self.module.get_manifest("OrcaSlicer")
        self.assertTrue(
            all(
                spec["expected_commit"]
                for spec in self.module.manifest_ref_specs(manifest)
            )
        )
        matrix = self.module.parser().parse_args(["matrix", "--all-refs"])
        build = self.module.parser().parse_args(["build-all", "--all-refs"])
        test = self.module.parser().parse_args(["test-all", "--all-refs"])
        self.assertTrue(matrix.all_refs)
        self.assertTrue(build.all_refs)
        self.assertTrue(test.all_refs)

    def test_dump_fallback_is_not_suppressed_by_unrelated_version_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slicer = Path(temporary) / "Fixture"
            version = slicer / "patches" / "v1"
            version.mkdir(parents=True)
            (version / "fix.patch").write_text("fix")
            fallback = slicer / "patches" / "dump_configs.patch"
            fallback.write_text("dump")
            manifest = self.module.Manifest(slicer / "slicer.toml", {"name": "Fixture"})
            patches = self.module.legacy_patch_files(manifest, "v1", "dump")
        self.assertEqual(
            [path.name for path in patches], ["fix.patch", "dump_configs.patch"]
        )

    def test_head_uses_nightly_patch_directory_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slicer = Path(temporary) / "Fixture"
            nightly = slicer / "patches" / "binary" / "nightly"
            nightly.mkdir(parents=True)
            (nightly / "fix.patch").write_text("fix")
            manifest = self.module.Manifest(slicer / "slicer.toml", {"name": "Fixture"})
            patches = self.module.legacy_patch_files(manifest, "HEAD", "binary")
        self.assertEqual([path.name for path in patches], ["fix.patch"])

    def test_empty_head_patch_directory_still_uses_nightly_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slicer = Path(temporary) / "Fixture"
            (slicer / "patches" / "binary" / "HEAD").mkdir(parents=True)
            nightly = slicer / "patches" / "binary" / "nightly"
            nightly.mkdir()
            (nightly / "fix.patch").write_text("fix")
            manifest = self.module.Manifest(slicer / "slicer.toml", {"name": "Fixture"})
            patches = self.module.legacy_patch_files(manifest, "HEAD", "binary")
        self.assertEqual([path.name for path in patches], ["fix.patch"])

    def test_binary_ref_alias_reuses_a_verified_overlay_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slicer = Path(temporary) / "Fixture"
            common = slicer / "patches" / "common" / "v2.4.1"
            canonical = slicer / "patches" / "binary" / "v2.4.2"
            common.mkdir(parents=True)
            canonical.mkdir(parents=True)
            (common / "actual-ref.patch").write_text("actual")
            (canonical / "canonical.patch").write_text("canonical")
            manifest = self.module.Manifest(
                slicer / "slicer.toml",
                {
                    "name": "Fixture",
                    "patches": {
                        "binary_ref_aliases": {"v2.4.1": "v2.4.2"},
                    },
                },
            )
            patches = self.module.legacy_patch_files(
                manifest, "v2.4.1", "binary"
            )
        self.assertEqual(
            [path.name for path in patches],
            ["actual-ref.patch", "canonical.patch"],
        )

    def test_shared_patch_sets_precede_local_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "patches" / "orca-common" / "binary" / "all"
            local = root / "slicers" / "Fixture" / "patches" / "binary" / "all"
            shared.mkdir(parents=True)
            local.mkdir(parents=True)
            (shared / "shared.patch").write_text("shared")
            (local / "local.patch").write_text("local")
            manifest = self.module.Manifest(
                root / "slicers" / "Fixture" / "slicer.toml",
                {"name": "Fixture", "patches": {"shared": ["orca-common"]}},
            )
            with mock.patch.object(self.module, "ROOT", root):
                patches = self.module.legacy_patch_files(manifest, "v1", "binary")
        self.assertEqual(
            [path.name for path in patches], ["shared.patch", "local.patch"]
        )

    def test_common_patches_apply_to_binary_and_dump_before_mode_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_root = root / "slicers" / "Fixture" / "patches"
            directories = {
                "common/all": "01-common-all.patch",
                "common/v1": "02-common-version.patch",
                "binary/all": "03-binary-all.patch",
                "binary/v1": "04-binary-version.patch",
                "all": "03-dump-all.patch",
                "v1": "04-dump-version.patch",
            }
            for directory, filename in directories.items():
                target = patch_root / directory
                target.mkdir(parents=True, exist_ok=True)
                (target / filename).write_text(filename)
            manifest = self.module.Manifest(
                root / "slicers" / "Fixture" / "slicer.toml",
                {"name": "Fixture"},
            )

            binary = self.module.legacy_patch_files(manifest, "v1", "binary")
            dump = self.module.legacy_patch_files(manifest, "v1", "dump")

        self.assertEqual(
            [path.name for path in binary],
            [
                "01-common-all.patch",
                "02-common-version.patch",
                "03-binary-all.patch",
                "04-binary-version.patch",
            ],
        )
        self.assertEqual(
            [path.name for path in dump],
            [
                "01-common-all.patch",
                "02-common-version.patch",
                "03-dump-all.patch",
                "04-dump-version.patch",
            ],
        )

    def test_patch_roots_preserve_shared_then_local_overlay_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "patches" / "shared"
            local = root / "slicers" / "Fixture" / "patches"
            patch_locations = (
                (shared / "common/all/shared-common.patch", "shared-common"),
                (shared / "binary/all/shared-binary.patch", "shared-binary"),
                (local / "common/all/local-common.patch", "local-common"),
                (local / "binary/all/local-binary.patch", "local-binary"),
            )
            for target, contents in patch_locations:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)
            manifest = self.module.Manifest(
                root / "slicers" / "Fixture" / "slicer.toml",
                {"name": "Fixture", "patches": {"shared": ["shared"]}},
            )
            with mock.patch.object(self.module, "ROOT", root):
                patches = self.module.legacy_patch_files(manifest, "v1", "binary")

        self.assertEqual(
            [path.name for path in patches],
            [
                "shared-common.patch",
                "shared-binary.patch",
                "local-common.patch",
                "local-binary.patch",
            ],
        )

    def test_storage_names_do_not_alias_distinct_refs(self) -> None:
        self.assertNotEqual(
            self.module.storage_name("feature/one"),
            self.module.storage_name("feature-one"),
        )

    def test_bounded_storage_names_preserve_the_uniqueness_suffix(self) -> None:
        prefix = "feature/" + "same-prefix-" * 20
        first = self.module.bounded_storage_name(prefix + "one", 80)
        second = self.module.bounded_storage_name(prefix + "two", 80)
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(first), 80)
        self.assertRegex(first, r"-[0-9a-f]{8}$")

    def test_prepared_source_has_container_safe_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            upstream.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.com"],
                cwd=upstream,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"], cwd=upstream, check=True
            )
            subprocess.run(
                ["git", "config", "commit.gpgsign", "false"],
                cwd=upstream,
                check=True,
            )
            (upstream / "README").write_text("fixture\n")
            subprocess.run(["git", "add", "README"], cwd=upstream, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=upstream, check=True
            )
            upstream_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=upstream,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            manifest_dir = root / "FixtureSlicer"
            manifest_dir.mkdir()
            declared_remote = upstream.as_uri()
            manifest = self.module.Manifest(
                manifest_dir / "slicer.toml",
                {
                    "name": "FixtureSlicer",
                    "repository": declared_remote,
                    "default_ref": "HEAD",
                    "expected_commit": upstream_commit,
                },
            )
            with mock.patch.object(self.module, "WORK_ROOT", root / "work"):
                checkout, _commit = self.module.prepare_source(
                    manifest, "HEAD", upstream, "test"
                )
                mirror = root / "work" / "git" / "FixtureSlicer.git"
                subprocess.run(
                    ["git", "update-ref", "refs/tags/cache-only", upstream_commit],
                    cwd=mirror,
                    check=True,
                )
                subprocess.run(
                    ["git", "remote", "set-url", "origin", "/stale/upstream"],
                    cwd=mirror,
                    check=True,
                )
                second_checkout, _commit = self.module.prepare_source(
                    manifest, "HEAD", upstream, "second-test"
                )
                remote = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=mirror,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout.strip()
                retained_tag = subprocess.run(
                    ["git", "rev-parse", "refs/tags/cache-only"],
                    cwd=mirror,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout.strip()
                mismatched = self.module.Manifest(
                    manifest.path,
                    {**manifest.data, "expected_commit": "0" * 40},
                )
                with self.assertRaisesRegex(SystemExit, "expected locked commit"):
                    self.module.prepare_source(
                        mismatched, "HEAD", None, "mismatched-lock"
                    )
                with self.assertRaisesRegex(SystemExit, "conflicting commit locks"):
                    self.module.prepare_source(
                        manifest,
                        "HEAD",
                        None,
                        "transient-mismatched-lock",
                        expected_commit="1" * 40,
                    )
            self.assertTrue((checkout / ".git").is_dir())
            self.assertTrue((second_checkout / ".git").is_dir())
            self.assertEqual(remote, declared_remote)
            self.assertEqual(retained_tag, upstream_commit)

    def test_dependency_fingerprint_can_be_shared_by_identical_forks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            sources = []
            for name in ("One", "Two"):
                directory = root / name
                (directory / "steps").mkdir(parents=True)
                (directory / "steps" / "build-deps.sh").write_text("#!/bin/sh\ntrue\n")
                source = root / f"source-{name}"
                (source / "deps").mkdir(parents=True)
                (source / "deps" / "versions.cmake").write_text("set(VERSION 1)\n")
                manifests.append(
                    self.module.Manifest(
                        directory / "slicer.toml",
                        {"name": name, "family": "orca"},
                    )
                )
                sources.append(source)
            env = {key: "" for key in self.module.FORWARDED_BUILD_ENV}
            env.update(self.module.BUILD_ENV_DEFAULTS)
            with mock.patch.object(self.module, "_hash_index_paths"):
                first = self.module.dependency_fingerprint(
                    manifests[0], sources[0], "x86-64", "sha256:image", env
                )
                second = self.module.dependency_fingerprint(
                    manifests[1], sources[1], "x86-64", "sha256:image", env
                )
        self.assertEqual(first, second)

    def test_dependency_fingerprint_separates_builder_images(self) -> None:
        manifest = self.module.get_manifest("OrcaSlicer")
        source = ROOT
        env = self.module.effective_build_environment()
        with mock.patch.object(self.module, "_hash_index_paths"):
            first = self.module.dependency_fingerprint(
                manifest, source, "x86-64", "sha256:one", env
            )
            second = self.module.dependency_fingerprint(
                manifest, source, "x86-64", "sha256:two", env
            )
        self.assertNotEqual(first, second)

    def test_dependency_fingerprint_includes_declared_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Fixture"
            (directory / "steps").mkdir(parents=True)
            (directory / "steps" / "build-deps.sh").write_text("#!/bin/sh\ntrue\n")
            helper = directory / "steps" / "dependency-env.sh"
            helper.write_text("VALUE=one\n")
            source = root / "source"
            source.mkdir()
            manifest = self.module.Manifest(
                directory / "slicer.toml",
                {
                    "name": "Fixture",
                    "family": "orca",
                    "build": {"dependency_inputs": ["steps/dependency-env.sh"]},
                },
            )
            env = self.module.effective_build_environment(manifest)
            with mock.patch.object(self.module, "_hash_index_paths"):
                first = self.module.dependency_fingerprint(
                    manifest, source, "x86-64", "sha256:image", env
                )
                helper.write_text("VALUE=two\n")
                second = self.module.dependency_fingerprint(
                    manifest, source, "x86-64", "sha256:image", env
                )
        self.assertNotEqual(first, second)

    def test_dependency_fingerprint_includes_tracked_driver_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "Fixture"
            (manifest_dir / "steps").mkdir(parents=True)
            (manifest_dir / "steps" / "build-deps.sh").write_text("#!/bin/sh\ntrue\n")
            source = root / "source"
            (source / "linux.d").mkdir(parents=True)
            (source / "scripts" / "linux.d").mkdir(parents=True)
            tracked = (
                source / "CMakeLists.txt",
                source / "linux.d" / "debian",
                source / "scripts" / "linux.d" / "debian",
            )
            for path in tracked:
                path.write_text(f"initial {path.name}\n")
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(["git", "add", "."], cwd=source, check=True)

            manifest = self.module.Manifest(
                manifest_dir / "slicer.toml",
                {"name": "Fixture", "family": "orca"},
            )
            env = self.module.effective_build_environment(manifest)
            fingerprints = [
                self.module.dependency_fingerprint(
                    manifest, source, "x86-64", "sha256:image", env
                )
            ]
            for path in tracked:
                path.write_text(path.read_text() + "changed\n")
                subprocess.run(["git", "add", path], cwd=source, check=True)
                fingerprints.append(
                    self.module.dependency_fingerprint(
                        manifest, source, "x86-64", "sha256:image", env
                    )
                )

        self.assertEqual(len(set(fingerprints)), len(fingerprints))

    def test_build_tree_fingerprint_tracks_dependencies_and_build_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "Fixture"
            (manifest_dir / "steps").mkdir(parents=True)
            manifest_path = manifest_dir / "slicer.toml"
            manifest_path.write_text("name = 'Fixture'\n")
            build_step = manifest_dir / "steps" / "build.sh"
            build_step.write_text("#!/bin/sh\ntrue\n")
            manifest = self.module.Manifest(
                manifest_path,
                {"name": "Fixture", "family": "orca"},
            )
            env = self.module.effective_build_environment(manifest)
            arguments = (
                manifest,
                "a" * 40,
                "x86-64",
                "sha256:image",
                "b" * 64,
                "c" * 64,
                env,
            )
            initial = self.module.build_tree_fingerprint(*arguments)
            build_step.write_text("#!/bin/sh\nfalse\n")
            changed_step = self.module.build_tree_fingerprint(*arguments)
            changed_dependencies = self.module.build_tree_fingerprint(
                *arguments[:4], "d" * 64, *arguments[5:]
            )

        self.assertNotEqual(initial, changed_step)
        self.assertNotEqual(changed_step, changed_dependencies)

    def test_build_tree_fingerprint_tracks_direct_cmake_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "slicers" / "Fixture"
            steps = manifest_dir / "steps"
            tools = root / "tools"
            steps.mkdir(parents=True)
            tools.mkdir()
            (steps / "build.sh").write_text(
                "#!/bin/sh\nbash ./tools/build_cmake_target.sh --help\n"
            )
            helper = tools / "build_cmake_target.sh"
            helper.write_text("#!/bin/sh\ntrue\n")
            (tools / "stamp_version_date.sh").write_text("#!/bin/sh\ntrue\n")
            manifest = self.module.Manifest(
                manifest_dir / "slicer.toml",
                {"name": "Fixture", "family": "orca"},
            )
            env = self.module.effective_build_environment(manifest)
            arguments = (
                manifest,
                "a" * 40,
                "x86-64",
                "sha256:image",
                "b" * 64,
                "c" * 64,
                env,
            )
            with mock.patch.object(self.module, "ROOT", root):
                initial = self.module.build_tree_fingerprint(*arguments)
                helper.write_text("#!/bin/sh\nfalse\n")
                changed = self.module.build_tree_fingerprint(*arguments)

        self.assertNotEqual(initial, changed)

    def test_build_tree_ignores_steps_that_never_configure_the_local_build(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "Fixture"
            steps = manifest_dir / "steps"
            steps.mkdir(parents=True)
            (steps / "build.sh").write_text("#!/bin/sh\ntrue\n")
            (steps / "install-deps.sh").write_text("#!/bin/sh\ntrue\n")
            (steps / "run.sh").write_text("#!/bin/sh\ntrue\n")
            manifest = self.module.Manifest(
                manifest_dir / "slicer.toml",
                {"name": "Fixture", "family": "orca"},
            )
            env = self.module.effective_build_environment(manifest)
            arguments = (
                manifest,
                "a" * 40,
                "x86-64",
                "sha256:image",
                "b" * 64,
                "c" * 64,
                env,
            )
            initial = self.module.build_tree_fingerprint(*arguments)
            (steps / "install-deps.sh").write_text("#!/bin/sh\nfalse\n")
            (steps / "run.sh").write_text("#!/bin/sh\nfalse\n")
            unchanged = self.module.build_tree_fingerprint(*arguments)

        self.assertEqual(initial, unchanged)

    def test_build_fingerprint_ignores_test_and_ci_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "Fixture"
            steps = manifest_dir / "steps"
            steps.mkdir(parents=True)
            for name in ("build-deps.sh", "build.sh", "package-binary.sh"):
                (steps / name).write_text("#!/bin/sh\ntrue\n")
            data = {
                "name": "Fixture",
                "family": "orca",
                "executable": "fixture-slicer",
                "test": {"contract": "bambu"},
                "ci": {"binary": {"enabled": True}},
            }
            manifest = self.module.Manifest(manifest_dir / "slicer.toml", data)
            arguments = (
                "a" * 40,
                [],
                "x86-64",
                "sha256:image",
                "b" * 64,
                "c" * 64,
                "d" * 64,
                {},
            )
            initial = self.module.build_fingerprint(manifest, *arguments)
            metadata_changed = self.module.Manifest(
                manifest.path,
                {
                    **data,
                    "test": {"contract": "bambu", "profile": "different"},
                    "ci": {"binary": {"enabled": False}},
                },
            )
            unchanged = self.module.build_fingerprint(metadata_changed, *arguments)
            executable_changed = self.module.Manifest(
                manifest.path, {**data, "executable": "different-slicer"}
            )
            changed = self.module.build_fingerprint(executable_changed, *arguments)

        self.assertEqual(initial, unchanged)
        self.assertNotEqual(initial, changed)

    def test_managed_build_tree_preserves_only_compatible_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / ".git").mkdir(parents=True)
            build = source / "build"
            build.mkdir()
            (build / "stale").write_text("old\n")
            dependency = source / "deps" / "build" / "keep"
            dependency.parent.mkdir(parents=True)
            dependency.write_text("warm\n")

            self.module.prepare_managed_build_tree(source, "a" * 64)
            self.assertFalse((build / "stale").exists())
            partial = build / "partial"
            partial.write_text("resume\n")
            self.module.prepare_managed_build_tree(source, "a" * 64)
            self.assertTrue(partial.is_file())

            self.module.prepare_managed_build_tree(source, "b" * 64)
            self.assertFalse(partial.exists())
            self.assertEqual(dependency.read_text(), "warm\n")
            self.assertTrue(
                self.module.valid_build_tree_stamp(
                    self.module.build_tree_stamp_path(source), "b" * 64
                )
            )

    def test_managed_build_tree_refuses_symlinked_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            outside = root / "outside"
            (source / ".git").mkdir(parents=True)
            outside.mkdir()
            sentinel = outside / "keep"
            sentinel.write_text("safe\n")
            (source / "build").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(SystemExit, "unsafe managed build path"):
                self.module.prepare_managed_build_tree(source, "a" * 64)
            self.assertEqual(sentinel.read_text(), "safe\n")

    def test_immutable_results_precede_and_survive_pointer_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {"name": "Fixture"},
            )
            commit = "a" * 40
            fingerprint = "b" * 64
            payload = {
                "upstream_commit": commit,
                "fingerprint": fingerprint,
                "dependency_cache_hit": False,
                "bundle_tree_sha256": "c" * 64,
            }
            writes: list[Path] = []
            atomic_json = self.module.atomic_json

            def tracked_write(path, value):
                writes.append(path)
                atomic_json(path, value)

            with (
                mock.patch.object(self.module, "WORK_ROOT", root / "work"),
                mock.patch.object(self.module, "atomic_json", tracked_write),
            ):
                first = self.module.publish_build_result(
                    manifest,
                    "HEAD",
                    "x86-64",
                    commit,
                    fingerprint,
                    payload,
                )
                immutable = Path(first["immutable_result"])
                pointer = Path(first["result"])
                changed = dict(payload, dependency_cache_hit=True)
                second = self.module.publish_build_result(
                    manifest,
                    "HEAD",
                    "x86-64",
                    commit,
                    fingerprint,
                    changed,
                )
                recorded = self.module.load_json(immutable)
                current = self.module.load_json(pointer)

            self.assertEqual(immutable.parent.name, commit)
            self.assertEqual(immutable.name, f"{fingerprint}.json")
            self.assertEqual(writes[:2], [immutable, pointer])
            self.assertEqual(first["immutable_result"], second["immutable_result"])
            self.assertFalse(recorded["dependency_cache_hit"])
            self.assertTrue(current["dependency_cache_hit"])

    def test_immutable_result_rejects_different_bundle_for_same_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {"name": "Fixture"},
            )
            payload = {
                "upstream_commit": "a" * 40,
                "fingerprint": "b" * 64,
                "bundle_tree_sha256": "c" * 64,
            }
            with mock.patch.object(self.module, "WORK_ROOT", root / "work"):
                self.module.publish_build_result(
                    manifest,
                    "HEAD",
                    "x86-64",
                    payload["upstream_commit"],
                    payload["fingerprint"],
                    payload,
                )
                with self.assertRaisesRegex(SystemExit, "different bundle contents"):
                    self.module.publish_build_result(
                        manifest,
                        "HEAD",
                        "x86-64",
                        payload["upstream_commit"],
                        payload["fingerprint"],
                        dict(payload, bundle_tree_sha256="d" * 64),
                    )

    def test_packaging_variants_keep_fingerprint_owned_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {"name": "Fixture", "executable": "fixture"},
            )
            commit = "a" * 40

            with mock.patch.object(self.module, "WORK_ROOT", work):
                first = self.publish_fixture_variant(
                    manifest, work, commit, "b" * 64, "variant-a\n"
                )
                second = self.publish_fixture_variant(
                    manifest, work, commit, "c" * 64, "variant-b\n"
                )
                first_record = self.module.load_json(Path(first["immutable_result"]))
                first_bundle = self.module.recorded_build_bundle(
                    manifest, first_record, ref="HEAD", arch="x86-64"
                )
                reused = self.module.publish_build_result(
                    manifest,
                    "HEAD",
                    "x86-64",
                    commit,
                    "b" * 64,
                    first_record,
                )
                current = self.module.load_json(Path(reused["result"]))

            self.assertNotEqual(first["bundle"], second["bundle"])
            self.assertEqual(
                (first_bundle / "bin" / "fixture").read_text(), "variant-a\n"
            )
            self.assertEqual(current["fingerprint"], "b" * 64)

    def test_corrupt_immutable_json_is_quarantined_and_republished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {"name": "Fixture", "executable": "fixture"},
            )
            commit = "a" * 40
            fingerprint = "b" * 64
            with mock.patch.object(self.module, "WORK_ROOT", work):
                initial = self.publish_fixture_variant(
                    manifest, work, commit, fingerprint, "valid\n"
                )
                immutable = Path(initial["immutable_result"])
                artifact = Path(initial["bundle"])
                immutable.write_text("{broken")

                self.assertIsNone(
                    self.module.validated_cached_build_result(
                        manifest,
                        "HEAD",
                        "x86-64",
                        commit,
                        fingerprint,
                        immutable,
                        authoritative=True,
                    )
                )
                self.assertFalse(artifact.exists())
                repaired = self.publish_fixture_variant(
                    manifest, work, commit, fingerprint, "valid\n"
                )

            self.assertTrue(Path(repaired["bundle"]).is_dir())
            self.assertGreaterEqual(len(list((work / "quarantine").iterdir())), 2)

    def test_mutated_immutable_artifact_is_quarantined_and_republished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            manifest = self.module.Manifest(
                root / "Fixture" / "slicer.toml",
                {"name": "Fixture", "executable": "fixture"},
            )
            commit = "a" * 40
            fingerprint = "b" * 64
            with mock.patch.object(self.module, "WORK_ROOT", work):
                initial = self.publish_fixture_variant(
                    manifest, work, commit, fingerprint, "valid\n"
                )
                immutable = Path(initial["immutable_result"])
                artifact = Path(initial["bundle"])
                (artifact / "bin" / "fixture").write_text("mutated\n")

                self.assertIsNone(
                    self.module.validated_cached_build_result(
                        manifest,
                        "HEAD",
                        "x86-64",
                        commit,
                        fingerprint,
                        immutable,
                        authoritative=True,
                    )
                )
                self.assertFalse(artifact.exists())
                repaired = self.publish_fixture_variant(
                    manifest, work, commit, fingerprint, "valid\n"
                )

            self.assertEqual(
                (Path(repaired["bundle"]) / "bin" / "fixture").read_text(),
                "valid\n",
            )

    def test_wrong_immutable_identity_is_quarantined_and_republished(self) -> None:
        for field, wrong_value in (
            ("fingerprint", "c" * 64),
            ("upstream_commit", "d" * 40),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                work = root / "work"
                manifest = self.module.Manifest(
                    root / "Fixture" / "slicer.toml",
                    {"name": "Fixture", "executable": "fixture"},
                )
                commit = "a" * 40
                fingerprint = "b" * 64
                with mock.patch.object(self.module, "WORK_ROOT", work):
                    initial = self.publish_fixture_variant(
                        manifest, work, commit, fingerprint, "valid\n"
                    )
                    immutable = Path(initial["immutable_result"])
                    artifact = Path(initial["bundle"])
                    corrupted = self.module.load_json(immutable)
                    corrupted[field] = wrong_value
                    self.module.atomic_json(immutable, corrupted)

                    self.assertIsNone(
                        self.module.validated_cached_build_result(
                            manifest,
                            "HEAD",
                            "x86-64",
                            commit,
                            fingerprint,
                            immutable,
                            authoritative=True,
                        )
                    )
                    self.assertFalse(artifact.exists())
                    repaired = self.publish_fixture_variant(
                        manifest, work, commit, fingerprint, "valid\n"
                    )

                self.assertTrue(Path(repaired["immutable_result"]).is_file())

    def test_bundle_tree_digest_tracks_files_and_symlink_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            executable = root / "bin" / "slicer"
            executable.write_text("one\n")
            alias = root / "bin" / "slicer-alias"
            alias.symlink_to("slicer")
            initial = self.module.directory_tree_sha256(root)
            executable.write_text("two\n")
            changed_file = self.module.directory_tree_sha256(root)
            alias.unlink()
            alias.symlink_to("missing")
            changed_link = self.module.directory_tree_sha256(root)

        self.assertEqual(len({initial, changed_file, changed_link}), 3)

    def test_bundle_inventory_must_match_the_staged_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "slicer-bundle-report.json"
            report.write_text(
                """{
                  "schema_version": 1,
                  "architecture": "x86-64",
                  "library_count": 4,
                  "bundle_bytes": 100,
                  "bundle_resource_bytes": 40,
                  "resources": {"staged_bytes": 40, "groups": []}
                }""",
                encoding="utf-8",
            )

            inventory = self.module.validated_bundle_inventory(
                report, "x86-64", 100, 40
            )
            self.assertEqual(inventory["library_count"], 4)
            with self.assertRaisesRegex(SystemExit, "architecture mismatch"):
                self.module.validated_bundle_inventory(report, "arm64", 100, 40)
            with self.assertRaisesRegex(SystemExit, "bundle_bytes mismatch"):
                self.module.validated_bundle_inventory(report, "x86-64", 101, 40)

    def test_recorded_bundle_requires_matching_tree_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            (bundle / "bin").mkdir(parents=True)
            executable = bundle / "bin" / "fixture-slicer"
            executable.write_text("original\n")
            manifest = self.module.Manifest(
                Path(temporary) / "Fixture" / "slicer.toml",
                {"name": "Fixture", "executable": "fixture-slicer"},
            )
            result = {
                "slicer": "Fixture",
                "bundle": str(bundle),
                "bundle_tree_sha256": self.module.directory_tree_sha256(bundle),
            }

            self.assertEqual(
                self.module.recorded_build_bundle(manifest, result), bundle
            )
            executable.write_text("mutated\n")
            with self.assertRaisesRegex(SystemExit, "bundle digest mismatch"):
                self.module.recorded_build_bundle(manifest, result)

    def test_recorded_bundle_rejects_missing_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            (bundle / "bin").mkdir(parents=True)
            (bundle / "bin" / "fixture-slicer").write_text("binary\n")
            manifest = self.module.Manifest(
                Path(temporary) / "Fixture" / "slicer.toml",
                {"name": "Fixture", "executable": "fixture-slicer"},
            )

            with self.assertRaisesRegex(SystemExit, "no valid bundle tree digest"):
                self.module.recorded_build_bundle(
                    manifest, {"slicer": "Fixture", "bundle": str(bundle)}
                )

    def test_recorded_bundle_rejects_wrong_build_identity(self) -> None:
        manifest = self.module.Manifest(
            Path("/repo/slicers/Fixture/slicer.toml"),
            {"name": "Fixture", "executable": "fixture-slicer"},
        )
        result = {"slicer": "Other", "ref": "v1", "arch": "x86-64"}

        with self.assertRaisesRegex(SystemExit, "slicer mismatch"):
            self.module.recorded_build_bundle(manifest, result, ref="v1", arch="x86-64")

    def test_corrupt_optional_build_result_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "build-result.json"
            result.write_text("{broken", encoding="utf-8")
            self.assertIsNone(self.module.optional_build_result(result))
            result.write_text("[]", encoding="utf-8")
            self.assertIsNone(self.module.optional_build_result(result))

    def test_skip_dependencies_requires_a_complete_cache_and_marker(self) -> None:
        with self.assertRaisesRegex(SystemExit, "checkout marker toolchain.cmake"):
            self.module.validate_dependency_skip(True, False, "toolchain.cmake")
        self.module.validate_dependency_skip(True, True, "toolchain.cmake")
        self.module.validate_dependency_skip(False, False, None)

    def test_build_holds_literal_ref_lock_around_resolution_and_publication(
        self,
    ) -> None:
        active: list[str] = []

        @contextmanager
        def tracked_lock(key: str):
            active.append(key)
            try:
                yield
            finally:
                active.pop()

        args = argparse.Namespace(
            slicer="OrcaSlicer",
            ref="feature/thumbnail",
            arch="x86-64",
            jobs=None,
        )
        expected = "ref-build:OrcaSlicer:x86-64:feature/thumbnail"

        def locked_build(*_args):
            self.assertEqual(active, [expected])
            return {"locked": True}

        with (
            mock.patch.object(
                self.module, "native_architecture", return_value="x86-64"
            ),
            mock.patch.object(self.module, "file_lock", tracked_lock),
            mock.patch.object(
                self.module, "_build_one_locked", side_effect=locked_build
            ),
        ):
            result = self.module.build_one(args)

        self.assertEqual(result, {"locked": True})
        self.assertEqual(active, [])

    def test_build_environment_override_is_observed(self) -> None:
        with mock.patch.dict(os.environ, {"SLICER_STRIP": "0", "SLICER_PCH": "OFF"}):
            env = self.module.effective_build_environment()
        self.assertEqual(env["SLICER_STRIP"], "0")
        self.assertEqual(env["SLICER_PCH"], "OFF")

    def test_pch_override_is_forwarded_only_to_supported_build_scripts(self) -> None:
        with mock.patch.dict(os.environ, {"SLICER_PCH": "OFF"}, clear=True):
            orca = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )
            prusa = self.module.effective_build_environment(
                self.module.get_manifest("PrusaSlicer")
            )
            bambu = self.module.effective_build_environment(
                self.module.get_manifest("BambuStudio")
            )
            cura = self.module.effective_build_environment(
                self.module.get_manifest("Cura")
            )

        self.assertEqual(orca["SLICER_PCH"], "OFF")
        self.assertEqual(prusa["SLICER_PCH"], "OFF")
        self.assertNotIn("SLICER_PCH", bambu)
        self.assertNotIn("SLICER_PCH", cura)

    def test_resource_allowlist_is_forwarded_but_not_a_build_tree_variant(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            copy_all = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )
        with mock.patch.dict(
            os.environ,
            {"SLICER_RESOURCE_INCLUDES": "shaders/**\ninfo/**"},
            clear=True,
        ):
            allowlisted = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )

        self.assertEqual(allowlisted["SLICER_RESOURCE_INCLUDES"], "shaders/**\ninfo/**")
        self.assertEqual(
            self.module.build_configuration_variant("image", copy_all),
            self.module.build_configuration_variant("image", allowlisted),
        )

    def test_obsolete_wrapper_skip_flags_are_not_forwarded(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SLICER_SKIP_PROFILE_VALIDATOR": "0",
                "SLICER_SKIP_UPSTREAM_PACKAGE": "0",
            },
        ):
            orca = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )
            bambu = self.module.effective_build_environment(
                self.module.get_manifest("BambuStudio")
            )
            prusa = self.module.effective_build_environment(
                self.module.get_manifest("PrusaSlicer")
            )

        self.assertNotIn("SLICER_SKIP_PROFILE_VALIDATOR", orca)
        self.assertNotIn("SLICER_SKIP_UPSTREAM_PACKAGE", orca)
        self.assertNotIn("SLICER_SKIP_PROFILE_VALIDATOR", bambu)
        self.assertNotIn("SLICER_SKIP_UPSTREAM_PACKAGE", bambu)
        self.assertNotIn("SLICER_SKIP_PROFILE_VALIDATOR", prusa)
        self.assertNotIn("SLICER_SKIP_UPSTREAM_PACKAGE", prusa)

    def test_version_date_is_forwarded_to_drivers_that_stamp_it(self) -> None:
        with mock.patch.dict(
            os.environ, {"SLICER_BUILD_DATE": "2026-07-21"}, clear=True
        ):
            bambu = self.module.effective_build_environment(
                self.module.get_manifest("BambuStudio")
            )
            qidi = self.module.effective_build_environment(
                self.module.get_manifest("QIDIStudio")
            )
            prusa = self.module.effective_build_environment(
                self.module.get_manifest("PrusaSlicer")
            )

        self.assertEqual(bambu["SLICER_BUILD_DATE"], "2026-07-21")
        self.assertEqual(qidi["SLICER_BUILD_DATE"], "2026-07-21")
        self.assertNotIn("SLICER_BUILD_DATE", prusa)

    def test_upstream_extra_build_arguments_are_scoped_and_fingerprinted(self) -> None:
        environment = {
            "ORCA_EXTRA_BUILD_ARGS": "-DFEATURE=ON",
            "ELEGOO_EXTRA_BUILD_ARGS": "-DELEGOO_FEATURE=ON",
            "ORCA_UPDATER_SIG_KEY": "fixture-key",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            anycubic = self.module.effective_build_environment(
                self.module.get_manifest("AnycubicSlicerNext")
            )
            elegoo = self.module.effective_build_environment(
                self.module.get_manifest("ElegooSlicer")
            )
            orca = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )
            qidi = self.module.effective_build_environment(
                self.module.get_manifest("QIDIStudio")
            )

        self.assertEqual(anycubic["ORCA_EXTRA_BUILD_ARGS"], "-DFEATURE=ON")
        self.assertEqual(orca["ORCA_EXTRA_BUILD_ARGS"], "-DFEATURE=ON")
        self.assertEqual(elegoo["ELEGOO_EXTRA_BUILD_ARGS"], "-DELEGOO_FEATURE=ON")
        self.assertEqual(elegoo["ORCA_UPDATER_SIG_KEY"], "fixture-key")
        self.assertNotIn("ELEGOO_EXTRA_BUILD_ARGS", orca)
        self.assertNotIn("ORCA_EXTRA_BUILD_ARGS", elegoo)
        self.assertNotIn("ORCA_UPDATER_SIG_KEY", qidi)

        changed = dict(orca)
        changed["ORCA_EXTRA_BUILD_ARGS"] = "-DFEATURE=OFF"
        self.assertNotEqual(
            self.module.build_configuration_variant("image", orca),
            self.module.build_configuration_variant("image", changed),
        )

    def test_version_date_slicer_set_is_explicit(self) -> None:
        self.assertEqual(
            self.module.VERSION_DATE_STAMP_SLICERS,
            {
                "AnycubicSlicerNext",
                "BambuStudio",
                "CrealityPrint",
                "ElegooSlicer",
                "OrcaSlicer",
                "QIDIStudio",
            },
        )

    def test_cura_environment_is_scoped_to_cura(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "CURA_CONAN_CONFIG_INSTALL": "1",
                "CURA_CONAN_CONFIG_REF": "1" * 40,
                "CURA_CONAN_CONFIG_URL": "https://example.invalid/config.git",
            },
        ):
            cura = self.module.effective_build_environment(
                self.module.get_manifest("Cura")
            )
            orca = self.module.effective_build_environment(
                self.module.get_manifest("OrcaSlicer")
            )
        self.assertEqual(cura["CURA_CONAN_CONFIG_INSTALL"], "1")
        self.assertEqual(cura["CURA_CONAN_CONFIG_REF"], "1" * 40)
        self.assertNotIn("CURA_CONAN_CONFIG_REF", orca)
        self.assertNotIn("CURA_CONAN_CONFIG_INSTALL", orca)
        self.assertNotIn("CURA_CONAN_CONFIG_URL", orca)
        self.assertNotIn("CMAKE_GENERATOR", orca)
        self.assertNotIn("SLICER_GUI", orca)

    def test_cura_manifest_provides_pinned_conan_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            env = self.module.effective_build_environment(
                self.module.get_manifest("Cura")
            )
        self.assertEqual(
            env["CURA_CONAN_CONFIG_REF"],
            "64a4bebfd76b4366c8156e68832252fd024ab704",
        )
        self.assertEqual(
            env["CURA_CONAN_CONFIG_URL"],
            "https://github.com/Ultimaker/conan-config.git",
        )

    def test_symbolic_git_ref_resolution_tracks_branch_and_peels_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "commit.gpgsign", "false"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "tag.gpgsign", "false"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "first"],
                cwd=repository,
                check=True,
            )
            first = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            subprocess.run(
                ["git", "tag", "-a", "v1", "-m", "one"],
                cwd=repository,
                check=True,
            )

            self.assertEqual(
                self.module.resolve_remote_git_ref(str(repository), branch), first
            )
            self.assertEqual(
                self.module.resolve_remote_git_ref(str(repository), "v1"), first
            )

            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "second"],
                cwd=repository,
                check=True,
            )
            second = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(
                self.module.resolve_remote_git_ref(str(repository), branch), second
            )
            self.assertEqual(
                self.module.resolve_remote_git_ref(str(repository), "v1"), first
            )

    def test_podman_run_options_keep_host_identity(self) -> None:
        with mock.patch.object(self.module, "is_podman_cli", return_value=True):
            options = self.module.container_run_identity_options("podman")
        self.assertIn("--userns=keep-id", options)
        self.assertIn("label=disable", options)
        self.assertEqual(options.count("--user"), 1)

    def test_patch_state_tracks_add_delete_and_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            upstream.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.com"],
                cwd=upstream,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"],
                cwd=upstream,
                check=True,
            )
            subprocess.run(
                ["git", "config", "commit.gpgsign", "false"],
                cwd=upstream,
                check=True,
            )
            (upstream / "old.txt").write_text("rename me\n")
            (upstream / "delete.txt").write_text("delete me\n")
            subprocess.run(["git", "add", "."], cwd=upstream, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=upstream, check=True
            )

            edited = root / "edited"
            source = root / "source"
            subprocess.run(["git", "clone", "-q", upstream, edited], check=True)
            subprocess.run(["git", "clone", "-q", upstream, source], check=True)
            (edited / "old.txt").rename(edited / "new.txt")
            (edited / "delete.txt").unlink()
            (edited / "added.txt").write_text("added\n")
            subprocess.run(["git", "add", "-A"], cwd=edited, check=True)
            patch = root / "changes.patch"
            with patch.open("wb") as output:
                subprocess.run(
                    ["git", "diff", "--cached", "--binary", "HEAD"],
                    cwd=edited,
                    check=True,
                    stdout=output,
                )

            self.assertEqual(
                self.module.patch_touched_paths([patch]),
                {"added.txt", "delete.txt", "new.txt", "old.txt"},
            )
            self.module.apply_legacy_patches(source, [patch])
            initial_state = self.module.source_state(source, [patch])
            self.module.apply_legacy_patches(source, [patch])
            self.assertEqual(initial_state, self.module.source_state(source, [patch]))

            (source / "added.txt").write_text("mutated\n")
            with self.assertRaises(SystemExit):
                self.module.source_state(source, [patch])
            self.module.restore_managed_worktree(source)
            self.assertEqual(initial_state, self.module.source_state(source, [patch]))

            (source / "delete.txt").write_text("recreated\n")
            with self.assertRaises(SystemExit):
                self.module.source_state(source, [patch])

    def test_gui_mode_separates_dependency_cache(self) -> None:
        manifest = self.module.get_manifest("PrusaSlicer")
        env = self.module.effective_build_environment()
        gui_env = dict(env, SLICER_GUI="1")
        with mock.patch.object(self.module, "_hash_index_paths"):
            cli = self.module.dependency_fingerprint(
                manifest, ROOT, "x86-64", "sha256:image", env
            )
            gui = self.module.dependency_fingerprint(
                manifest, ROOT, "x86-64", "sha256:image", gui_env
            )
        self.assertNotEqual(cli, gui)

    def test_build_all_continues_and_reports_failures(self) -> None:
        args = argparse.Namespace(
            slicers=None,
            arch="x86-64",
            fail_fast=False,
            all_refs=False,
            ref=None,
            source=None,
        )
        with (
            mock.patch.object(
                self.module,
                "selected_manifest_targets",
                return_value=[("One", "HEAD"), ("Two", "HEAD")],
            ),
            mock.patch.object(
                self.module,
                "build_one",
                side_effect=[subprocess.CalledProcessError(2, ["false"]), {}],
            ),
            self.assertRaises(SystemExit),
        ):
            self.module.command_build_all(args)

    def test_parallel_collection_cancels_pending_work_on_interrupt(self) -> None:
        class Future:
            def __init__(self, interrupt: bool = False) -> None:
                self.interrupt = interrupt
                self.cancelled = False

            def result(self):
                if self.interrupt:
                    raise KeyboardInterrupt
                return ("value", None)

            def cancel(self) -> None:
                self.cancelled = True

        class Executor:
            def __init__(self) -> None:
                self.futures: list[Future] = []
                self.shutdown_calls: list[dict[str, bool]] = []

            def submit(self, _action, name: str) -> Future:
                future = Future(interrupt=name == "first")
                self.futures.append(future)
                return future

            def shutdown(self, **kwargs: bool) -> None:
                self.shutdown_calls.append(kwargs)

        executor = Executor()
        with (
            mock.patch.object(
                self.module, "ThreadPoolExecutor", return_value=executor
            ),
            mock.patch.object(
                self.module, "as_completed", return_value=iter(executor.futures)
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            # as_completed needs the futures created by submit, so expose a
            # lazy iterator rather than taking a snapshot in the mock above.
            self.module.run_many_collect(
                ["first", "second", "pending"], 2, False, lambda _name: None
            )

        self.assertTrue(all(future.cancelled for future in executor.futures))
        self.assertEqual(
            executor.shutdown_calls,
            [{"wait": False, "cancel_futures": True}],
        )

    def test_fail_fast_build_all_keeps_the_full_per_build_job_budget(self) -> None:
        args = argparse.Namespace(
            slicers=None,
            arch="x86-64",
            fail_fast=True,
            all_refs=False,
            ref=None,
            source=None,
            workers=4,
            jobs=None,
        )
        observed_jobs: list[int | None] = []

        def capture_build(child):
            observed_jobs.append(child.jobs)
            return {}

        with (
            mock.patch.object(
                self.module,
                "selected_manifest_targets",
                return_value=[("One", "HEAD"), ("Two", "HEAD")],
            ),
            mock.patch.object(self.module, "default_build_jobs", return_value=16),
            mock.patch.object(self.module, "build_one", side_effect=capture_build),
            redirect_stdout(io.StringIO()),
        ):
            self.module.command_build_all(args)

        self.assertEqual(observed_jobs, [None, None])


if __name__ == "__main__":
    unittest.main()
