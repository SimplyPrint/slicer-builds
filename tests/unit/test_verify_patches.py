from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_controller_module():
    path = ROOT / "tools" / "slicerctl.py"
    spec = importlib.util.spec_from_file_location("slicerctl_verify_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(cwd: Path, *arguments: str, capture: bool = False) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout if capture else ""


class VerifyPatchStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_controller_module()

    def test_ordered_stack_handles_dependent_and_structural_patches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            git(source, "init", "-q")
            git(source, "config", "user.email", "fixture@example.com")
            git(source, "config", "user.name", "Fixture")
            git(source, "config", "commit.gpgsign", "false")
            (source / "value.txt").write_text("zero\n")
            (source / "delete.txt").write_text("delete me\n")
            (source / "rename-old.txt").write_text("rename me\n")
            (source / "binary.bin").write_bytes(b"\x00\x01\x02")
            git(source, "add", ".")
            git(source, "commit", "-qm", "base")
            base_commit = git(source, "rev-parse", "HEAD", capture=True).strip()

            (source / "value.txt").write_text("one\n")
            first_patch = root / "01-value.patch"
            first_patch.write_text(git(source, "diff", "--binary", capture=True))
            git(source, "add", "value.txt")
            git(source, "commit", "-qm", "first patch state")

            (source / "delete.txt").unlink()
            (source / "rename-old.txt").rename(source / "rename-new.txt")
            (source / "binary.bin").write_bytes(b"\x00\x09\x08\x07")
            (source / "added.txt").write_text("added\n")
            (source / "value.txt").write_text("two\n")
            git(source, "add", "-A")
            second_patch = root / "02-structural.patch"
            second_patch.write_text(
                git(
                    source,
                    "diff",
                    "--cached",
                    "--binary",
                    "--find-renames",
                    capture=True,
                )
            )

            mirror = root / "mirror.git"
            git(root, "clone", "-q", "--mirror", str(source), str(mirror))
            self.assertIsNone(
                self.module.verify_patch_stack(
                    mirror, base_commit, [first_patch, second_patch]
                )
            )
            failure = self.module.verify_patch_stack(
                mirror, base_commit, [second_patch, first_patch]
            )

            self.assertIsNotNone(failure)
            assert failure is not None
            self.assertEqual(failure["stage"], "check")
            self.assertEqual(failure["patch"], str(second_patch))
            self.assertFalse((mirror / "index").exists())

    def test_cached_mirror_refreshes_head_after_default_branch_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            git(source, "init", "-q", "-b", "main")
            git(source, "config", "user.email", "fixture@example.com")
            git(source, "config", "user.name", "Fixture")
            git(source, "config", "commit.gpgsign", "false")
            (source / "value.txt").write_text("main\n")
            git(source, "add", "value.txt")
            git(source, "commit", "-qm", "main")
            main_commit = git(source, "rev-parse", "HEAD", capture=True).strip()
            git(source, "tag", "-a", "--no-sign", "v1", "-m", "annotated release")

            git(source, "switch", "-qc", "next")
            (source / "value.txt").write_text("next\n")
            git(source, "commit", "-qam", "next")
            next_commit = git(source, "rev-parse", "HEAD", capture=True).strip()
            git(source, "switch", "-q", "main")

            upstream = root / "upstream.git"
            git(root, "clone", "-q", "--bare", str(source), str(upstream))
            git(upstream, "symbolic-ref", "HEAD", "refs/heads/main")
            manifest_dir = root / "slicers" / "Fixture"
            manifest_dir.mkdir(parents=True)
            manifest = self.module.Manifest(
                manifest_dir / "slicer.toml",
                {
                    "name": "Fixture",
                    "repository": upstream.as_uri(),
                    "default_ref": "HEAD",
                    "capabilities": {"binary": True, "config_dump": True},
                },
            )

            with mock.patch.object(self.module, "WORK_ROOT", root / "work"):
                mirror = self.module.prepare_patch_verification_mirror(manifest)
                first, error = self.module.resolve_patch_verification_ref(
                    mirror, "HEAD"
                )
                self.assertEqual(error, "")
                self.assertEqual(first, main_commit)
                annotated, error = self.module.resolve_patch_verification_ref(
                    mirror, "v1"
                )
                self.assertEqual(error, "")
                self.assertEqual(annotated, main_commit)

                git(upstream, "symbolic-ref", "HEAD", "refs/heads/next")
                refreshed = self.module.prepare_patch_verification_mirror(manifest)
                second, error = self.module.resolve_patch_verification_ref(
                    refreshed, "HEAD"
                )

            self.assertEqual(refreshed, mirror)
            self.assertEqual(error, "")
            self.assertEqual(second, next_commit)
            self.assertEqual(
                git(mirror, "symbolic-ref", "HEAD", capture=True).strip(),
                "refs/heads/next",
            )


class VerifyPatchesCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_controller_module()

    def fixture_manifest(self, root: Path, refs: list[dict[str, str]] | None = None):
        slicer = root / "slicers" / "Fixture"
        slicer.mkdir(parents=True)
        return self.module.Manifest(
            slicer / "slicer.toml",
            {
                "name": "Fixture",
                "repository": "https://example.invalid/fixture.git",
                "default_ref": "HEAD",
                "supported_refs": refs or [],
                "capabilities": {"binary": True, "config_dump": True},
            },
        )

    def run_json(self, args, *, expect_failure: bool = False):
        output = io.StringIO()
        with redirect_stdout(output):
            if expect_failure:
                with self.assertRaisesRegex(SystemExit, "1"):
                    args.handler(args)
            else:
                args.handler(args)
        return json.loads(output.getvalue())

    def test_defaults_check_every_ref_and_mode_with_one_resolution_per_ref(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture_manifest(
                root, [{"ref": "v1", "expected_commit": "b" * 40}]
            )
            nightly = manifest.directory / "patches" / "binary" / "nightly"
            nightly.mkdir(parents=True)
            (nightly / "head.patch").write_text("head")
            (manifest.directory / "patches" / "dump_configs.patch").write_text("dump")
            args = self.module.parser().parse_args(
                ["verify-patches", "--slicer", "Fixture", "--json"]
            )

            def resolve(_mirror: Path, ref: str):
                return (("a" if ref == "HEAD" else "b") * 40, "")

            with (
                mock.patch.object(self.module, "ROOT", root),
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "prepare_patch_verification_mirror",
                    return_value=Path("/mirror"),
                ) as prepare,
                mock.patch.object(
                    self.module,
                    "resolve_patch_verification_ref",
                    side_effect=resolve,
                ) as resolve_ref,
                mock.patch.object(
                    self.module, "verify_patch_stack", return_value=None
                ) as verify,
            ):
                payload = self.run_json(args)

            prepare.assert_called_once_with(manifest, None)
            self.assertEqual(
                [call.args[1] for call in resolve_ref.call_args_list], ["HEAD", "v1"]
            )
            self.assertEqual(verify.call_count, 4)
            head_binary = next(
                result
                for result in payload["results"]
                if result["ref"] == "HEAD" and result["mode"] == "binary"
            )
            self.assertEqual(
                head_binary["patches"],
                ["slicers/Fixture/patches/binary/nightly/head.patch"],
            )
            self.assertEqual(payload["summary"]["planned_stacks"], 4)
            self.assertTrue(payload["ok"])

    def test_defaults_skip_modes_disabled_by_selected_slicer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture_manifest(root)
            manifest.data["capabilities"]["config_dump"] = False
            args = self.module.parser().parse_args(
                ["verify-patches", "--slicer", "Fixture", "--json"]
            )

            with (
                mock.patch.object(self.module, "ROOT", root),
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "prepare_patch_verification_mirror",
                    return_value=Path("/mirror"),
                ),
                mock.patch.object(
                    self.module,
                    "resolve_patch_verification_ref",
                    return_value=("a" * 40, ""),
                ),
                mock.patch.object(self.module, "verify_patch_stack", return_value=None),
            ):
                payload = self.run_json(args)

            self.assertEqual(
                [result["mode"] for result in payload["results"]], ["binary"]
            )
            self.assertTrue(payload["ok"])

    def test_include_head_uses_nightly_stack_and_resolves_head_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slicer = root / "slicers" / "ReleaseFixture"
            nightly = slicer / "patches" / "binary" / "nightly"
            nightly.mkdir(parents=True)
            (nightly / "nightly.patch").write_text("nightly")
            manifest = self.module.Manifest(
                slicer / "slicer.toml",
                {
                    "name": "ReleaseFixture",
                    "repository": "https://example.invalid/release.git",
                    "default_ref": "v1",
                    "expected_commit": "a" * 40,
                    "capabilities": {"binary": True, "config_dump": True},
                    "ci": {
                        "binary": {"enabled": True, "nightly": True},
                        "config": {"enabled": False, "nightly": True},
                    },
                },
            )
            args = self.module.parser().parse_args(
                [
                    "verify-patches",
                    "--slicer",
                    "ReleaseFixture",
                    "--include-head",
                    "--json",
                ]
            )

            def resolve(_mirror: Path, ref: str):
                return (("a" if ref == "v1" else "b") * 40, "")

            with (
                mock.patch.object(self.module, "ROOT", root),
                mock.patch.object(
                    self.module,
                    "manifests",
                    return_value={"ReleaseFixture": manifest},
                ),
                mock.patch.object(
                    self.module,
                    "prepare_patch_verification_mirror",
                    return_value=Path("/mirror"),
                ),
                mock.patch.object(
                    self.module,
                    "resolve_patch_verification_ref",
                    side_effect=resolve,
                ) as resolve_ref,
                mock.patch.object(self.module, "verify_patch_stack", return_value=None),
            ):
                payload = self.run_json(args)

            self.assertEqual(
                [call.args[1] for call in resolve_ref.call_args_list], ["v1", "HEAD"]
            )
            head_binary = next(
                result
                for result in payload["results"]
                if result["ref"] == "HEAD" and result["mode"] == "binary"
            )
            self.assertEqual(
                head_binary["patches"],
                ["slicers/ReleaseFixture/patches/binary/nightly/nightly.patch"],
            )

    def test_include_head_policy_ignores_dormant_nightly_lanes(self) -> None:
        included = {
            name
            for name, manifest in self.module.manifests().items()
            if any(
                spec["ref"] == "HEAD"
                for spec in self.module.patch_verification_ref_specs(
                    manifest, include_head=True
                )
            )
        }
        self.assertEqual(
            included,
            {
                "AnycubicSlicerNext",
                "BambuStudio",
                "CrealityPrint",
                "ElegooSlicer",
                "OrcaSlicer",
                "PrusaSlicer",
            },
        )

    def test_expected_commit_failure_skips_apply_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture_manifest(
                root,
                [
                    {"ref": "locked", "expected_commit": "b" * 40},
                    {"ref": "good", "expected_commit": "c" * 40},
                ],
            )
            args = self.module.parser().parse_args(
                ["verify-patches", "--slicer", "Fixture", "--json"]
            )

            commits = {"HEAD": "a" * 40, "locked": "d" * 40, "good": "c" * 40}
            with (
                mock.patch.object(self.module, "ROOT", root),
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "prepare_patch_verification_mirror",
                    return_value=Path("/mirror"),
                ),
                mock.patch.object(
                    self.module,
                    "resolve_patch_verification_ref",
                    side_effect=lambda _mirror, ref: (commits[ref], ""),
                ),
                mock.patch.object(
                    self.module, "verify_patch_stack", return_value=None
                ) as verify,
            ):
                payload = self.run_json(args, expect_failure=True)

            self.assertEqual(payload["summary"]["planned_stacks"], 6)
            self.assertEqual(payload["summary"]["failed_stacks"], 2)
            self.assertEqual(payload["summary"]["passed_stacks"], 4)
            self.assertEqual(verify.call_count, 4)
            locked = [r for r in payload["results"] if r["ref"] == "locked"]
            self.assertTrue(all(r["stage"] == "expected-commit" for r in locked))

    def test_fail_fast_stops_after_first_locked_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture_manifest(
                root, [{"ref": "locked", "expected_commit": "b" * 40}]
            )
            # Select only the mismatching ref so the unlocked default does not
            # precede it in manifest order.
            args = self.module.parser().parse_args(
                [
                    "verify-patches",
                    "--slicer",
                    "Fixture",
                    "--ref",
                    "locked",
                    "--fail-fast",
                    "--json",
                ]
            )
            with (
                mock.patch.object(self.module, "ROOT", root),
                mock.patch.object(
                    self.module, "manifests", return_value={"Fixture": manifest}
                ),
                mock.patch.object(
                    self.module,
                    "prepare_patch_verification_mirror",
                    return_value=Path("/mirror"),
                ),
                mock.patch.object(
                    self.module,
                    "resolve_patch_verification_ref",
                    return_value=("d" * 40, ""),
                ),
                mock.patch.object(self.module, "verify_patch_stack") as verify,
            ):
                payload = self.run_json(args, expect_failure=True)

            self.assertEqual(payload["summary"]["planned_stacks"], 2)
            self.assertEqual(payload["summary"]["attempted_stacks"], 1)
            self.assertTrue(payload["stopped_early"])
            verify.assert_not_called()

    def test_source_requires_exactly_one_slicer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.fixture_manifest(root)
            second_path = root / "slicers" / "Second"
            second_path.mkdir()
            second = self.module.Manifest(
                second_path / "slicer.toml",
                {
                    **first.data,
                    "name": "Second",
                },
            )
            args = self.module.parser().parse_args(
                ["verify-patches", "--source", str(root)]
            )
            with mock.patch.object(
                self.module,
                "manifests",
                return_value={"Fixture": first, "Second": second},
            ):
                with self.assertRaisesRegex(
                    SystemExit, "--source requires exactly one --slicer"
                ):
                    args.handler(args)

    def test_parser_accepts_repeatable_selectors(self) -> None:
        args = self.module.parser().parse_args(
            [
                "verify-patches",
                "--slicer",
                "OrcaSlicer",
                "--ref",
                "v2.4.2",
                "--mode",
                "dump",
                "--json",
            ]
        )
        self.assertEqual(args.slicers, ["OrcaSlicer"])
        self.assertEqual(args.refs, ["v2.4.2"])
        self.assertEqual(args.modes, ["dump"])
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()
