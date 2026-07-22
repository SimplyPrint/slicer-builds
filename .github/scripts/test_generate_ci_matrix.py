#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_ci_matrix import MatrixError, generate_matrix  # noqa: E402


class CurrentRepositoryPolicyTests(unittest.TestCase):
    def test_binary_matrix_preserves_current_x86_jobs(self) -> None:
        rows = generate_matrix(REPO_ROOT, "binary", selected_arch="x86-64")["include"]
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {row["slicer"] for row in rows},
            {
                "BambuStudio",
                "CrealityPrint",
                "Cura",
                "ElegooSlicer",
                "OrcaSlicer",
                "PrusaSlicer",
                "SuperSlicer",
            },
        )
        self.assertEqual(
            {(row["slicer"], row["build-type"]) for row in rows},
            {
                (slicer, build_type)
                for slicer in {
                    "BambuStudio",
                    "CrealityPrint",
                    "ElegooSlicer",
                    "OrcaSlicer",
                    "PrusaSlicer",
                }
                for build_type in {"nightly", "latest_release"}
            }
            | {
                ("Cura", "latest_release"),
                ("SuperSlicer", "latest_release"),
            },
        )
        self.assertTrue(all(row["publish"] for row in rows))
        self.assertEqual(
            {row["family"] for row in rows}, {"bambu", "cura", "orca", "prusa"}
        )
        cura = [row for row in rows if row["slicer"] == "Cura"]
        self.assertEqual([row["build-type"] for row in cura], ["latest_release"])
        self.assertEqual(cura[0]["release_tag"], "5.13.0")
        self.assertEqual(
            cura[0]["expected_commit"],
            "636f9608621fd1d6c1b75ad2a969de53beabfbc6",
        )
        self.assertFalse(cura[0]["nightly_enabled"])

        dynamic_rows = [
            row
            for row in rows
            if row["slicer"] == "OrcaSlicer" and row["build-type"] == "latest_release"
        ]
        self.assertEqual(dynamic_rows[0]["expected_commit"], "")

    def test_binary_all_architectures_only_adds_supported_arm_jobs(self) -> None:
        rows = generate_matrix(REPO_ROOT, "binary", selected_arch="all")["include"]
        arm_slicers = {row["slicer"] for row in rows if row["arch"] == "arm64"}
        self.assertEqual(arm_slicers, {"BambuStudio", "OrcaSlicer"})
        self.assertEqual(len(rows), 16)

    def test_consumer_matrix_rebuilds_the_downloaded_version_window(self) -> None:
        rows = generate_matrix(
            REPO_ROOT,
            "binary",
            selected_build_type="consumer",
            selected_arch="x86-64",
        )["include"]
        stable = [row for row in rows if row["build-type"] == "latest_release"]
        self.assertEqual(len(rows), 22)
        self.assertEqual(len(stable), 17)
        self.assertTrue(all(row["consumer_rebuild"] for row in rows))
        self.assertEqual(
            [row["release_tag"] for row in stable if row["slicer"] == "OrcaSlicer"],
            ["v2.4.2", "v2.4.1", "v2.4.0"],
        )

    def test_config_matrix_preserves_pins_and_cura_override(self) -> None:
        rows = generate_matrix(REPO_ROOT, "config")["include"]
        self.assertEqual(len(rows), 11)
        by_slicer: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_slicer.setdefault(str(row["slicer"]), []).append(row)

        self.assertTrue(all(row["publish"] for row in rows))

        self.assertEqual(by_slicer["BambuStudio"][0]["release_tag"], "v02.05.00.67")
        self.assertEqual(by_slicer["PrusaSlicer"][0]["release_tag"], "version_2.9.5")
        prusa_release = next(
            row
            for row in by_slicer["PrusaSlicer"]
            if row["build-type"] == "latest_release"
        )
        self.assertEqual(
            prusa_release["expected_commit"],
            "9a583bd438b195856f3bcf7ea99b69ba4003a961",
        )
        self.assertTrue(by_slicer["PrusaSlicer"][0]["gui"])
        self.assertEqual(by_slicer["Cura"][0]["repo"], "Ultimaker/Cura")
        self.assertEqual(by_slicer["Cura"][0]["expected_commit"], "")
        self.assertTrue(by_slicer["Cura"][0]["config_generator_only"])
        self.assertEqual(
            [row["build-type"] for row in by_slicer["Cura"]], ["latest_release"]
        )

    def test_disabled_and_unknown_slicers_fail_clearly(self) -> None:
        with self.assertRaisesRegex(MatrixError, "enabled=false"):
            generate_matrix(REPO_ROOT, "binary", selected_slicer="AnycubicSlicerNext")
        with self.assertRaisesRegex(MatrixError, "unknown slicer"):
            generate_matrix(REPO_ROOT, "binary", selected_slicer="DoesNotExist")

    def test_release_override_requires_one_slicer(self) -> None:
        with self.assertRaisesRegex(MatrixError, "exactly one slicer"):
            generate_matrix(REPO_ROOT, "binary", release_tag_override="v1")

    def test_unsafe_release_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(MatrixError, "unsafe"):
            generate_matrix(
                REPO_ROOT,
                "binary",
                selected_slicer="OrcaSlicer",
                release_tag_override="$(echo injected)",
            )

    def test_release_override_uses_declared_commit_lock(self) -> None:
        rows = generate_matrix(
            REPO_ROOT,
            "binary",
            selected_slicer="OrcaSlicer",
            selected_build_type="latest_release",
            selected_arch="x86-64",
            release_tag_override="v2.4.1",
        )["include"]
        self.assertEqual(
            rows[0]["expected_commit"],
            "19db9aa9c3f6720bceaff2d4c9c377362c440f4f",
        )

    def test_unsupported_filter_fails_before_build(self) -> None:
        with self.assertRaisesRegex(MatrixError, "no enabled"):
            generate_matrix(
                REPO_ROOT,
                "binary",
                selected_slicer="Cura",
                selected_build_type="nightly",
            )


class PreviewPolicyTests(unittest.TestCase):
    def _write_manifest(self, root: Path, ci: str) -> None:
        manifest_dir = root / "slicers" / "PreviewSlicer"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "slicer.toml").write_text(
            """\
schema_version = 1
name = "PreviewSlicer"
repository = "https://github.com/example/preview.git"
default_ref = "HEAD"
family = "orca"
executable = "preview"
architectures = ["x86-64"]

[capabilities]
binary = true
config_dump = true
thumbnail = true

"""
            + ci,
            encoding="utf-8",
        )

    def test_unpublished_head_preview_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_manifest(
                root,
                """
[ci.binary]
enabled = true
publish = false
release = "none"
nightly = true

[ci.config]
enabled = false
publish = false
release = "none"
nightly = false
""",
            )
            rows = generate_matrix(root, "binary")["include"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["release_mode"], "none")
            self.assertEqual(rows[0]["family"], "orca")
            self.assertFalse(rows[0]["publish"])

    def test_publishing_without_release_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_manifest(
                root,
                """
[ci.binary]
enabled = true
publish = true
release = "none"
nightly = true

[ci.config]
enabled = false
publish = false
release = "none"
nightly = false
""",
            )
            with self.assertRaisesRegex(
                MatrixError, "cannot publish without a release"
            ):
                generate_matrix(root, "binary")


if __name__ == "__main__":
    unittest.main()
