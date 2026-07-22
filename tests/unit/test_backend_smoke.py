from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_smoke_module():
    path = ROOT / "tests" / "integration" / "backend_smoke.py"
    spec = importlib.util.spec_from_file_location("backend_smoke_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(payload, zlib.crc32(kind))
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def rgb_png(width: int = 64, height: int = 64, uniform: bool = False) -> bytes:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            pixel = (
                (32, 64, 96)
                if uniform
                else (x * 3 % 256, y * 5 % 256, (x + y) * 7 % 256)
            )
            rows.extend(pixel)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows)))
        + png_chunk(b"IEND", b"")
    )


def rgba_png_with_hidden_color(width: int = 64, height: int = 64) -> bytes:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend((x * 3 % 256, y * 5 % 256, (x + y) * 7 % 256))
            rows.append(255 if x == 0 and y == 0 else 0)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows)))
        + png_chunk(b"IEND", b"")
    )


class BackendSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_smoke_module()

    def test_inherited_profile_flattens_parent_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.json").write_text('{"speed":"20","temperature":"210"}\n')
            (root / "child.json").write_text(
                '{"inherits":"base","name":"child","speed":"40"}\n'
            )
            resolved = self.module.inherited_profile(root / "child.json")
        self.assertEqual(
            resolved,
            {"name": "child", "speed": "40", "temperature": "210"},
        )

    def test_frontend_bambu_fixture_has_no_reusable_thumbnail(self) -> None:
        fixture = ROOT / "tests/integration/fixtures/calicat-bambu-v1.3mf"
        with zipfile.ZipFile(fixture) as archive:
            members = {name for name in archive.namelist() if not name.endswith("/")}
            self.assertEqual(
                members,
                {
                    "3D/3dmodel.model",
                    "Metadata/model_settings.config",
                    "[Content_Types].xml",
                    "_rels/.rels",
                },
            )
            config = archive.read("Metadata/model_settings.config")
            self.assertNotIn(b"thumbnail_file", config)
            self.assertFalse(any(name.lower().endswith(".png") for name in members))

    def test_inherited_profile_rejects_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.json").write_text('{"inherits":"two"}\n')
            (root / "two.json").write_text('{"inherits":"one"}\n')
            with self.assertRaisesRegex(AssertionError, "inheritance cycle"):
                self.module.inherited_profile(root / "one.json")

    def test_inherited_profile_falls_back_to_sibling_json_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "QIDI PLA Rapido.json").write_text(
                '{"name":"QIDI PLA Rapido@Q-Series","temperature":"220"}\n'
            )
            (root / "printer-specific.json").write_text(
                '{"inherits":"QIDI PLA Rapido@Q-Series",'
                '"name":"QIDI PLA Rapido @Q1","flow":"0.98"}\n'
            )
            resolved = self.module.inherited_profile(root / "printer-specific.json")

        self.assertEqual(
            resolved,
            {
                "name": "QIDI PLA Rapido @Q1",
                "temperature": "220",
                "flow": "0.98",
            },
        )

    def test_inherited_profile_rejects_duplicate_sibling_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first.json").write_text('{"name":"shared"}\n')
            (root / "second.json").write_text('{"name":"shared"}\n')
            (root / "child.json").write_text('{"inherits":"shared","name":"child"}\n')
            with self.assertRaisesRegex(AssertionError, "Duplicate sibling"):
                self.module.inherited_profile(root / "child.json")

    def test_native_source_id_is_narrowed_to_machine_variant(self) -> None:
        records = [
            {
                "context": {
                    "native_id": "generic_pla#resolved-02",
                    "source_native_id": "generic_pla",
                },
                "compatible_printers": {"Fixture Printer": ["0.2"]},
            },
            {
                "context": {
                    "native_id": "generic_pla#resolved-04",
                    "source_native_id": "generic_pla",
                },
                "compatible_printers": {"Fixture Printer": ["0.4"]},
            },
        ]

        selected = self.module.require_one(
            records,
            "native_id",
            "generic_pla",
            machine_name="Fixture Printer",
            machine_variant="0.4",
        )

        self.assertEqual(selected["context"]["native_id"], "generic_pla#resolved-04")

    def test_profile_selector_rejects_wrong_machine_variant(self) -> None:
        records = [
            {
                "name": "Fixture PLA",
                "compatible_printers": {"Fixture Printer": ["0.2"]},
            }
        ]

        with self.assertRaisesRegex(AssertionError, "variant '0.4'.*found 0"):
            self.module.require_one(
                records,
                "name",
                "Fixture PLA",
                machine_name="Fixture Printer",
                machine_variant="0.4",
            )

    def test_cura_runtime_wrapper_declares_consumer_transport(self) -> None:
        wrapped = self.module.runtime_wrapper(
            {
                "name": "Fixture Cura profile",
                "data": {"layer_height": 0.2},
                "context": {"native_id": "fixture"},
                "setting_scopes": {"layer_height": "global"},
            }
        )

        self.assertEqual(wrapped["transport"], "envelope.v1")
        self.assertEqual(wrapped["data"], {"layer_height": 0.2})

    def test_thumbnail_archive_requires_decodable_nonuniform_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "result.3mf"
            config = (
                '<config><plate><metadata key="thumbnail_file" '
                'value="plate_1.png"/></plate></config>'
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Metadata/model_settings.config", config)
                archive.writestr("Metadata/plate_1.png", rgb_png())
            report = self.module.validate_thumbnails(archive_path)
            self.assertEqual(report["thumbnail_file"]["width"], 64)
            self.assertEqual(report["thumbnail_file"]["height"], 64)

    def test_thumbnail_archive_allows_structurally_valid_auxiliary_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "result.3mf"
            config = (
                '<config><plate><metadata key="thumbnail_file" '
                'value="plate_1.png"/><metadata key="pick_file" '
                'value="pick_1.png"/></plate></config>'
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Metadata/model_settings.config", config)
                archive.writestr("Metadata/plate_1.png", rgb_png())
                archive.writestr(
                    "Metadata/pick_1.png", rgb_png(uniform=True)
                )

            report = self.module.validate_thumbnails(archive_path)

        self.assertEqual(report["thumbnail_file"]["width"], 64)
        self.assertEqual(report["pick_file"]["height"], 64)

    def test_uniform_thumbnail_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "blank or nearly uniform"):
            self.module.validate_png(rgb_png(uniform=True), "uniform.png")

    def test_corrupt_thumbnail_is_rejected(self) -> None:
        png = bytearray(rgb_png())
        png[-8] ^= 1
        with self.assertRaisesRegex(AssertionError, "checksum"):
            self.module.validate_png(bytes(png), "corrupt.png")

    def test_transparent_hidden_colors_do_not_count_as_visible_detail(self) -> None:
        with self.assertRaisesRegex(AssertionError, "too few visible pixels"):
            self.module.validate_png(
                rgba_png_with_hidden_color(), "mostly-transparent.png"
            )

    def test_slicer_return_code_must_be_zero_even_when_artifacts_exist(self) -> None:
        self.assertEqual(self.module.validate_slicer_return_code(0), 0)
        for value in (None, 1, -9):
            with self.subTest(value=value):
                with self.assertRaisesRegex(AssertionError, "exited with status"):
                    self.module.validate_slicer_return_code(value)

    def test_compatibility_profile_identity_must_be_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "machine-ref.json").write_text('{"name":"Q1 Pro"}\n')
            (work / "process-ref.json").write_text(
                '{"compatible_printers":["Q1 Pro"]}\n'
            )
            identity = self.module.validate_compatibility_profile_identity(
                work, {"name": "Q1 Pro"}
            )

        self.assertEqual(identity["generated_machine_name"], "Q1 Pro")

    def test_compatibility_profile_identity_rejects_adapter_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "machine-ref.json").write_text('{"name":"Bambu Lab"}\n')
            (work / "process-ref.json").write_text(
                '{"compatible_printers":["Bambu Lab"]}\n'
            )
            with self.assertRaisesRegex(AssertionError, "changed machine identity"):
                self.module.validate_compatibility_profile_identity(
                    work, {"name": "Q1 Pro"}
                )


if __name__ == "__main__":
    unittest.main()
