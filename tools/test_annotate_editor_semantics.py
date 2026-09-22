#!/usr/bin/env python3
"""Tests for the editor-semantics annotator.

Run with ``python3 -m unittest tools.test_annotate_editor_semantics`` from the
repository root, or ``python3 tools/test_annotate_editor_semantics.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate_editor_semantics import (  # noqa: E402
    annotate,
    dump_like_source,
    is_derived_tool_reference,
    main,
)


def chooser(**overrides):
    definition = {"type": "int", "gui_type": "i_enum_open", "min": "0", "default_value": "0"}
    definition.update(overrides)
    return definition


class DerivationTests(unittest.TestCase):
    def test_recognises_a_filament_chooser(self):
        self.assertTrue(is_derived_tool_reference(chooser()))

    def test_rejects_a_count_measured_in_layers(self):
        # support_interface_top_layers is spelled the same way but has a unit.
        self.assertFalse(is_derived_tool_reference(chooser(sidetext="layers", default_value="3")))

    def test_rejects_a_count_that_admits_minus_one(self):
        # support_interface_bottom_layers starts below the first tool.
        self.assertFalse(is_derived_tool_reference(chooser(min="-1")))

    def test_rejects_non_integer_and_unmarked_options(self):
        self.assertFalse(is_derived_tool_reference(chooser(type="float")))
        self.assertFalse(is_derived_tool_reference(chooser(gui_type=None)))

    def test_finds_renamed_keys_without_naming_them(self):
        # Orca 2.4 renamed the region choosers; the rule must not care.
        defs = {
            "outer_wall_filament_id": chooser(),
            "support_filament": chooser(),
            "support_interface_top_layers": chooser(sidetext="layers"),
        }
        self.assertEqual(
            annotate(defs, None),
            ["outer_wall_filament_id", "support_filament"],
        )


class EditorBlockTests(unittest.TestCase):
    def test_optional_selection_carries_the_inherit_sentinel(self):
        defs = {"support_filament": chooser()}
        annotate(defs, None)
        self.assertEqual(
            defs["support_filament"]["editor"],
            {
                "kind": "tool_reference",
                "index_base": 1,
                "display_index_base": 1,
                "selection": "optional",
                "sentinels": [{"value": 0, "semantic": "inherit"}],
            },
        )

    def test_a_chooser_that_must_name_a_tool_has_no_sentinel(self):
        # PrusaSlicer's perimeter_extruder starts at 1: there is no "inherit"
        # value to fall back to. It reaches us through the overlay, because a
        # min of 1 is also how the Orca family spells a plain count.
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "semantics.toml"
            overlay.write_text('[tool_reference]\nkeys = ["perimeter_extruder"]\n')
            defs = {"perimeter_extruder": {"type": "int", "min": "1", "default_value": "1"}}
            annotate(defs, overlay)
            editor = defs["perimeter_extruder"]["editor"]
            self.assertEqual(editor["selection"], "required")
            self.assertNotIn("sentinels", editor)

    def test_never_overwrites_a_generator_that_described_itself(self):
        # Cura publishes index_base 0 with a -1 sentinel; leave it alone.
        cura = {"kind": "tool_reference", "index_base": 0, "selection": "optional"}
        defs = {"support_extruder_nr": chooser(editor=dict(cura))}
        self.assertEqual(annotate(defs, None), [])
        self.assertEqual(defs["support_extruder_nr"]["editor"], cura)


class OverlayTests(unittest.TestCase):
    def test_declared_keys_are_annotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "semantics.toml"
            overlay.write_text('[tool_reference]\nkeys = ["support_material_extruder"]\n')
            defs = {"support_material_extruder": {"type": "int", "min": "0"}}
            self.assertEqual(annotate(defs, overlay), ["support_material_extruder"])

    def test_a_renamed_declared_key_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "semantics.toml"
            overlay.write_text('[tool_reference]\nkeys = ["gone_upstream"]\n')
            with self.assertRaises(SystemExit) as caught:
                annotate({"still_here": chooser()}, overlay)
            self.assertIn("gone_upstream", str(caught.exception))


class SerializationTests(unittest.TestCase):
    def test_matches_the_boost_writer(self):
        # Escaped slashes and raw UTF-8, so annotating shows only new blocks.
        rendered = dump_like_source({"a": {"tooltip": "http://x", "sidetext": "°C"}})
        self.assertIn('"http:\\/\\/x"', rendered)
        self.assertIn('"°C"', rendered)
        self.assertTrue(rendered.endswith("\n"))


class CliTests(unittest.TestCase):
    def test_check_mode_leaves_the_file_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "print_config_def.json"
            path.write_text(dump_like_source({"support_filament": chooser()}))
            before = path.read_text()
            self.assertEqual(main(["--output", tmp, "--check"]), 0)
            self.assertEqual(path.read_text(), before)

    def test_writing_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "print_config_def.json"
            path.write_text(dump_like_source({"support_filament": chooser()}))
            main(["--output", tmp])
            once = path.read_text()
            main(["--output", tmp])
            self.assertEqual(path.read_text(), once)
            self.assertEqual(json.loads(once)["support_filament"]["editor"]["kind"], "tool_reference")

    def test_require_flag_fails_when_nothing_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "print_config_def.json"
            path.write_text(dump_like_source({"layer_height": {"type": "float"}}))
            with self.assertRaises(SystemExit):
                main(["--output", tmp, "--require-tool-reference"])


if __name__ == "__main__":
    unittest.main()
