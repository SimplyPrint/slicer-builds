import json
import tempfile
import unittest
from pathlib import Path

from tools.convert_prusa_3_config_model import convert


def item(name, location, category, group, value, **extra):
    return {
        "name": name,
        "type": "Float",
        "location": location,
        "overrides_in": [],
        "category": category,
        "option_group": group,
        "order": 1,
        "category_order": 100,
        "option_group_order": 100,
        "gui_type": "spinbox",
        "label": name,
        "tooltip": "help",
        "value": value,
        **extra,
    }


class Prusa3ConfigModelTests(unittest.TestCase):
    def test_converts_native_boxes_and_tool_print_settings(self):
        model = {
            "printer": {
                "items": [
                    item(
                        "max_print_height",
                        "Printer",
                        "Printer_General",
                        "Printer_General_SizeClearances",
                        250,
                    )
                ],
                "overrides": [],
            },
            "print": {
                "items": [
                    item(
                        "layer_height",
                        "Print",
                        "Print_LayersSurfaces",
                        "Print_LayersSurfaces_LayerHeight",
                        0.2,
                        units=["mm"],
                    )
                ],
                "overrides": [],
            },
            "tool_print": {
                "items": [
                    item(
                        "retract_length",
                        "Tool",
                        "Print_ExtrusionRetraction",
                        "Print_ExtrusionRetraction_Retraction",
                        0.8,
                    )
                ],
                "overrides": [],
            },
            "filament": {
                "items": [
                    {
                        **item(
                            "filament_type",
                            "Filament",
                            "Filament_MaterialTemperatures",
                            "Filament_MaterialTemperatures_MaterialProperty",
                            "PLA",
                        ),
                        "type": "Enum",
                        "enum_values": ["PLA", "PETG"],
                    }
                ],
                "overrides": [],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config-model.json"
            output = root / "out"
            output.mkdir()
            source.write_text(json.dumps(model), encoding="utf-8")

            icons = root / "icons"
            icons.mkdir()
            (icons / "layers.svg").write_text("<svg></svg>", encoding="utf-8")

            convert(source, output, icons)

            definitions = json.loads(
                (output / "print_config_def.json").read_text(encoding="utf-8")
            )
            process = json.loads((output / "process.json").read_text(encoding="utf-8"))
            metadata = json.loads(
                (output / "ui_metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(definitions["layer_height"]["sidetext"], "mm")
        self.assertEqual(definitions["filament_type"]["type"], "enum")
        self.assertEqual(definitions["filament_type"]["default_value"], "PLA")
        self.assertNotIn("wire_codec", definitions["layer_height"])
        self.assertNotIn("wire_type", definitions["layer_height"])
        self.assertTrue(
            any(
                "retract_length" in settings
                for groups in process.values()
                for settings in groups.values()
            )
        )
        self.assertEqual(metadata["settings_codec"], "prusa-domain-config.v1")
        self.assertEqual(
            metadata["capabilities"]["profile_metadata_transport"], "envelope.v1"
        )
        self.assertEqual(
            metadata["runtime_setting_scopes"],
            {name: "global" for name in sorted(definitions)},
        )

    def test_preserves_native_category_metadata_order_icons_and_arrays(self):
        late = item(
            "late",
            "Print",
            "Print_PrecisionSlicing",
            "Print_PrecisionSlicing_ScriptSubstitutions",
            1,
            category_order=109,
            option_group_order=1005,
            category_label="Precision & Slicing",
            option_group_label="Scripts & Substitutions",
            category_icon="precision",
        )
        early = item(
            "early",
            "Print",
            "Print_LayersSurfaces",
            "Print_LayerSurfaces_LayerHeight",
            1,
            category_order=100,
            option_group_order=100,
            category_label="Layers & Surfaces",
            option_group_label="Layer height",
            category_icon="layers",
        )
        substitutions = {
            **late,
            "name": "gcode_substitutions",
            "type": "Array",
            "element_type": "String",
            "value": [],
        }
        model = {
            "printer": {"items": [early], "overrides": []},
            "print": {"items": [substitutions, early], "overrides": []},
            "tool_print": {"items": [], "overrides": []},
            "filament": {"items": [early], "overrides": []},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config-model.json"
            output = root / "out"
            icons = root / "icons"
            output.mkdir()
            icons.mkdir()
            (icons / "precision.svg").write_text("<svg>precision</svg>", encoding="utf-8")
            (icons / "layers.svg").write_text("<svg>layers</svg>", encoding="utf-8")
            source.write_text(json.dumps(model), encoding="utf-8")

            convert(source, output, icons)

            process = json.loads((output / "process.json").read_text(encoding="utf-8"))
            definitions = json.loads(
                (output / "print_config_def.json").read_text(encoding="utf-8")
            )
            metadata = json.loads((output / "ui_metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(list(process), ["Layers & Surfaces", "Precision & Slicing"])
        self.assertEqual(
            list(process["Precision & Slicing"]),
            ["Scripts & Substitutions"],
        )
        self.assertEqual(definitions["gcode_substitutions"]["default_value"], [])
        self.assertEqual(definitions["gcode_substitutions"]["native_type"], "array")
        self.assertEqual(definitions["gcode_substitutions"]["editor"]["kind"], "list")
        self.assertEqual(metadata["panels"]["process"][1]["icon"], "prusa:precision")
        self.assertEqual(metadata["icons"]["prusa:precision"]["svg"], "<svg>precision</svg>")


if __name__ == "__main__":
    unittest.main()
