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

            convert(source, output)

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


if __name__ == "__main__":
    unittest.main()
