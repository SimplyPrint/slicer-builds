import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.stage_cli_profiles import complete_filaments, stage_filaments, stage_machine_models


class StageCliProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.resources)
        self.profiles = self.resources / "profiles"

    def write(self, vendor: str, kind: str, preset: dict) -> None:
        path = self.profiles / vendor / kind / f"{preset['name'].replace('/', '-')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preset), encoding="utf-8")

    def test_each_printer_model_is_where_the_cli_reads_its_model_id(self) -> None:
        self.write("BBL", "machine", {"type": "machine_model", "name": "Bambu Lab X1E", "model_id": "C13"})
        self.write("BBL", "machine", {"type": "machine", "name": "Bambu Lab X1E 0.4 nozzle"})

        self.assertEqual(stage_machine_models(self.resources), ["Bambu Lab X1E"])

        staged = self.profiles / "BBL" / "machine_full"
        self.assertEqual([p.name for p in staged.iterdir()], ["Bambu Lab X1E.json"])
        self.assertEqual(json.loads((staged / "Bambu Lab X1E.json").read_text())["model_id"], "C13")

    def test_a_filament_is_completed_as_the_app_loads_it(self) -> None:
        self.write("BBL", "filament", {"name": "fdm_filament_common", "instantiation": "false",
                                       "nozzle_temperature": ["200"], "filament_type": ["PLA"]})
        self.write("BBL", "filament", {"name": "fdm_filament_template_direct_dual", "instantiation": "false",
                                       "filament_extruder_variant": ["Direct Drive Standard", "Direct Drive High Flow"]})
        self.write("BBL", "filament", {"name": "Generic ASA @base", "inherits": "fdm_filament_common",
                                       "instantiation": "false", "filament_id": "GFB98",
                                       "setting_id": "GFSB98", "filament_type": ["ASA"]})
        self.write("BBL", "filament", {"name": "Generic ASA @BBL X1E", "inherits": "Generic ASA @base",
                                       "include": ["fdm_filament_template_direct_dual"],
                                       "instantiation": "true", "setting_id": "GFSB98_02",
                                       "nozzle_temperature": ["260", "260"]})

        asa = complete_filaments(self.profiles)["Generic ASA @BBL X1E"]

        self.assertEqual(asa["filament_id"], "GFB98")
        self.assertEqual(asa["setting_id"], "GFSB98_02")
        self.assertEqual(asa["filament_type"], ["ASA"])
        self.assertEqual(asa["nozzle_temperature"], ["260", "260"])
        self.assertEqual(asa["filament_extruder_variant"], ["Direct Drive Standard", "Direct Drive High Flow"])
        self.assertEqual((asa["type"], asa["from"]), ("filament", "system"))
        self.assertNotIn("inherits", asa)
        self.assertNotIn("include", asa)

    def test_a_parent_is_found_in_another_vendor_and_a_slash_is_a_folder(self) -> None:
        self.write("OrcaFilamentLibrary", "filament", {"name": "Generic PLA @System", "filament_id": "OGFL99",
                                                       "nozzle_temperature": ["220"]})
        self.write("Vendor", "filament", {"name": "PLA/Silk @Vendor", "inherits": "Generic PLA @System",
                                          "instantiation": "true"})

        self.assertEqual(stage_filaments(self.resources), 2)

        staged = self.profiles / "BBL" / "filament_full" / "PLA" / "Silk @Vendor.json"
        silk = json.loads(staged.read_text())
        self.assertEqual((silk["filament_id"], silk["nozzle_temperature"]), ("OGFL99", ["220"]))


if __name__ == "__main__":
    unittest.main()
