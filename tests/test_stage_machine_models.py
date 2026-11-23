import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.stage_machine_models import stage_machine_models


class StageMachineModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.resources)
        self.machine = self.resources / "profiles" / "BBL" / "machine"
        self.machine.mkdir(parents=True)

    def write(self, file: str, preset: dict) -> None:
        (self.machine / file).write_text(json.dumps(preset), encoding="utf-8")

    def test_each_printer_model_is_where_the_cli_reads_its_model_id(self) -> None:
        self.write("Bambu Lab X1E.json", {"type": "machine_model", "name": "Bambu Lab X1E", "model_id": "C13"})
        self.write("Bambu Lab X1E 0.4 nozzle.json", {"type": "machine", "name": "Bambu Lab X1E 0.4 nozzle"})
        self.write("fdm_machine_common.json", {"type": "machine", "name": "fdm_machine_common"})

        self.assertEqual(stage_machine_models(self.resources), ["Bambu Lab X1E"])

        staged = self.resources / "profiles" / "BBL" / "machine_full"
        self.assertEqual([p.name for p in staged.iterdir()], ["Bambu Lab X1E.json"])
        self.assertEqual(json.loads((staged / "Bambu Lab X1E.json").read_text())["model_id"], "C13")

    def test_nothing_is_staged_without_printer_models(self) -> None:
        self.assertEqual(stage_machine_models(self.resources), [])
        self.assertFalse((self.resources / "profiles" / "BBL" / "machine_full").exists())


if __name__ == "__main__":
    unittest.main()
