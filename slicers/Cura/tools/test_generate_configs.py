import runpy
import unittest
from pathlib import Path


GENERATOR = runpy.run_path(Path(__file__).with_name("generate-configs.py"))


class RuntimeScopeTest(unittest.TestCase):
    def test_global_scope_requires_an_explicit_false(self):
        runtime_scope = GENERATOR["runtime_scope"]

        self.assertEqual("global", runtime_scope({"settable_per_extruder": False}))
        self.assertEqual("extruder.0", runtime_scope({"settable_per_extruder": True}))
        self.assertEqual("extruder.0", runtime_scope({}))


if __name__ == "__main__":
    unittest.main()
