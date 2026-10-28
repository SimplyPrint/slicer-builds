import unittest
from pathlib import Path

from tools.slicerctl import Manifest, patch_series, version_parts


def manifest(*series: dict) -> Manifest:
    return Manifest(Path("slicers/Example/slicer.toml"), {"patches": {"series": list(series)}})


class PatchSeriesTests(unittest.TestCase):
    def test_release_takes_the_newest_series_it_has_reached(self) -> None:
        example = manifest(
            {"name": "orca-2.3", "from": "v1.4.0"},
            {"name": "orca-2.4", "from": "v1.5.3.5"},
        )

        self.assertIsNone(patch_series(example, "v1.3.0.11"))
        self.assertEqual(patch_series(example, "v1.4.2")["name"], "orca-2.3")
        self.assertEqual(patch_series(example, "v1.5.3.5")["name"], "orca-2.4")
        self.assertEqual(patch_series(example, "v1.6.0")["name"], "orca-2.4")

    def test_nightly_follows_the_newest_series_unless_it_opts_out(self) -> None:
        self.assertEqual(
            patch_series(manifest({"name": "modern", "from": "1.7.15"}), "nightly")["name"],
            "modern",
        )
        stale_branch = manifest({"name": "modern", "from": "1.7.15", "nightly": False})
        self.assertIsNone(patch_series(stale_branch, "nightly"))
        self.assertIsNone(patch_series(stale_branch, "HEAD"))

    def test_versions_compare_numerically_across_prefixes(self) -> None:
        self.assertLess(version_parts("v1.5.2.2"), version_parts("v1.5.3.5"))
        self.assertLess(version_parts("1.7.9"), version_parts("1.7.15"))
        self.assertLess(version_parts("V2.3.6"), version_parts("v2.4.1"))
        self.assertIsNone(version_parts("nightly"))


if __name__ == "__main__":
    unittest.main()
