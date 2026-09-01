import unittest

from tools.update_slicer_index import version_sort_key


class SlicerIndexVersionTests(unittest.TestCase):
    def test_prerelease_order(self) -> None:
        self.assertLess(
            version_sort_key("version_3.0.0-alpha2"),
            version_sort_key("version_3.0.0-alpha11"),
        )
        self.assertLess(
            version_sort_key("version_3.0.0-alpha11"),
            version_sort_key("version_3.0.0-beta1"),
        )
        self.assertLess(
            version_sort_key("version_3.0.0-rc1"),
            version_sort_key("version_3.0.0"),
        )


if __name__ == "__main__":
    unittest.main()
