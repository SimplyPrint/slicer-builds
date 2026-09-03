import unittest

from tools.update_slicer_index import release_channel, should_promote, version_sort_key


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

    def test_release_channels_keep_prereleases_out_of_latest(self) -> None:
        index = {
            "latest": "version_2.9.6",
            "channels": {"stable": "version_2.9.6"},
            "versions": {"version_2.9.6": {}, "version_3.0.0-alpha11": {}},
        }

        self.assertEqual(release_channel("version_3.0.0-alpha11"), "prerelease")
        self.assertTrue(should_promote(index, "prerelease", "version_3.0.0-alpha11"))
        self.assertEqual(release_channel("version_3.0.0"), "stable")

    def test_existing_latest_remains_the_stable_channel_fallback(self) -> None:
        index = {
            "latest": "version_2.9.6",
            "versions": {"version_2.9.6": {}, "version_2.9.5": {}},
        }

        self.assertFalse(should_promote(index, "stable", "version_2.9.5"))


if __name__ == "__main__":
    unittest.main()
