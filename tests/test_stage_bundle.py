import shutil
import tempfile
import unittest
from pathlib import Path

from tools.stage_bundle import BundleError, _resolve_missing_search_dirs


class MissingLibraryResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root)
        self.app = self.root / "build" / "src" / "Release"
        self.install = self.root / "deps" / "build" / "usr" / "local" / "lib"
        self.build_tree = self.root / "deps" / "build" / "dep_FFMPEG" / "libavformat"
        for directory in (self.app, self.install, self.build_tree):
            directory.mkdir(parents=True)
        (self.install / "libavformat.so.61").write_bytes(b"stripped")
        (self.app / "libavformat.so.61").write_bytes(b"stripped")
        (self.build_tree / "libavformat.so.61").write_bytes(b"unstripped")
        self.roots = [self.root / "build" / "src", self.root / "deps" / "build"]

    def test_differing_copies_are_ambiguous_without_an_application_copy(self) -> None:
        with self.assertRaisesRegex(BundleError, "ambiguous"):
            _resolve_missing_search_dirs(self.roots, ["libavformat.so.61"])

    def test_copy_next_to_the_executable_wins(self) -> None:
        self.assertEqual(
            _resolve_missing_search_dirs(self.roots, ["libavformat.so.61"], self.app),
            [self.app],
        )

    def test_byte_identical_copies_are_one_artifact(self) -> None:
        (self.build_tree / "libavformat.so.61").unlink()
        self.assertEqual(
            len(_resolve_missing_search_dirs(self.roots, ["libavformat.so.61"])), 1
        )


if __name__ == "__main__":
    unittest.main()
