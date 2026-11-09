import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.stage_bundle import (
    BundleError,
    _dependency_closure,
    _resolve_missing_search_dirs,
)


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


@unittest.skipUnless(
    shutil.which("cc")
    and any(
        Path(directory, "libz.so.1").exists()
        for directory in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
                          "/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu")
    ),
    "needs a C compiler and the system's libz.so.1",
)
class BundledLibraryTests(unittest.TestCase):
    """A library the build host's system also has (OrcaSlicer 2.5's libpython)."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root)
        self.deps = self.root / "deps" / "build" / "libz" / "lib"
        self.deps.mkdir(parents=True)
        source = self.root / "empty.c"
        source.write_text("int sp_empty;\n")
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-Wl,-soname,libz.so.1", "-o",
             str(self.deps / "libz.so.1"), str(source)],
            check=True,
        )
        main = self.root / "main.c"
        main.write_text("int main(void) { return 0; }\n")
        self.executable = self.root / "build" / "src" / "app"
        self.executable.parent.mkdir(parents=True)
        subprocess.run(
            ["cc", "-o", str(self.executable), str(main), "-Wl,--no-as-needed", "-lz"],
            check=True,
        )
        self.roots = [self.root / "build" / "src", self.root / "deps" / "build"]

    def test_the_systems_copy_is_left_out_by_default(self) -> None:
        self.assertEqual(_dependency_closure(self.executable, self.roots), {})

    def test_a_named_library_comes_from_the_library_roots(self) -> None:
        self.assertEqual(
            _dependency_closure(self.executable, self.roots, ["libz.so.1"]),
            {self.deps / "libz.so.1": {"libz.so.1"}},
        )


if __name__ == "__main__":
    unittest.main()
