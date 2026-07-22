from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAMP = ROOT / "tools" / "stamp_version_date.sh"


class VersionStampTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=source,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"], cwd=source, check=True
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"], cwd=source, check=True
        )
        (source / "version.inc").write_text("VERSION=1.0+UNKNOWN\n")
        env = dict(
            os.environ,
            GIT_AUTHOR_DATE="2024-02-03T12:00:00Z",
            GIT_COMMITTER_DATE="2024-02-03T12:00:00Z",
        )
        subprocess.run(["git", "add", "version.inc"], cwd=source, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=source, env=env, check=True
        )
        return source

    def test_defaults_to_commit_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            subprocess.run(["bash", STAMP, source], check=True)
            value = (source / "version.inc").read_text()
        self.assertEqual(value, "VERSION=1.0_2024-02-03\n")

    def test_explicit_date_override_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            env = dict(os.environ, SLICER_BUILD_DATE="2025-06-07")
            subprocess.run(["bash", STAMP, source], env=env, check=True)
            value = (source / "version.inc").read_text()
        self.assertEqual(value, "VERSION=1.0_2025-06-07\n")

    def test_invalid_date_override_fails_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            env = dict(os.environ, SLICER_BUILD_DATE="today")
            result = subprocess.run(["bash", STAMP, source], env=env, check=False)
            value = (source / "version.inc").read_text()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(value, "VERSION=1.0+UNKNOWN\n")


if __name__ == "__main__":
    unittest.main()
