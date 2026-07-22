from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONAN_ENV = ROOT / "slicers" / "Cura" / "steps" / "conan-env.sh"


class CuraConanEnvironmentTests(unittest.TestCase):
    def test_config_stamp_reuses_and_tracks_url_and_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config"
            config.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=config, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.com"],
                cwd=config,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"], cwd=config, check=True
            )
            subprocess.run(
                ["git", "config", "commit.gpgsign", "false"], cwd=config, check=True
            )
            (config / "global.conf").write_text("core:default_profile = cura.jinja\n")
            subprocess.run(["git", "add", "."], cwd=config, check=True)
            subprocess.run(["git", "commit", "-qm", "one"], cwd=config, check=True)
            first_ref = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()

            workspace = root / "workspace"
            (workspace / "slicer-src" / "deps" / "build").mkdir(parents=True)
            conan_home = root / "conan-home"
            log = root / "conan.log"
            fake_conan = root / "conan"
            fake_conan.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CURA_CONAN_TEST_LOG"
if [[ "$1 $2" == "config install" ]]; then
  mkdir -p "$CONAN_HOME"
  printf '%s\\n' 'core:default_profile = cura.jinja' > "$CONAN_HOME/global.conf"
elif [[ "$1 $2" == "profile detect" ]]; then
  mkdir -p "$CONAN_HOME/profiles"
  : > "$CONAN_HOME/profiles/default"
fi
"""
            )
            fake_conan.chmod(0o755)

            def initialize(url: Path, ref: str) -> None:
                env = dict(
                    os.environ,
                    CONAN=str(fake_conan),
                    CONAN_HOME=str(conan_home),
                    CURA_CONAN_CONFIG_URL=str(url),
                    CURA_CONAN_CONFIG_REF=ref,
                    CURA_CONAN_TEST_LOG=str(log),
                )
                subprocess.run(
                    ["bash", "-c", 'source "$1"; cura_conan_env', "bash", CONAN_ENV],
                    cwd=workspace,
                    env=env,
                    check=True,
                )

            initialize(config, first_ref)
            initialize(config, first_ref)

            (config / "revision").write_text("two\n")
            subprocess.run(["git", "add", "."], cwd=config, check=True)
            subprocess.run(["git", "commit", "-qm", "two"], cwd=config, check=True)
            second_ref = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            initialize(config, second_ref)

            alternate = root / "alternate.git"
            subprocess.run(
                ["git", "clone", "-q", "--bare", config, alternate], check=True
            )
            initialize(alternate, second_ref)

            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=config,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            initialize(config, branch)
            (config / "revision").write_text("three\n")
            subprocess.run(["git", "add", "."], cwd=config, check=True)
            subprocess.run(["git", "commit", "-qm", "three"], cwd=config, check=True)
            initialize(config, branch)
            initialize(config, branch)

            calls = log.read_text().splitlines()
            self.assertEqual(
                len([call for call in calls if call.startswith("config install ")]), 5
            )
            self.assertEqual(
                len([call for call in calls if call.startswith("profile detect ")]), 7
            )
            self.assertTrue((conan_home / ".slicer-builds-cura-conan-config").is_file())


if __name__ == "__main__":
    unittest.main()
