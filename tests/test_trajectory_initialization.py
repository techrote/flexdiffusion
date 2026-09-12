import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_ROOT = os.path.join(REPO_ROOT, "ui")
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

from easydiffusion.trajectory import TrajectoryRecorder  # noqa: E402


class _FailingWriter:
    def __init__(self, **kwargs):
        raise RuntimeError("synthetic writer initialization failure")


class TrajectoryInitializationTests(unittest.TestCase):
    def test_writer_initialization_failure_finalizes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("easydiffusion.trajectory_artifacts.ArtifactWriter", _FailingWriter):
                with self.assertRaisesRegex(RuntimeError, "synthetic writer initialization failure"):
                    TrajectoryRecorder(
                        {
                            "enabled": True,
                            "output_root": tmp,
                            "project_name": "init-failure",
                            "capture_schedule": "1",
                            "persistence_mode": "latent",
                        },
                        total_steps=1,
                    )

            run_dirs = [os.path.join(tmp, name) for name in os.listdir(tmp)]
            self.assertEqual(len(run_dirs), 1)
            manifest_path = os.path.join(run_dirs[0], "manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["status"], "initialisation_failed")
            self.assertIsNotNone(manifest["completed_at"])
            self.assertTrue(manifest["errors"])
            self.assertIn("synthetic writer initialization failure", manifest["errors"][0])


if __name__ == "__main__":
    unittest.main()
