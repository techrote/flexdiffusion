import json
import os
import sys
import tempfile
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_ROOT = os.path.join(REPO_ROOT, "ui")
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

from easydiffusion.trajectory import (  # noqa: E402
    CaptureScheduleError,
    TrajectoryRecorder,
    compile_capture_schedule,
)
from easydiffusion.types import RenderTaskData  # noqa: E402


class CaptureScheduleTests(unittest.TestCase):
    def test_exact_steps_are_sorted_and_deduplicated(self):
        self.assertEqual(compile_capture_schedule("10,5,10,1", 10), (1, 5, 10))

    def test_ranges_are_inclusive_when_stride_lands_on_end(self):
        self.assertEqual(compile_capture_schedule("1-10:3", 10), (1, 4, 7, 10))

    def test_ranges_do_not_invent_an_unaligned_endpoint(self):
        self.assertEqual(compile_capture_schedule("2-10:3", 10), (2, 5, 8))

    def test_percentages_use_one_based_ceiling_semantics(self):
        self.assertEqual(
            compile_capture_schedule("10%,25%,50%,75%,100%", 20),
            (2, 5, 10, 15, 20),
        )
        self.assertEqual(compile_capture_schedule("1%", 20), (1,))

    def test_mixed_schedule_is_supported(self):
        self.assertEqual(
            compile_capture_schedule("1-3,50%,10", 10),
            (1, 2, 3, 5, 10),
        )

    def test_empty_schedule_is_valid(self):
        self.assertEqual(compile_capture_schedule("", 20), ())
        self.assertEqual(compile_capture_schedule("   ", 20), ())

    def test_invalid_schedule_items_raise(self):
        invalid = (
            "0",
            "11",
            "0%",
            "101%",
            "5-2",
            "1-11",
            "1-10:0",
            "1,,2",
            "banana",
        )
        for spec in invalid:
            with self.subTest(spec=spec):
                with self.assertRaises(CaptureScheduleError):
                    compile_capture_schedule(spec, 10)

    def test_invalid_total_steps_raise(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(CaptureScheduleError):
                    compile_capture_schedule("1", value)


class TrajectoryConfigTests(unittest.TestCase):
    def test_render_task_defaults_to_trajectory_disabled(self):
        task = RenderTaskData()
        self.assertFalse(task.trajectory.enabled)
        self.assertEqual(task.trajectory.capture_schedule, "")

    def test_nested_trajectory_config_is_parsed(self):
        task = RenderTaskData.parse_obj(
            {
                "trajectory": {
                    "enabled": True,
                    "capture_schedule": "5,10,100%",
                    "persistence_mode": "latent",
                    "max_checkpoints": 8,
                }
            }
        )
        self.assertTrue(task.trajectory.enabled)
        self.assertEqual(task.trajectory.capture_schedule, "5,10,100%")
        self.assertEqual(task.trajectory.persistence_mode, "latent")
        self.assertEqual(task.trajectory.max_checkpoints, 8)


class _FakeTensor:
    shape = (1, 4, 64, 64)
    dtype = "torch.float16"
    device = "cuda:0"


class TrajectoryRecorderTests(unittest.TestCase):
    def test_disabled_recorder_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TrajectoryRecorder(
                {"enabled": False, "output_root": tmp, "capture_schedule": "1,2"},
                total_steps=2,
            )
            self.assertFalse(recorder.enabled)
            self.assertIsNone(recorder.manifest_path)
            self.assertEqual(os.listdir(tmp), [])

    def test_requested_steps_are_recorded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TrajectoryRecorder(
                {
                    "enabled": True,
                    "output_root": tmp,
                    "project_name": "unit test",
                    "capture_schedule": "1,3,100%",
                    "persistence_mode": "hybrid",
                },
                total_steps=5,
                run_metadata={"seed": 42},
            )

            for callback_index in range(5):
                recorder.record_step(callback_index, _FakeTensor())
                recorder.record_step(callback_index, _FakeTensor())

            recorder.finish()

            with open(recorder.manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["capture_steps"], [1, 3, 5])
            self.assertEqual([c["step"] for c in manifest["checkpoints"]], [1, 3, 5])
            self.assertEqual(manifest["run_metadata"]["seed"], 42)
            self.assertEqual(manifest["checkpoints"][0]["tensor"]["shape"], [1, 4, 64, 64])

    def test_max_checkpoint_budget_is_enforced_up_front(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TrajectoryRecorder(
                    {
                        "enabled": True,
                        "output_root": tmp,
                        "capture_schedule": "1-5",
                        "max_checkpoints": 4,
                    },
                    total_steps=5,
                )


if __name__ == "__main__":
    unittest.main()
