import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_ROOT = os.path.join(REPO_ROOT, "ui")
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

from easydiffusion.output_steps import (  # noqa: E402
    FIXED_STEP_K_DIFFUSION_SAMPLERS,
    OutputAfterStepCollector,
    flatten_output_step_results,
    normalize_output_after_step,
)
from easydiffusion.types import GenerateImageRequest  # noqa: E402


class OutputAfterStepRequestTests(unittest.TestCase):
    def test_request_defaults_to_disabled_final_only_behavior(self):
        req = GenerateImageRequest()
        self.assertIsNone(req.output_after_step)

    def test_request_parses_explicit_output_step(self):
        req = GenerateImageRequest.parse_obj({"num_inference_steps": 20, "output_after_step": 10})
        self.assertEqual(req.num_inference_steps, 20)
        self.assertEqual(req.output_after_step, 10)


class OutputAfterStepSelectionTests(unittest.TestCase):
    def test_normalization_defaults_and_clamps(self):
        self.assertEqual(normalize_output_after_step(None, 20), 20)
        self.assertEqual(normalize_output_after_step(10, 20), 10)
        self.assertEqual(normalize_output_after_step(0, 20), 1)
        self.assertEqual(normalize_output_after_step(999, 20), 20)

    def test_invalid_total_steps_raise(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_output_after_step(1, value)

    def test_callback_indexes_map_to_completed_steps(self):
        collector = OutputAfterStepCollector(10, 20)
        wanted = [i for i in range(20) if collector.wants_callback_index(i)]
        self.assertEqual(wanted, list(range(10, 20)))

    def test_final_only_value_does_not_capture_callback_latents(self):
        collector = OutputAfterStepCollector(20, 20)
        self.assertFalse(collector.enabled)
        self.assertFalse(any(collector.wants_callback_index(i) for i in range(20)))

    def test_supported_sampler_scope_is_deliberately_fixed_step_k_diffusion(self):
        self.assertIn("dpmpp_2m", FIXED_STEP_K_DIFFUSION_SAMPLERS)
        self.assertIn("euler", FIXED_STEP_K_DIFFUSION_SAMPLERS)
        self.assertNotIn("dpm_fast", FIXED_STEP_K_DIFFUSION_SAMPLERS)
        self.assertNotIn("dpm_adaptive", FIXED_STEP_K_DIFFUSION_SAMPLERS)
        self.assertNotIn("ddim", FIXED_STEP_K_DIFFUSION_SAMPLERS)

    def test_flatten_results_is_chronological_and_preserves_batch_seeds(self):
        flattened = flatten_output_step_results(
            [
                {"step": 11, "images": ["11-a", "11-b"]},
                {"step": 10, "images": ["10-a", "10-b"]},
            ],
            seed=42,
            total_steps=20,
        )
        self.assertEqual(
            [(image, seed, metadata["output_step"]) for image, seed, metadata in flattened],
            [
                ("10-a", 42, 10),
                ("10-b", 43, 10),
                ("11-a", 42, 11),
                ("11-b", 43, 11),
            ],
        )
        self.assertTrue(all(metadata["is_intermediate"] for _, _, metadata in flattened))
        self.assertTrue(all(metadata["total_steps"] == 20 for _, _, metadata in flattened))


class _FakeTensor:
    def __init__(self):
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def clone(self):
        self.calls.append("clone")
        return self

    def contiguous(self):
        self.calls.append("contiguous")
        return self


class OutputAfterStepCollectorTests(unittest.TestCase):
    def test_capture_owns_one_cpu_snapshot_per_selected_step(self):
        collector = OutputAfterStepCollector(2, 4)
        tensor = _FakeTensor()

        self.assertFalse(collector.capture_step(1, tensor))
        self.assertTrue(collector.capture_step(2, tensor))
        self.assertFalse(collector.capture_step(2, tensor))
        self.assertTrue(collector.capture_step(3, tensor))

        snapshots = collector.pop_snapshots()
        self.assertEqual([step for step, _ in snapshots], [2, 3])
        self.assertEqual(
            tensor.calls,
            ["detach", "cpu", "clone", "contiguous", "detach", "cpu", "clone", "contiguous"],
        )
        self.assertEqual(collector.pop_snapshots(), [])


if __name__ == "__main__":
    unittest.main()
