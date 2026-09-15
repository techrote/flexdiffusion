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
    normalize_intermediate_representation,
    normalize_output_after_step,
)
from easydiffusion.types import GenerateImageRequest  # noqa: E402


class OutputAfterStepRequestTests(unittest.TestCase):
    def test_request_defaults_to_disabled_final_only_behavior(self):
        req = GenerateImageRequest()
        self.assertIsNone(req.output_after_step)
        self.assertEqual(req.intermediate_representation, "both")

    def test_request_parses_explicit_output_step_and_representation(self):
        req = GenerateImageRequest.parse_obj(
            {
                "num_inference_steps": 20,
                "output_after_step": 10,
                "intermediate_representation": "denoised",
            }
        )
        self.assertEqual(req.num_inference_steps, 20)
        self.assertEqual(req.output_after_step, 10)
        self.assertEqual(req.intermediate_representation, "denoised")


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

    def test_representation_normalization(self):
        self.assertEqual(normalize_intermediate_representation(None), "both")
        self.assertEqual(normalize_intermediate_representation(" BOTH "), "both")
        self.assertEqual(normalize_intermediate_representation("denoised"), "denoised")
        self.assertEqual(normalize_intermediate_representation("solver_state"), "solver_state")
        with self.assertRaises(ValueError):
            normalize_intermediate_representation("mystery")

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

    def test_flatten_results_orders_denoised_before_solver_state_and_preserves_batch_seeds(self):
        flattened = flatten_output_step_results(
            [
                {"step": 11, "representation": "solver_state", "images": ["11s-a", "11s-b"]},
                {"step": 10, "representation": "solver_state", "images": ["10s-a", "10s-b"]},
                {"step": 10, "representation": "denoised", "images": ["10d-a", "10d-b"]},
                {"step": 11, "representation": "denoised", "images": ["11d-a", "11d-b"]},
            ],
            seed=42,
            total_steps=20,
        )
        self.assertEqual(
            [(image, seed, metadata["output_step"], metadata["intermediate_representation"]) for image, seed, metadata in flattened],
            [
                ("10d-a", 42, 10, "denoised"),
                ("10d-b", 43, 10, "denoised"),
                ("10s-a", 42, 10, "solver_state"),
                ("10s-b", 43, 10, "solver_state"),
                ("11d-a", 42, 11, "denoised"),
                ("11d-b", 43, 11, "denoised"),
                ("11s-a", 42, 11, "solver_state"),
                ("11s-b", 43, 11, "solver_state"),
            ],
        )
        self.assertTrue(all(metadata["is_intermediate"] for _, _, metadata in flattened))
        self.assertTrue(all(metadata["total_steps"] == 20 for _, _, metadata in flattened))


class _FakeTensor:
    def __init__(self, name):
        self.name = name
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
    def test_both_mode_owns_denoised_then_solver_state_per_selected_step(self):
        collector = OutputAfterStepCollector(2, 4, representation="both")
        solver = _FakeTensor("solver")
        denoised = _FakeTensor("denoised")

        self.assertFalse(collector.capture_step(1, solver, denoised=denoised))
        self.assertTrue(collector.capture_step(2, solver, denoised=denoised))
        self.assertFalse(collector.capture_step(2, solver, denoised=denoised))
        self.assertTrue(collector.capture_step(3, solver, denoised=denoised))

        snapshots = collector.pop_snapshots()
        self.assertEqual(
            [(step, representation, tensor.name) for step, representation, tensor in snapshots],
            [
                (2, "denoised", "denoised"),
                (2, "solver_state", "solver"),
                (3, "denoised", "denoised"),
                (3, "solver_state", "solver"),
            ],
        )
        expected_calls = ["detach", "cpu", "clone", "contiguous"] * 2
        self.assertEqual(denoised.calls, expected_calls)
        self.assertEqual(solver.calls, expected_calls)
        self.assertEqual(collector.pop_snapshots(), [])

    def test_denoised_mode_requires_and_captures_only_clean_estimate(self):
        collector = OutputAfterStepCollector(1, 3, representation="denoised")
        solver = _FakeTensor("solver")
        denoised = _FakeTensor("denoised")

        with self.assertRaises(ValueError):
            collector.capture_step(1, solver, denoised=None)

        self.assertTrue(collector.capture_step(1, solver, denoised=denoised))
        snapshots = collector.pop_snapshots()
        self.assertEqual([(step, rep, tensor.name) for step, rep, tensor in snapshots], [(1, "denoised", "denoised")])
        self.assertEqual(solver.calls, [])

    def test_solver_state_mode_does_not_require_denoised_value(self):
        collector = OutputAfterStepCollector(1, 3, representation="solver_state")
        solver = _FakeTensor("solver")
        self.assertTrue(collector.capture_step(1, solver, denoised=None))
        snapshots = collector.pop_snapshots()
        self.assertEqual([(step, rep, tensor.name) for step, rep, tensor in snapshots], [(1, "solver_state", "solver")])


if __name__ == "__main__":
    unittest.main()
