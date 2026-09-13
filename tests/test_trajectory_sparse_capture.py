import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_ROOT = os.path.join(REPO_ROOT, "ui")
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

from easydiffusion.output_steps import (  # noqa: E402
    OutputAfterStepCollector,
    decode_sparse_representation,
    encode_sparse_representation,
    format_capture_steps,
    normalize_intermediate_representation,
    parse_capture_step_expression,
)


class SparseCaptureExpressionTests(unittest.TestCase):
    def test_printer_style_expression_expands_inclusively(self):
        self.assertEqual(
            parse_capture_step_expression("10, 16, 20-24", 30),
            [10, 16, 20, 21, 22, 23, 24],
        )

    def test_expression_sorts_deduplicates_and_formats(self):
        steps = parse_capture_step_expression("24, 10, 16, 20-22, 21", 30)
        self.assertEqual(steps, [10, 16, 20, 21, 22, 24])
        self.assertEqual(format_capture_steps(steps), "10,16,20-22,24")

    def test_invalid_ranges_are_rejected(self):
        for expression in ("", "1,,2", "24-20", "abc", "3-", "0", "31", "1-31"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse_capture_step_expression(expression, 30)

    def test_sparse_transport_round_trips(self):
        marker = encode_sparse_representation("10, 16, 20-24", 30, True)
        self.assertEqual(marker, "sparse:10,16,20-24|endpoints=1")
        self.assertEqual(
            decode_sparse_representation(marker, 30),
            ([10, 16, 20, 21, 22, 23, 24], True),
        )
        self.assertEqual(normalize_intermediate_representation(marker), marker)


class _FakeTensor:
    def __init__(self, name):
        self.name = name

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self

    def contiguous(self):
        return self


class SparseCollectorTests(unittest.TestCase):
    def test_only_requested_denoised_steps_are_captured(self):
        marker = encode_sparse_representation("10, 16, 20-24", 30, False)
        collector = OutputAfterStepCollector(10, 30, representation=marker)
        wanted = [i for i in range(30) if collector.wants_callback_index(i)]
        self.assertEqual(wanted, [10, 16, 20, 21, 22, 23, 24])

        solver = _FakeTensor("solver")
        denoised = _FakeTensor("denoised")
        self.assertFalse(collector.capture_step(11, solver, denoised=denoised))
        self.assertTrue(collector.capture_step(10, solver, denoised=denoised))
        self.assertEqual(
            [(step, rep, tensor.name) for step, rep, tensor in collector.pop_snapshots()],
            [(10, "denoised", "denoised")],
        )

    def test_endpoint_mode_adds_only_first_and_last_solver_states(self):
        marker = encode_sparse_representation("10, 16, 20-24", 30, True)
        collector = OutputAfterStepCollector(10, 30, representation=marker)
        solver = _FakeTensor("solver")
        denoised = _FakeTensor("denoised")

        for step in (10, 16, 20, 21, 22, 23, 24):
            self.assertTrue(collector.capture_step(step, solver, denoised=denoised))

        self.assertEqual(
            [(step, rep) for step, rep, _ in collector.pop_snapshots()],
            [
                (10, "denoised"),
                (10, "solver_state"),
                (16, "denoised"),
                (20, "denoised"),
                (21, "denoised"),
                (22, "denoised"),
                (23, "denoised"),
                (24, "denoised"),
                (24, "solver_state"),
            ],
        )

    def test_true_final_step_uses_ordinary_final_instead_of_fake_solver_callback(self):
        marker = encode_sparse_representation("16,20", 20, True)
        collector = OutputAfterStepCollector(16, 20, representation=marker)
        self.assertTrue(collector.wants_callback_index(16))
        self.assertFalse(collector.wants_callback_index(19))


if __name__ == "__main__":
    unittest.main()
