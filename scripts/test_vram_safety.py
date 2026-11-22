import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ui"))

from easydiffusion.vram_safety import advise_vram


class TestVramSafety(unittest.TestCase):
    def advise(self, **overrides):
        kwargs = dict(requested="high", width=1024, height=1024, num_outputs=1,
                      model_name="sd-v1-4", total_vram_gb=4.0, enabled=True)
        kwargs.update(overrides)
        return advise_vram(**kwargs)

    def test_four_gb_1024_derates_to_low(self):
        advice = self.advise()
        self.assertEqual((advice.recommended, advice.risk), ("low", "high"))
        self.assertTrue(advice.disable_live_preview)

    def test_explicit_low_is_not_promoted(self):
        self.assertEqual(self.advise(requested="low").recommended, "low")

    def test_auto_off_preserves_profile(self):
        advice = self.advise(enabled=False)
        self.assertEqual(advice.recommended, "high")
        self.assertFalse(advice.disable_live_preview)

    def test_unknown_vram_does_not_guess(self):
        self.assertEqual(self.advise(total_vram_gb=None).risk, "unknown")

    def test_roomy_gpu_does_not_derate_sd1(self):
        self.assertEqual(self.advise(total_vram_gb=16).recommended, "high")

    def test_midrange_gpu_moderate_image_derates_one_level(self):
        advice = self.advise(width=768, height=768, total_vram_gb=6.0)
        self.assertEqual((advice.risk, advice.recommended), ("normal", "high"))

    def test_512_on_four_gb_is_elevated(self):
        advice = self.advise(width=512, height=512)
        self.assertEqual((advice.risk, advice.recommended), ("elevated", "balanced"))

    def test_1024_parallel_requires_serial_outputs(self):
        self.assertTrue(self.advise(num_outputs=4).prefer_serial_outputs)

    def test_heavy_model_gets_extra_protection(self):
        advice = self.advise(width=768, height=768, model_name="my-SDXL-model",
                             total_vram_gb=8.0)
        self.assertEqual((advice.risk, advice.recommended), ("high", "low"))

    def test_reject_invalid_profile(self):
        with self.assertRaises(ValueError):
            self.advise(requested="ultra_low")


if __name__ == "__main__":
    unittest.main()
