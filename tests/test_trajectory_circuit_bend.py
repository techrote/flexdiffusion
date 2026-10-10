"""Numerical and routing contracts for memory, envelopes and PLMS freezing."""
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))
from easydiffusion.circuit_bend import CircuitBendOptions, NoiseMemory, PLMSHistoryFreeze, sample_bent
from easydiffusion.sampler_adapters import sampling_session
from easydiffusion.types import GenerateImageRequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sample_render_policy


class EnvelopeTests(unittest.TestCase):
    def test_sample_policy_rechecks_explicit_idle_server(self):
        import io
        server = {"pid": 999999, "process_name": "python.exe", "status_url": "http://127.0.0.1:9000/ping"}
        with patch.object(sample_render_policy, "_idle_servers", [server]), patch.object(
            sample_render_policy, "query_smi", return_value=[[sample_render_policy.TITAN_UUID, "999999", "python.exe"]]
        ), patch.object(sample_render_policy, "urlopen") as status:
            status.return_value = io.BytesIO(b'{"status":"Online"}')
            sample_render_policy.assert_no_other_compute()
            status.return_value = io.BytesIO(b'{"status":"Rendering"}')
            with self.assertRaisesRegex(RuntimeError, "not idle"):
                sample_render_policy.assert_no_other_compute()
            status.side_effect = OSError("unreachable")
            with self.assertRaises(OSError): sample_render_policy.assert_no_other_compute()

    def test_sample_policy_requires_fresh_check_and_refuses_another_renderer(self):
        sample_render_policy.validate_thread_check({"checked_at": 1000, "titan_available": True, "notes": "checked chats"}, now=1100)
        for record in [{}, {"checked_at": 1000, "titan_available": False, "notes": "other chat"},
                       {"checked_at": 1, "titan_available": True, "notes": "stale"}]:
            with self.assertRaises(RuntimeError): sample_render_policy.validate_thread_check(record, now=1100)
        with patch.object(sample_render_policy, "query_smi", return_value=[
            [sample_render_policy.TITAN_UUID, "999999", "python.exe"]
        ]):
            with self.assertRaisesRegex(RuntimeError, "another compute process"):
                sample_render_policy.assert_no_other_compute()

    def test_percent_native_count_and_endpoints(self):
        options = CircuitBendOptions(window_unit="percent", start=25, end=75, fade_in=25, fade_out=25)
        self.assertEqual(options.resolve_window(20), (5, 15))
        self.assertEqual(options.resolve_window(31), (8, 24))
        self.assertEqual([round(options.weight(i, 20), 6) for i in [4, 5, 6, 8, 10, 12, 14, 15, 16]],
                         [0, 0, .4, 1, 1, 1, .4, 0, 0])

    def test_pulses_single_iteration_and_overlap(self):
        options = CircuitBendOptions(start=3, end=7, every=2)
        self.assertEqual([options.weight(i, 10) for i in range(1, 9)], [0, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(CircuitBendOptions(start=3, end=3, fade_in=100, fade_out=100).weight(3, 10), 1)
        overlap = CircuitBendOptions(start=1, end=5, fade_in=100, fade_out=100)
        self.assertEqual([overlap.weight(i, 5) for i in range(1, 6)], [0, .25, .5, .25, 0])

    def test_legacy_memory_and_new_request_roundtrip(self):
        self.assertEqual(CircuitBendOptions(pattern="held").noise_memory, 1)
        self.assertEqual(CircuitBendOptions(pattern="fresh").noise_memory, 0)
        self.assertEqual(CircuitBendOptions(pattern="held", memory=.4).noise_memory, .4)
        req = GenerateImageRequest(circuit_bend=dict(enabled=True, mode="plms_history", history_strength=.5,
            window_unit="percent", start=25, end=75, fade_out=20), bend_grid={"run_id": "r", "cell_id": "c"})
        self.assertEqual(GenerateImageRequest.parse_raw(req.json()), req)
        self.assertEqual(req.bend_grid["cell_id"], "c")

    def test_seed_blend_curves_window_fades_and_reverse(self):
        for curve, period, expected in [("constant", 4, [1]*5), ("progressive", 4, [0, .25, .5, .75, 1]),
                ("oscillate", 4, [0, .5, 1, .5, 0]), ("interlace", 2, [0, 0, 1, 1, 0])]:
            bend = CircuitBendOptions(mode="plms_history", history_source="seed", blend_curve=curve, blend_period=period)
            self.assertEqual([round(bend.blend_weight(i, 5), 6) for i in range(1, 6)], expected)
            reverse = bend.copy(update={"blend_start": 1, "blend_end": 0})
            self.assertEqual([round(reverse.blend_weight(i, 5), 6) for i in range(1, 6)], [1-v for v in expected])
        bend = CircuitBendOptions(blend_curve="progressive", start=3, end=7, history_strength=.5, every=2, fade_out=50)
        self.assertEqual([bend.blend_weight(i, 9) for i in range(1, 10)], [0, 0, 0, 0, .25, 0, 0, 0, 0])
        for mode in ("plms_history", "ddim"):
            req = GenerateImageRequest(circuit_bend=dict(enabled=True, mode=mode, schedule="exact", donor_seed=5,
                blend_target="trajectory", blend_curve="interlace", blend_period=2, eta=.8, seed=123))
            self.assertEqual(GenerateImageRequest.parse_raw(req.json()), req)

    def test_invalid_configuration_and_mode_gates(self):
        for bad in [dict(memory=1.1), dict(fade_in=-1), dict(fade_out=float("nan")), dict(start=1.2),
                    dict(start=0), dict(window_unit="percent", end=101), dict(history_strength=2),
                    dict(every=1.5), dict(seed=2.5)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError): CircuitBendOptions(**bad)
        options = CircuitBendOptions(enabled=True, mode="plms_history")
        options.validate_render("plms", 20)
        for name in ["heun", "ddim"]:
            with self.assertRaises(ValueError): options.validate_render(name, 20)
        self.assertFalse(CircuitBendOptions(enabled=True, mode="plms_history", history_strength=0).active)


HAS_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("sdkit") is not None


@unittest.skipUnless(HAS_RUNTIME, "Optional classic runtime (torch/sdkit) is not installed")
class BendNumericsTests(unittest.TestCase):
    def test_backend_strips_grid_metadata_before_native_generation(self):
        os.environ.setdefault("SD_UI_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui")))
        from easydiffusion import app
        from easydiffusion.backends import sdkit_common as backend
        context = SimpleNamespace(test_diffusers=False)
        with patch.object(backend, "gc"), patch("sdkit.generate.generate_images", return_value=[]) as generate:
            backend.generate_images(context, sampler_name="plms", num_inference_steps=20,
                init_image=None, control_image=None, bend_grid={"run_id": "grid"},
                circuit_bend={"enabled": True, "mode": "plms_history", "history_strength": .5})
        self.assertNotIn("circuit_bend", generate.call_args.kwargs)
        self.assertNotIn("bend_grid", generate.call_args.kwargs)

    def test_saved_text_metadata_retains_bend_and_grid(self):
        import json
        os.environ.setdefault("SD_UI_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui")))
        from easydiffusion import app
        from easydiffusion.types import RenderTaskData, ModelsData, OutputFormatData, SaveToDiskData
        from easydiffusion.utils.save_utils import get_metadata_entries_for_request
        req = GenerateImageRequest(circuit_bend=dict(enabled=True, mode="plms_history", history_strength=.5,
            window_unit="percent", start=25, end=75, fade_out=25), bend_grid={"run_id": "r", "cell_id": "2:0:0"})
        with patch.object(app, "getConfig", return_value={"backend": "ed_classic"}):
            entries = get_metadata_entries_for_request(req, RenderTaskData(), ModelsData(), OutputFormatData(), SaveToDiskData())
        self.assertEqual(json.loads(entries[0]["Circuit Bend"])["history_strength"], .5)
        self.assertEqual(json.loads(entries[0]["Bend Grid"])["cell_id"], "2:0:0")

    def test_noise_variance_correlation_and_private_rng(self):
        import torch
        global_state = torch.random.get_rng_state().clone()
        for memory in [0, .5, .95, 1]:
            source = NoiseMemory(torch.zeros(100000), 123, memory)
            first, second = source.next(), source.next()
            self.assertAlmostEqual(second.var().item(), 1, delta=.02)
            self.assertAlmostEqual((first * second).mean().item(), memory, delta=.02)
            repeated = NoiseMemory(torch.zeros(100000), 123, memory)
            self.assertTrue(torch.equal(first, repeated.next()))
            self.assertTrue(torch.equal(second, repeated.next()))
            if memory == 1: self.assertIs(first, second)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_memory_endpoints_equal_legacy_noise_sequence(self):
        import torch
        reference = torch.zeros(1, 4, 8, 8)
        generator = torch.Generator().manual_seed(123)
        source = NoiseMemory(reference, 123, 0)
        for _ in range(4):
            self.assertTrue(torch.equal(source.next(), torch.randn(reference.shape, generator=generator)))
        sigmas = torch.tensor([3., 2., 1., 0.])
        for pattern, memory in [("fresh", 0), ("held", 1)]:
            old = sample_bent(lambda x, t: x * .1, reference, sigmas, "euler",
                              CircuitBendOptions(enabled=True, pattern=pattern))
            new = sample_bent(lambda x, t: x * .1, reference, sigmas, "euler",
                              CircuitBendOptions(enabled=True, memory=memory))
            self.assertTrue(torch.equal(old, new))

    def test_freeze_owns_history_preserves_length_and_window(self):
        import torch
        freeze = PLMSHistoryFreeze(CircuitBendOptions(mode="plms_history", start=2, end=6, history_strength=.5), 10)
        short = [torch.tensor([1.])]
        self.assertIs(freeze.apply(short, 2)[0], short)
        initial = [torch.tensor([float(n)]) for n in (1, 2, 3)]
        freeze.apply(initial, 4)
        initial[0].fill_(99)
        later = [torch.tensor([float(n)]) for n in (4, 5, 6)]
        mixed, strength = freeze.apply(later, 5)
        self.assertEqual(strength, .5)
        self.assertEqual([value.item() for value in mixed], [2.5, 3.5, 4.5])
        self.assertEqual([value.item() for value in later], [4, 5, 6])
        self.assertEqual(freeze.frozen_at, 4)
        self.assertIs(freeze.apply(later, 7)[0], later)

    def test_plms_native_zero_strength_frozen_repeat_and_model_count(self):
        import torch
        from sdkit.generate.sampler import default_samplers
        def run(options=None):
            model = SimpleNamespace(device=torch.device("cpu"), num_timesteps=1000, parameterization="eps", calls=0)
            model.betas = torch.linspace(.00085, .012, 1000)
            model.alphas_cumprod = torch.cumprod(1 - model.betas, 0)
            model.alphas_cumprod_prev = torch.cat((torch.ones(1), model.alphas_cumprod[:-1]))
            def predict(x, t, c):
                model.calls += 1
                return torch.sin(x * .3) + t.reshape(-1, 1, 1, 1) * .0001
            model.apply_model = predict
            context = SimpleNamespace(models={"stable-diffusion": model})
            events = []
            def callback(x, i, info=None): events.append(info)
            noise = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(2))
            with torch.no_grad(), sampling_session(context, bend=options, enabled=options is not None):
                result = default_samplers.sample(context, sampler_name="plms", noise=noise, batch_size=1,
                    shape=(4, 8, 8), steps=10, cond=torch.ones(1, 1), uncond=torch.zeros(1, 1),
                    guidance_scale=7.5, callback=callback)
            return result, model.calls, events
        base, calls, _ = run()
        zero, zero_calls, _ = run(CircuitBendOptions(enabled=True, mode="plms_history", history_strength=0))
        self.assertTrue(torch.equal(base, zero))
        self.assertEqual(calls, zero_calls)
        options = CircuitBendOptions(enabled=True, mode="plms_history", history_strength=.5, start=4, end=8, fade_out=25)
        bent, bent_calls, events = run(options)
        repeat, _, _ = run(options)
        self.assertTrue(torch.equal(bent, repeat))
        self.assertFalse(torch.equal(base, bent))
        self.assertEqual(calls, bent_calls)
        self.assertEqual(events[3]["observation"]["history_freeze_at"], 4)
        self.assertEqual(events[-1]["observation"]["history_strength"], 0)


if __name__ == "__main__":
    unittest.main()
