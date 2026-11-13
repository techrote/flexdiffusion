"""Contract tests plus optional tests against the installed classic solvers."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch
import json
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))

from easydiffusion.circuit_bend import CircuitBendOptions, ExperimentalSamplingStopped, sample_bent
from easydiffusion.output_steps import OutputAfterStepCollector, flatten_output_step_results
from easydiffusion.sampler_adapters import (
    SamplerSession, UNIPC_SAMPLERS, _active, sampling_session,
)
from easydiffusion.types import GenerateImageRequest


class CaptureContractTests(unittest.TestCase):
    def test_native_schedule_retains_requested_boundary_at_thirty(self):
        collector = OutputAfterStepCollector(10, 30, "sparse:10,30|endpoints=1")
        collector.set_actual_steps(31)
        self.assertTrue(collector.wants_callback_index(30))
        self.assertFalse(collector.wants_callback_index(31))
        self.assertEqual(collector.requested_steps, 30)
        results = flatten_output_step_results([
            {"step": 30, "representation": "denoised", "images": ["image"],
             "metadata": {"estimate_basis": "plms_history", "requested_steps": 30}},
        ], 2, 31)
        self.assertEqual(results[0][2]["total_steps"], 31)
        self.assertEqual(results[0][2]["estimate_basis"], "plms_history")

    def test_bend_request_roundtrip_and_validation(self):
        req = GenerateImageRequest(circuit_bend={"enabled": True, "pattern": "held", "seed": 42})
        self.assertEqual(req.dict()["circuit_bend"]["pattern"], "held")
        for bad in ({"churn": float("nan")}, {"noise": float("inf")}, {"start": 5, "end": 2},
                    {"pattern": "unknown"}, {"every": 0}, {"seed": -1}, {"unknown": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CircuitBendOptions(**bad)
        with self.assertRaisesRegex(ValueError, "Euler and Heun"):
            CircuitBendOptions(enabled=True).validate_render("plms", 20)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            CircuitBendOptions(enabled=True, end=30).validate_render("heun", 20)
        self.assertFalse(CircuitBendOptions(enabled=True, churn=0).active)


HAS_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("sdkit") is not None


@unittest.skipUnless(HAS_RUNTIME, "Optional classic runtime (torch/sdkit) is not installed")
class NativeSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from sdkit.generate.sampler import default_samplers, k_samplers, unipc_samplers
        cls.torch = torch
        cls.modules = {"ddim": default_samplers, "plms": default_samplers,
                       "euler": k_samplers, "heun": k_samplers,
                       **{name: unipc_samplers for name in UNIPC_SAMPLERS}}

    def context(self):
        torch = self.torch
        model = SimpleNamespace(device=torch.device("cpu"), num_timesteps=1000, parameterization="eps", calls=0)
        model.betas = torch.linspace(0.00085, 0.012, 1000)
        model.alphas_cumprod = torch.cumprod(1 - model.betas, 0)
        model.alphas_cumprod_prev = torch.cat((torch.ones(1), model.alphas_cumprod[:-1]))

        def apply_model(x, t, cond):
            model.calls += 1
            return x * 0.1 + t.reshape(-1, 1, 1, 1) * 0.00001

        model.apply_model = apply_model
        return SimpleNamespace(models={"stable-diffusion": model}, sampling_metadata={})

    def run_sampler(self, name, steps=5, adapted=False, batch=1, bend=None, seed=19):
        torch = self.torch
        context = self.context()
        from easydiffusion.sampler_adapters import seed_noise_like
        noise = seed_noise_like(torch.zeros(batch, 4, 8, 8), seed)
        collector = OutputAfterStepCollector(1, steps, "both")
        events = []

        def callback(x, i, info=None):
            events.append((i, x.clone(), info))
            if info is not None:
                collector.capture_step(i, x, info["denoised"], info["observation"])

        kwargs = dict(sampler_name=name, noise=noise, batch_size=batch, shape=(4, 8, 8), steps=steps,
                      cond=torch.ones(batch, 1), uncond=torch.zeros(batch, 1), guidance_scale=7.5,
                      callback=callback)
        torch.manual_seed(4)
        with torch.no_grad(), sampling_session(context, collector, bend, enabled=adapted):
            result = self.modules[name].sample(context, **kwargs)
        return result, events, collector, context

    def test_all_families_preserve_native_final_and_model_call_count(self):
        for name in self.modules:
            with self.subTest(name=name):
                baseline, _, _, baseline_context = self.run_sampler(name)
                result, events, collector, context = self.run_sampler(name, adapted=True)
                self.assertTrue(self.torch.equal(baseline, result))
                self.assertEqual(baseline_context.models["stable-diffusion"].calls,
                                 context.models["stable-diffusion"].calls)
                wanted = list(range(1, 5)) if name in UNIPC_SAMPLERS else list(range(0, 5))
                self.assertEqual([event[0] for event in events if event[0] < 5], wanted)
                self.assertEqual(len(collector.pop_snapshots()), 8)
                self.assertEqual(context.sampling_metadata["status"], "complete")

    def test_ddim_plms_thirty_and_invalid_schedule(self):
        for name in ("ddim", "plms"):
            with self.subTest(name=name):
                base, _, _, _ = self.run_sampler(name, 30)
                result, events, collector, context = self.run_sampler(name, 30, adapted=True)
                self.assertTrue(self.torch.equal(base, result))
                self.assertEqual(len(events), 31)
                self.assertEqual(collector.total_steps, 31)
                self.assertEqual(context.sampling_metadata["actual_steps"], 31)
                with self.assertRaisesRegex(ValueError, "native schedule is invalid"):
                    self.run_sampler(name, 9, adapted=True)

    def test_unipc_phase_and_batch_limits(self):
        for name in UNIPC_SAMPLERS:
            with self.subTest(name=name):
                _, events, collector, _ = self.run_sampler(name, adapted=True)
                self.assertEqual([event[0] for event in events], [1, 2, 3, 4, 5])
                self.assertEqual(events[0][2]["observation"]["estimate_basis"], "predictor_candidate")
                self.assertIsNone(events[-1][2]["denoised"])
                if name == "unipc_snr_2":
                    with self.assertRaisesRegex(ValueError, "one image per batch"):
                        self.run_sampler(name, adapted=True, batch=2)
                else:
                    base, _, _, _ = self.run_sampler(name, batch=2)
                    result, _, _, _ = self.run_sampler(name, adapted=True, batch=2)
                    self.assertTrue(self.torch.equal(base, result))

    def test_sampler_mix_single_sampler_limits_and_coupling_zero(self):
        for name in ('plms', 'ddim'):
            native, _, _, ctx = self.run_sampler(name)
            bend = CircuitBendOptions(enabled=True, mode='sampler_mix', eta=0, mix_first=name, mix_block=100)
            mixed, events, _, mixed_ctx = self.run_sampler('plms', adapted=True, bend=bend)
            self.assertTrue(self.torch.equal(native, mixed))
            self.assertEqual([e[0] for e in events], [1,2,3,4,5])
            self.assertEqual(ctx.models['stable-diffusion'].calls, mixed_ctx.models['stable-diffusion'].calls)
            self.assertTrue(all(e[2]['observation']['active_sampler'] == name for e in events))
            coupled = bend.copy(update=dict(mix_method='coupled', mix_exchange=0, mix_output=name))
            result, _, _, cctx = self.run_sampler('plms', adapted=True, bend=coupled)
            self.assertTrue(self.torch.equal(native, result))
            self.assertEqual(cctx.models['stable-diffusion'].calls, 11)

    def test_sampler_mix_alternation_history_and_timestep_alignment(self):
        expected_histories = {'shared':[0,1,2,3,3], 'stale':[0,1,1,2,2], 'reset':[0,0,0,0,0]}
        for policy, lengths in expected_histories.items():
            bend = CircuitBendOptions(enabled=True, mode='sampler_mix', eta=0, mix_history=policy)
            result, events, _, ctx = self.run_sampler('plms', adapted=True, bend=bend)
            repeat = self.run_sampler('ddim', adapted=True, bend=bend)[0]
            self.assertTrue(self.torch.equal(result, repeat))
            obs = [e[2]['observation'] for e in events]
            self.assertEqual([o['active_sampler'] for o in obs], ['plms','ddim','plms','ddim','plms'])
            self.assertEqual([o['history_length'] for o in obs], lengths)
            self.assertEqual([o['timestep'] for o in obs], [801,601,401,201,1])
            self.assertEqual(ctx.models['stable-diffusion'].calls, 8 if policy == 'reset' else 6)
        options = bend.copy(update=dict(mix_first='ddim', mix_block=2, mix_history='shared'))
        _, events, _, _ = self.run_sampler('ddim', adapted=True, bend=options)
        self.assertEqual([e[2]['observation']['active_sampler'] for e in events], ['ddim','ddim','plms','plms','ddim'])

    def test_sampler_mix_exchange_is_simultaneous_and_reproducible(self):
        bend = CircuitBendOptions(enabled=True, mode='sampler_mix', eta=.7, seed=123,
            mix_method='coupled', mix_exchange=.5, mix_output='plms')
        a = self.run_sampler('plms', adapted=True, bend=bend)[0]
        b = self.run_sampler('plms', adapted=True, bend=bend.copy(update={'mix_output':'ddim'}))[0]
        self.assertTrue(self.torch.equal(a, b))  # half exchange joins both latents exactly
        repeated = self.run_sampler('plms', adapted=True, bend=bend)[0]
        self.assertTrue(self.torch.equal(a, repeated))
        changed = self.run_sampler('plms', adapted=True, bend=bend.copy(update={'seed':456}))[0]
        self.assertFalse(self.torch.equal(a, changed))
        for branch, swapped in (('plms','ddim'),('ddim','plms')):
            swap = bend.copy(update=dict(eta=0,mix_exchange=1,mix_output=branch))
            actual = self.run_sampler('plms',steps=1,adapted=True,bend=swap)[0]
            expected = self.run_sampler(swapped,steps=1)[0]
            self.assertTrue(self.torch.equal(actual,expected))
        for curve in ('constant','progressive','oscillate'):
            opts = bend.copy(update=dict(mix_curve=curve, start=2, end=4, mix_period=4))
            result, events, _, _ = self.run_sampler('plms', adapted=True, bend=opts)
            self.assertTrue(self.torch.isfinite(result).all())
            self.assertEqual([e[2]['observation']['exchange_weight'] for e in events],
                             [opts.exchange_weight(i,5) for i in range(1,6)])

    def test_sampler_mix_incompatible_model_and_finite_recovery(self):
        from easydiffusion.sampler_mixing import sample_mixed
        bend = CircuitBendOptions(enabled=True, mode='sampler_mix')
        with self.assertRaisesRegex(ValueError, 'epsilon-prediction'):
            sample_mixed(SimpleNamespace(parameterization='v'), {}, bend, False, 5, lambda *a, **k:None)
        context = self.context()
        context.models['stable-diffusion'].apply_model = lambda x,t,c: x * float('nan')
        with patch.object(self, 'context', return_value=context), self.assertRaises(ExperimentalSamplingStopped) as caught:
            self.run_sampler('plms', adapted=True, bend=bend)
        self.assertEqual(caught.exception.completed_steps, 0)
        self.assertTrue(self.torch.isfinite(caught.exception.latent).all())

    def test_sampler_mix_passes_the_updated_latent_between_solvers(self):
        from sdkit.generate.sampler import default_samplers
        from easydiffusion.sampler_adapters import seed_noise_like
        model = self.context().models['stable-diffusion']
        plms, ddim = (default_samplers.samplers[name](model) for name in ('plms','ddim'))
        for sampler in (plms,ddim): sampler.make_schedule(5, ddim_eta=0, verbose=False)
        torch = self.torch
        x = seed_noise_like(torch.zeros(1,4,8,8),19)
        history, states = [], []
        cond = torch.ones(1,1)
        cfg = dict(unconditional_guidance_scale=7.5, unconditional_conditioning=torch.zeros(1,1))
        times = list(reversed(plms.ddim_timesteps))
        for i, timestep in enumerate(times):
            t = torch.full((1,),int(timestep),dtype=torch.long)
            if i % 2 == 0:
                x, clean, eps = plms.p_sample_plms(x,cond,t,index=4-i,old_eps=history,
                    t_next=torch.full_like(t,int(times[min(i+1,4)])),**cfg)
            else:
                # Test model ignores conditioning: native guided eps equals this prediction.
                eps = x * .1 + t.reshape(-1,1,1,1) * .00001
                x, clean = ddim.p_sample_ddim(x,cond,t,index=4-i,**cfg)
            history = (history + [eps])[-3:]
            states.append(x.clone())
        bend = CircuitBendOptions(enabled=True,mode='sampler_mix',eta=0)
        result, events, _, _ = self.run_sampler('plms',adapted=True,bend=bend)
        self.assertTrue(torch.equal(result,x))
        for event, state in zip(events,states): self.assertTrue(torch.equal(event[1],state))

    def test_exact_schedule_counts_boundaries_and_native_equivalence(self):
        from easydiffusion.sampler_adapters import exact_ldm_timesteps
        for count in range(1, 1001):
            timetable = exact_ldm_timesteps("uniform", count, 1000)
            self.assertEqual(len(set(timetable)), count)
            self.assertTrue(0 <= min(timetable) <= max(timetable) < 1000)
        for name in ("ddim", "plms"):
            mode = "ddim" if name == "ddim" else "plms_history"
            bend = CircuitBendOptions(enabled=True, mode=mode, schedule="exact", eta=0, history_strength=0)
            for count in (9, 16, 27, 30):
                result, events, collector, context = self.run_sampler(name, count, True, bend=bend)
                self.assertEqual(len(events), count)
                self.assertEqual(collector.total_steps, count)
                self.assertTrue(self.torch.isfinite(result).all())
                self.assertEqual(context.sampling_metadata["schedule"], "exact")
            ordinary = self.run_sampler(name, 20)[0]
            exact = self.run_sampler(name, 20, True, bend=bend)[0]
            self.assertTrue(self.torch.equal(ordinary, exact))

    def test_plms_second_seed_endpoints_repeatability_and_nfe(self):
        for batch in (1, 2):
            base, _, _, context = self.run_sampler("plms", batch=batch)
            donor = self.run_sampler("plms", batch=batch, seed=23)[0]
            for target in ("history", "trajectory", "independent"):
                options = dict(enabled=True, mode="plms_history", history_source="seed", donor_seed=23,
                               blend_target=target)
                zero, _, _, zero_context = self.run_sampler("plms", adapted=True, batch=batch,
                    bend=CircuitBendOptions(**options, history_strength=0))
                self.assertTrue(self.torch.equal(zero, base))
                self.assertEqual(zero_context.models["stable-diffusion"].calls, context.models["stable-diffusion"].calls)
                mixed, events, _, mixed_context = self.run_sampler("plms", adapted=True, batch=batch,
                    bend=CircuitBendOptions(**options))
                repeated = self.run_sampler("plms", adapted=True, batch=batch, bend=CircuitBendOptions(**options))[0]
                self.assertTrue(self.torch.equal(mixed, repeated))
                self.assertFalse(self.torch.equal(mixed, base))
                self.assertEqual(mixed_context.models["stable-diffusion"].calls, 2 * context.models["stable-diffusion"].calls)
                self.assertEqual(events[0][2]["observation"]["donor_seed"], 23)
                if target in ("trajectory", "independent"): self.assertTrue(self.torch.equal(mixed, donor))
        for curve in ("progressive", "oscillate", "interlace"):
            bend = CircuitBendOptions(**options, blend_curve=curve, blend_period=2)
            a, events, _, _ = self.run_sampler("plms", adapted=True, bend=bend)
            b = self.run_sampler("plms", adapted=True, bend=bend)[0]
            self.assertTrue(self.torch.equal(a, b))
            self.assertEqual([e[2]["observation"]["blend_weight"] for e in events],
                             [bend.blend_weight(i, 5) for i in range(1, 6)])

    def test_ddim_eta_private_noise_window_and_zero_endpoint(self):
        base = self.run_sampler("ddim")[0]
        zero = self.run_sampler("ddim", adapted=True, bend=CircuitBendOptions(enabled=True, mode="ddim", eta=0))[0]
        self.assertTrue(self.torch.equal(base, zero))
        for memory in (0, .5, 1):
            bend = CircuitBendOptions(enabled=True, mode="ddim", eta=.7, seed=123, memory=memory, start=2, end=4)
            a, events, _, _ = self.run_sampler("ddim", adapted=True, bend=bend)
            b = self.run_sampler("ddim", adapted=True, bend=bend)[0]
            different = self.run_sampler("ddim", adapted=True, bend=bend.copy(update={"seed": 456}))[0]
            self.assertTrue(self.torch.equal(a, b))
            self.assertFalse(self.torch.equal(a, base))
            self.assertFalse(self.torch.equal(a, different))
            self.assertEqual([e[2]["observation"]["eta"] for e in events], [0, .7, .7, .7, 0])

    def test_plms_independent_interlace_returns_to_untouched_seed_a(self):
        base, a_events, _, _ = self.run_sampler("plms", adapted=True)
        _, b_events, _, _ = self.run_sampler("plms", adapted=True, seed=23)
        bend = CircuitBendOptions(enabled=True, mode="plms_history", history_source="seed", donor_seed=23,
            blend_target="independent", blend_curve="interlace", blend_period=1)
        result, mixed_events, _, _ = self.run_sampler("plms", adapted=True, bend=bend)
        self.assertTrue(self.torch.equal(result, base))
        for i, event in enumerate(mixed_events):
            expected = (a_events if i % 2 == 0 else b_events)[i][2]["denoised"]
            self.assertTrue(self.torch.equal(event[2]["denoised"], expected))
        bend = bend.copy(update={"blend_curve": "oscillate", "blend_period": 4})
        _, mixed_events, _, _ = self.run_sampler("plms", adapted=True, bend=bend)
        from easydiffusion.sampler_adapters import mix_tensor
        for i, event in enumerate(mixed_events):
            expected = mix_tensor(a_events[i][2]["denoised"], b_events[i][2]["denoised"], bend.blend_weight(i+1, 5))
            self.assertTrue(self.torch.equal(event[2]["denoised"], expected))

    def test_instance_overrides_are_isolated_between_threads(self):
        def task(index):
            bend = CircuitBendOptions(enabled=True, mode="ddim", schedule="exact", eta=.6, seed=index)
            return self.run_sampler("ddim", 9, True, bend=bend)[0]
        serial = [task(i) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            parallel = list(pool.map(task, range(2)))
        for a, b in zip(serial, parallel): self.assertTrue(self.torch.equal(a, b))
        with self.assertRaisesRegex(ValueError, "native schedule is invalid"):
            self.run_sampler("ddim", 9, True)

    def test_dispatch_context_cleanup_on_cancel_and_thread_isolation(self):
        def task(index):
            context = self.context()
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                with sampling_session(context):
                    self.assertIs(_active.get().context, context)
                    raise RuntimeError("cancel")
            self.assertIsNone(_active.get())
            return index
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(task, range(4))), [0, 1, 2, 3])

    def test_zero_churn_is_exact_and_bends_are_reproducible(self):
        for name in ("euler", "heun"):
            with self.subTest(name=name):
                base, _, _, _ = self.run_sampler(name)
                zero, _, _, _ = self.run_sampler(name, adapted=True, bend=CircuitBendOptions(enabled=True, churn=0))
                self.assertTrue(self.torch.equal(base, zero))
                options = CircuitBendOptions(enabled=True, pattern="held", seed=123, start=2, end=4, every=2)
                bent, events, _, _ = self.run_sampler(name, adapted=True, bend=options)
                repeat, _, _, _ = self.run_sampler(name, adapted=True, bend=options)
                self.assertTrue(self.torch.equal(bent, repeat))
                self.assertFalse(self.torch.equal(base, bent))
                phases = [event[2]["observation"]["observation_phase"] for event in events]
                self.assertEqual(phases, ["before_update", "after_churn", "before_update", "after_churn", "before_update"])

    def test_bend_private_rng_and_finite_partial_on_divergence(self):
        torch = self.torch
        x = torch.ones(1, 4, 8, 8)
        sigmas = torch.tensor([3.0, 2.0, 1.0, 0.0])
        state = torch.random.get_rng_state().clone()
        options = CircuitBendOptions(enabled=True)
        sample_bent(lambda x, sigma: x * 0.1, x, sigmas, "euler", options)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        with self.assertRaises(ExperimentalSamplingStopped) as caught:
            sample_bent(lambda x, sigma: x * float("nan"), x, sigmas, "heun", options)
        self.assertTrue(torch.isfinite(caught.exception.latent).all())
        self.assertEqual(caught.exception.completed_steps, 0)

    def test_task_response_preserves_native_counts_and_bend_settings(self):
        os.environ.setdefault("SD_UI_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui")))
        from easydiffusion import app
        from easydiffusion.tasks import render_images
        from easydiffusion.types import RenderTaskData, ModelsData, OutputFormatData, SaveToDiskData

        context = SimpleNamespace(output_step_results=[
            {"step": 30, "representation": "denoised", "images": ["intermediate"],
             "metadata": {"estimate_basis": "plms_history", "requested_steps": 30, "actual_steps": 31}},
        ], sampling_metadata={"actual_steps": 31, "completed_steps": 31, "status": "complete"},
            output_step_warnings=[])
        req = GenerateImageRequest(sampler_name="plms", num_inference_steps=30)
        output = queue.Queue()
        with patch.object(render_images, "print_task_info"), patch.object(
            render_images, "make_images_internal", return_value=(["final"], [2])
        ):
            response = render_images.make_images(context, req, RenderTaskData(), ModelsData(),
                          OutputFormatData(), SaveToDiskData(), output, [], lambda: None, None)
        self.assertEqual([item["output_step"] for item in response["output"]], [30, 31])
        self.assertEqual(response["output"][0]["estimate_basis"], "plms_history")
        self.assertEqual(response["output"][-1]["requested_steps"], 30)
        self.assertEqual(json.loads(output.get())["sampling_metadata"]["actual_steps"], 31)
        self.assertEqual(context.output_step_results, [])


if __name__ == "__main__":
    unittest.main()
