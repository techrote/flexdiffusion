"""No-GPU HTTP contract checks through the real FastAPI /render route.

These tests deliberately do not claim a checkpoint/GPU render. They verify the
request's actual browser-facing transport into the queued RenderTask, including
nested circuit-bend settings, and reject invalid controls before enqueue.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))
os.environ.setdefault("SD_UI_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui")))


class RenderHttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from easydiffusion import server
        from fastapi.testclient import TestClient

        cls.server = server
        # Clean CI checkouts have no per-user plugins folder; mount the real
        # production static routes that actually exist on disk, without
        # creating user directories or changing application source.
        available_plugins = tuple(
            (directory, prefix) for directory, prefix in server.app.UI_PLUGINS_SOURCES
            if os.path.isdir(directory)
        )
        with patch.object(server.app, "UI_PLUGINS_SOURCES", available_plugins):
            server.init()
        # Without the context-manager, production startup/shutdown hooks and
        # render workers are not started by this in-process HTTP transport.
        cls.client = TestClient(server.server_api)

    def post_render(self, request):
        queued = []

        def capture(task):
            queued.append(task)
            from starlette.responses import JSONResponse
            return JSONResponse({"queued": True, "task": task.id})

        with patch.object(self.server, "enqueue_task", side_effect=capture), \
             patch.object(self.server.app, "getConfig", return_value={}), \
             patch.object(self.server.app, "save_to_config"):
            response = self.client.post("/render", json=request)
        return response, queued

    def test_sampler_mix_settings_reach_queued_task_through_http(self):
        payload = {
            "prompt": "tiny numerical smoke", "use_stable_diffusion_model": "sd-v1-4",
            "sampler_name": "plms", "num_inference_steps": 20, "seed": 2,
            "output_after_step": 10, "intermediate_representation": "both",
            "circuit_bend": {
                "enabled": True, "mode": "sampler_mix", "mix_method": "coupled",
                "mix_history": "stale", "mix_exchange": 0.75,
                "mix_output": "ddim", "mix_curve": "oscillate", "mix_period": 4,
                "eta": 0.6, "seed": 123, "schedule": "exact",
            },
            "bend_grid": {"run_id": "http-contract", "cell_id": "2:0:0"},
        }
        response, queued = self.post_render(payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["queued"])
        self.assertEqual(len(queued), 1)
        task = queued[0]
        self.assertEqual(task.render_request.sampler_name, "plms")
        self.assertEqual(task.render_request.output_after_step, 10)
        self.assertEqual(task.render_request.circuit_bend.mix_method, "coupled")
        self.assertEqual(task.render_request.circuit_bend.mix_history, "stale")
        self.assertEqual(task.render_request.circuit_bend.seed, 123)
        self.assertEqual(task.render_request.circuit_bend.eta, 0.6)
        self.assertEqual(task.render_request.bend_grid["cell_id"], "2:0:0")
        self.assertEqual(task.models_data.model_paths["stable-diffusion"], "sd-v1-4")

    def test_plain_generate_has_no_experimental_payload(self):
        response, queued = self.post_render({
            "prompt": "normal", "use_stable_diffusion_model": "sd-v1-4",
            "sampler_name": "heun", "num_inference_steps": 20,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(queued), 1)
        self.assertIsNone(queued[0].render_request.circuit_bend)
        self.assertIsNone(queued[0].render_request.bend_grid)

    def test_invalid_nested_solver_options_fail_without_queueing(self):
        for broken in (
            {"enabled": True, "mode": "sampler_mix", "mix_block": 0},
            {"enabled": True, "mode": "sampler_mix", "mix_exchange": 1.2},
            {"enabled": True, "mode": "sampler_mix", "mix_curve": "unknown"},
            {"enabled": True, "mode": "sampler_mix", "eta": "NaN"},
        ):
            with self.subTest(broken=broken):
                response, queued = self.post_render({
                    "prompt": "invalid", "use_stable_diffusion_model": "sd-v1-4",
                    "sampler_name": "plms", "circuit_bend": broken,
                })
                self.assertEqual(response.status_code, 500, response.text)
                self.assertEqual(queued, [])


    def test_http_enqueue_worker_pipeline_and_streamed_sampler_result(self):
        """Exercise /render -> RenderTask.run -> /image/stream without model weights.

        The actual API, request objects, worker pipeline and result serializer
        run unmodified. Only the model loader and sampler kernel are replaced;
        numerical correctness is covered separately by the real-native tests.
        """
        from types import SimpleNamespace
        from unittest.mock import Mock
        from easydiffusion import backend_manager, model_manager, runtime, task_manager

        payload = {
            "session_id": "sampler-worker-http",
            "prompt": "model-free worker smoke",
            "use_stable_diffusion_model": "sd-v1-4",
            "sampler_name": "plms", "num_inference_steps": 8,
            "output_after_step": 4, "seed": 2,
            "circuit_bend": {
                "enabled": True, "mode": "sampler_mix",
                "mix_method": "alternate", "mix_history": "stale",
                "mix_block": 2, "seed": 123, "schedule": "exact",
            },
        }
        response, queued = self.post_render(payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(queued), 1)
        task = queued[0]
        self.assertEqual(response.json()["stream"], f"/image/stream/{task.id}")

        invoked = {}
        context = SimpleNamespace(
            test_diffusers=False, output_step_results=[],
            sampling_metadata={}, output_step_warnings=[],
        )

        def fake_sampler(context, callback, output_type, **kwargs):
            invoked.update(kwargs)
            self.assertEqual(output_type, "base64")
            callback(None, 1, {"actual_steps": 8})
            context.sampling_metadata = {
                "actual_steps": 8, "completed_steps": 8, "status": "complete",
            }
            # Encoded 1x1 PNG; no external checkpoint or GPU is accessed.
            return ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9yKlTUwAAAAASUVORK5CYII="]

        backend = SimpleNamespace(
            ed_info={"capabilities": {
                "circuit_bend": True, "output_after_step": True,
                "trajectory_capture": True,
            }},
            set_options=Mock(), generate_images=fake_sampler,
        )
        with patch.object(runtime, "context", context), \
             patch.object(runtime, "set_vram_optimizations", return_value=False), \
             patch.object(model_manager, "resolve_model_paths"), \
             patch.object(model_manager, "reload_models_if_necessary"), \
             patch.object(model_manager, "fail_if_models_did_not_load"), \
             patch.object(backend_manager, "backend", backend), \
             patch.object(task_manager, "keep_task_alive"), \
             patch.object(self.server.app, "getConfig", return_value={}):
            task.run()

        self.assertEqual(invoked["sampler_name"], "plms")
        self.assertEqual(invoked["output_after_step"], 4)
        self.assertEqual(invoked["circuit_bend"]["mix_history"], "stale")
        self.assertEqual(invoked["circuit_bend"]["seed"], 123)
        self.assertEqual(task.response["sampling_metadata"]["completed_steps"], 8)

        with patch.object(task_manager, "get_cached_task", return_value=task):
            streamed = self.client.get(response.json()["stream"])
            self.assertEqual(streamed.status_code, 200, streamed.text)
            self.assertIn('"step": 1', streamed.text)
            self.assertIn('"sampling_metadata"', streamed.text)

            cached = self.client.get(response.json()["stream"])
            self.assertEqual(cached.status_code, 200, cached.text)
            result = cached.json()
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["output"][0]["seed"], 2)
            self.assertEqual(result["sampling_metadata"]["status"], "complete")
            self.assertEqual(result["render_request"]["circuit_bend"]["mix_block"], 2)


if __name__ == "__main__":
    unittest.main()
