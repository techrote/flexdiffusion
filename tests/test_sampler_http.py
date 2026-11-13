"""No-GPU HTTP contract checks through the real FastAPI /render route.

These tests deliberately do not claim a checkpoint/GPU render. They verify the
request's actual browser-facing transport into the queued RenderTask, including
nested circuit-bend settings, and reject invalid controls before enqueue.
"""

import json
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
        # UI mounts require a user plugins folder that a clean CI checkout
        # intentionally lacks. Only static-file mounts are stubbed; real route
        # registration, Pydantic parsing and RenderTask construction are used.
        with patch.object(server.server_api, "mount"):
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


if __name__ == "__main__":
    unittest.main()
