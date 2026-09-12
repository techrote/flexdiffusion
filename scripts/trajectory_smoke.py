#!/usr/bin/env python3
"""Run the first FlexDiffusion SD1.x trajectory GPU smoke matrix.

This talks to an already-running Easy Diffusion server over its local HTTP API.
It intentionally uses only the Python standard library, so it can be launched
from an ordinary system Python while Easy Diffusion owns its own environment.

Prerequisites:
- FlexDiffusion branch with trajectory capture code installed/running.
- Settings -> Engine to use -> v2.0 (`ed_classic`), followed by restart.
- One SD1.x model known to Easy Diffusion.

Example (PowerShell, one command):
    py scripts/trajectory_smoke.py --model "sd-v1-4.ckpt" --trajectory-root "C:\\FlexDiffusionTrajectories"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _request_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    return json.loads(body)


def _output_bytes(value: str) -> bytes:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return value.encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_task(base_url: str, session_id: str, task_id: int, stream_path: str, timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    status = "pending"
    while time.time() < deadline:
        ping_url = f"{base_url}/ping?{urllib.parse.urlencode({'session_id': session_id})}"
        ping = _request_json(ping_url)
        tasks = ping.get("tasks", {})
        status = tasks.get(str(task_id), tasks.get(task_id, status))
        if status == "error":
            raise RuntimeError(f"Easy Diffusion task {task_id} entered error state")
        if status in ("completed", "stopped"):
            break
        time.sleep(0.25)
    else:
        raise TimeoutError(f"task {task_id} did not complete within {timeout:.1f}s (last status={status})")

    # Once complete and its queue has drained, Easy Diffusion serves the cached
    # final response as ordinary JSON from the stream endpoint. Retry briefly
    # because task status and cache visibility can cross by a few milliseconds.
    stream_url = base_url + stream_path
    for _ in range(40):
        try:
            return _request_json(stream_url, timeout=30.0)
        except RuntimeError as exc:
            if "HTTP 425" not in str(exc):
                raise
            time.sleep(0.1)
    raise RuntimeError(f"final response for task {task_id} never became readable")


def _submit(base_url: str, payload: Dict[str, Any], timeout: float) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    queued = _request_json(f"{base_url}/render", payload)
    task_id = int(queued["task"])
    final = _wait_for_task(base_url, payload["session_id"], task_id, queued["stream"], timeout)
    return final, time.perf_counter() - started


def _new_manifest(before: Iterable[str], root: str) -> str:
    after = {
        os.path.join(root, entry, "manifest.json")
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, "manifest.json"))
    }
    candidates = sorted(after - set(before), key=os.path.getmtime)
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one new trajectory manifest, found {len(candidates)}")
    return candidates[0]


def _manifest_set(root: str) -> set:
    if not os.path.isdir(root):
        return set()
    return {
        os.path.join(root, entry, "manifest.json")
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, "manifest.json"))
    }


def _verify_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_dir = os.path.dirname(path)
    errors = []
    artifact_count = 0
    artifact_bytes = 0

    for checkpoint in manifest.get("checkpoints", []):
        entries = []
        if checkpoint.get("latent"):
            entries.append(checkpoint["latent"])
        entries.extend(checkpoint.get("previews") or [])
        for entry in entries:
            artifact_count += 1
            artifact_path = os.path.join(run_dir, entry["path"].replace("/", os.sep))
            if not os.path.isfile(artifact_path):
                errors.append(f"missing {entry['path']}")
                continue
            actual_size = os.path.getsize(artifact_path)
            actual_hash = _sha256_file(artifact_path)
            if actual_size != entry.get("bytes"):
                errors.append(f"size mismatch {entry['path']}: {actual_size} != {entry.get('bytes')}")
            if actual_hash != entry.get("sha256"):
                errors.append(f"hash mismatch {entry['path']}")
            artifact_bytes += actual_size

    if artifact_bytes != manifest.get("artifact_bytes", artifact_bytes):
        errors.append(
            f"manifest artifact byte total mismatch: files={artifact_bytes}, manifest={manifest.get('artifact_bytes')}"
        )

    leftovers = []
    for dirpath, _, filenames in os.walk(run_dir):
        leftovers.extend(os.path.join(dirpath, name) for name in filenames if name.endswith(".tmp"))
    if leftovers:
        errors.append(f"temporary files remain: {leftovers}")

    return {
        "manifest": manifest,
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
        "errors": errors,
    }


def _trajectory_for(mode: str, root: str, steps: int) -> Optional[Dict[str, Any]]:
    if mode == "baseline":
        return None

    sparse = "1,25%,50%,75%,100%"
    config: Dict[str, Any] = {
        "enabled": True,
        "capture_schedule": sparse,
        "persistence_mode": mode,
        "output_root": root,
        "project_name": f"smoke-{mode}",
        "preview_format": "png",
        "preview_quality": 90,
        "writer_queue_size": 2,
    }
    if mode == "dense":
        config["capture_schedule"] = f"1-{steps}:1"
        config["persistence_mode"] = "hybrid"
        config["project_name"] = "smoke-dense"
        config["writer_queue_size"] = 1
    elif mode == "budget":
        config["capture_schedule"] = f"1-{steps}:1"
        config["persistence_mode"] = "preview"
        config["project_name"] = "smoke-budget"
        config["preview_format"] = "png"
        config["writer_queue_size"] = 1
        config["storage_budget_mb"] = 1
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9000", help="running Easy Diffusion server")
    parser.add_argument("--model", required=True, help="SD1.x model name/path as shown by Easy Diffusion")
    parser.add_argument("--vae", default=None, help="optional VAE model name/path")
    parser.add_argument("--trajectory-root", required=True, help="absolute directory for test trajectory runs")
    parser.add_argument("--prompt", default="a chrome mechanical moth resting on a fern, studio photograph")
    parser.add_argument("--negative-prompt", default="text, watermark")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--sampler", default="euler", help="use a deterministic sampler for the equality gate")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--vram", choices=("low", "balanced", "high"), default="low")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--modes",
        default="baseline,latent,preview,hybrid,dense,budget",
        help="comma-separated subset of baseline,latent,preview,hybrid,dense,budget",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    root = os.path.abspath(args.trajectory_root)
    os.makedirs(root, exist_ok=True)

    config = _request_json(f"{base_url}/get/app_config")
    backend = config.get("backend")
    if backend != "ed_classic":
        print(
            f"ERROR: running backend is {backend!r}; set Settings -> Engine to use -> v2.0 (ed_classic), save and restart.",
            file=sys.stderr,
        )
        return 2

    requested_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    valid_modes = {"baseline", "latent", "preview", "hybrid", "dense", "budget"}
    unknown = [mode for mode in requested_modes if mode not in valid_modes]
    if unknown:
        parser.error(f"unknown modes: {', '.join(unknown)}")
    if not requested_modes or requested_modes[0] != "baseline":
        requested_modes.insert(0, "baseline")

    common: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_outputs": 1,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
        "sampler_name": args.sampler,
        "use_stable_diffusion_model": args.model,
        "use_vae_model": args.vae,
        "stream_image_progress": False,
        "vram_usage_level": args.vram,
        "output_format": "png",
        "output_lossless": True,
    }

    baseline_hash = None
    failures: List[str] = []
    summaries = []

    for mode in requested_modes:
        session_id = f"trajectory-smoke-{mode}-{int(time.time() * 1000)}"
        payload = dict(common)
        payload["session_id"] = session_id
        trajectory = _trajectory_for(mode, root, args.steps)
        before = _manifest_set(root)
        if trajectory is not None:
            payload["trajectory"] = trajectory

        print(f"[{mode}] submitting...")
        final, elapsed = _submit(base_url, payload, args.timeout)
        outputs = final.get("output") or []
        if len(outputs) != 1:
            failures.append(f"{mode}: expected 1 output, got {len(outputs)}")
            continue
        image_hash = _sha256_bytes(_output_bytes(outputs[0]["data"]))
        if mode == "baseline":
            baseline_hash = image_hash
        elif baseline_hash is not None and image_hash != baseline_hash:
            failures.append(f"{mode}: final image hash differs from baseline")

        manifest_summary = None
        if trajectory is not None:
            try:
                manifest_path = _new_manifest(before, root)
                manifest_summary = _verify_manifest(manifest_path)
                if manifest_summary["errors"]:
                    failures.extend(f"{mode}: {error}" for error in manifest_summary["errors"])
                expected_capture_error = mode == "budget"
                manifest_errors = manifest_summary["manifest"].get("errors") or []
                if expected_capture_error and not manifest_errors:
                    failures.append("budget: expected a capture/storage-budget error but manifest has none")
                if not expected_capture_error and manifest_errors:
                    failures.append(f"{mode}: unexpected manifest errors: {manifest_errors}")
            except Exception as exc:
                failures.append(f"{mode}: manifest verification failed: {type(exc).__name__}: {exc}")

        summaries.append(
            {
                "mode": mode,
                "seconds": round(elapsed, 3),
                "image_sha256": image_hash,
                "manifest_status": manifest_summary["manifest"].get("status") if manifest_summary else None,
                "checkpoints": len(manifest_summary["manifest"].get("checkpoints", [])) if manifest_summary else 0,
                "artifacts": manifest_summary["artifact_count"] if manifest_summary else 0,
                "artifact_bytes": manifest_summary["artifact_bytes"] if manifest_summary else 0,
            }
        )
        print(json.dumps(summaries[-1], indent=2))

    print("\n=== trajectory smoke summary ===")
    print(json.dumps(summaries, indent=2))
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nPASS: HTTP render outputs and persisted trajectory artefacts passed the smoke matrix.")
    print("Note: peak VRAM/RAM must still be measured externally during the dense run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
