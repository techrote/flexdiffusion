"""Titan-only sample policy; check Codex ownership before launching a renderer.

The Codex thread check is performed with the app's tools, outside this process.
Pass a recent JSON record: {"checked_at": <Unix seconds>, "titan_available":
true, "notes": "<which chats were checked and why the Titan is available>"}.
On Windows, WDDM also lists desktop graphics connections. A human/agent may
record individually verified desktop-only connections as desktop_processes:
[{"pid": 123, "process_name": "<exact nvidia-smi name>"}]. Never put a model
server or renderer in this list. Unknown/new connections stop the sample run.
An independently verified idle validation server may instead be recorded in
idle_servers with pid, process_name and status_url (/ping). Its Online status
is rechecked before every render. desktop_utilization_verified may be set only
after inspecting the WDDM graphics processes responsible for aggregate GPU
activity. Neither exception grants permission to use another chat's reservation.
This record does not replace checking GPU processes, nor make a cooling pause
in another chat's benchmark available time.
"""
import csv
import io
import json
import os
import subprocess
import time
from urllib.request import urlopen
from pathlib import Path

TITAN_UUID = "GPU-a8ce1477-4b52-2e8a-350e-6d54c14d1cb8"
SAMPLE_SIZE = 512
SAMPLE_SEEDS = (2, 3, 4, 5)
_desktop_processes = set()
_idle_servers = []


def validate_thread_check(record, now=None):
    now = time.time() if now is None else now
    age = now - float(record.get("checked_at", 0))
    if record.get("titan_available") is not True or not record.get("notes") or not 0 <= age <= 300:
        raise RuntimeError("A fresh Codex-chat check must confirm the Titan is available (not reserved by another chat)")
    return record


def query_smi(query):
    output = subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(output)) if row]


def assert_no_other_compute():
    # Explicitly inspected resident servers are allowed only while Online.
    # Check again before every image; a failed status query fails closed.
    idle = set()
    for server in _idle_servers:
        with urlopen(server["status_url"], timeout=3) as response:
            status = json.load(response)
        if status.get("status") != "Online":
            raise RuntimeError(f"Titan resident server is not idle: {status.get('status')}")
        idle.add((str(server["pid"]), server["process_name"]))
    processes = query_smi("--query-compute-apps=gpu_uuid,pid,process_name")
    busy = [row for row in processes if row[0] == TITAN_UUID and row[1] != str(os.getpid())
            and (row[1], row[2]) not in _desktop_processes | idle]
    if busy:
        raise RuntimeError(f"Titan has another compute process; no sample render started: {busy}")
    return processes


def prepare_titan(thread_check):
    global _desktop_processes, _idle_servers
    record = validate_thread_check(json.loads(Path(thread_check).read_text(encoding="utf-8")))
    _desktop_processes = {(str(item["pid"]), item["process_name"]) for item in record.get("desktop_processes", [])}
    _idle_servers = record.get("idle_servers", [])
    gpu = next((row for row in query_smi("--query-gpu=uuid,name,utilization.gpu,memory.used,power.limit,clocks.mem")
                if row[0] == TITAN_UUID), None)
    if gpu is None or "TITAN" not in gpu[1].upper():
        raise RuntimeError("The expected Titan RTX is not available")
    processes = assert_no_other_compute()
    if float(gpu[2]) > 5 and not record.get("desktop_utilization_verified"):
        raise RuntimeError(f"Titan is busy ({gpu[2]}% utilization); check ownership again later")
    # Must run before importing torch/sdkit. No clock or power settings change.
    os.environ["CUDA_VISIBLE_DEVICES"] = TITAN_UUID
    return {"checked_at": time.time(), "gpu": gpu, "compute_processes": processes,
            "codex_thread_check": record, "size": SAMPLE_SIZE, "seeds": SAMPLE_SEEDS}
