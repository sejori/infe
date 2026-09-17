#!/usr/bin/env python3
"""Exploratory two-pair graph-coverage control after the scheduler profiles.

Keep this separate from the prespecified six-pair instrumentation experiment.
Both configurations use stock Python scheduling, with no timing plugin.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import socket

import scheduler_investigation as runner
from treecore_sessions import command, gpu_snapshot, save


def launch_command(extra):
    def run(*args):
        if args[:2] == ("docker", "run"):
            if args[-2] != "-c":
                raise ValueError("Unexpected server launch shape")
            args = (*args[:-1], args[-1] + " " + shlex.join(extra))
        return command(*args)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--hf-cache", type=Path, required=True)
    ap.add_argument("--plugin", type=Path, required=True, help="Existing probe directory; mounted but not installed")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image", default="lmsysorg/sglang:v0.5.19")
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--control-port", type=int, default=19097)
    a = ap.parse_args()
    a.hf_cache = a.hf_cache.resolve(strict=True)
    a.plugin = a.plugin.resolve(strict=True)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    a.model = "Qwen/Qwen2.5-1.5B-Instruct"
    a.revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    a.concurrency, a.workloads, a.pilot_count = 256, ["short"], 0
    device = command("nvidia-smi", "-i", a.gpu, "--query-gpu=uuid", "--format=csv,noheader")
    lock = open(f"/tmp/infe-treecore-{device}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for port in [a.port, a.control_port]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    image = json.loads(command("docker", "image", "inspect", a.image))[0]
    configurations = {"default": [], "decode_graph_256": ["--cuda-graph-max-bs-decode", "256"]}
    orders = [["default", "decode_graph_256"], ["decode_graph_256", "default"]]
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name("treecore_sessions.py")]
    save(a.output / "manifest.json", {
        "purpose": "Exploratory graph-coverage control; two balanced pairs, selected after inspecting traces",
        "image_id": image["Id"], "image_digests": image["RepoDigests"],
        "model": a.model, "revision": a.revision, "gpu": gpu_snapshot(a.gpu),
        "case": "short_c256", "warmup_requests": 256, "measured_requests": 4096,
        "orders": orders, "configurations": configurations,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    })
    sessions = []
    for pair, order in enumerate(orders):
        for position, configuration in enumerate(order):
            extra = configurations[configuration]
            runner.command = launch_command(extra)
            result = runner.run_session(a, image["Id"], pair, position, "stock")
            result.update(configuration=configuration, extra_server_args=extra)
            save(a.output / f"pair-{pair:02d}-{position}-stock/session.json", result)
            sessions.append(result)
    paired = []
    for pair in range(2):
        values = {s["configuration"]: s["workloads"][0]["summary"] for s in sessions if s["pair"] == pair}
        paired.append({"pair": pair, "summaries": values,
                       "graph_effect_percent": {key: 100 * (values["decode_graph_256"][key] / values["default"][key] - 1)
                                                for key in ("wall_s", "requests_per_s", "e2e_p50_ms", "ttft_p50_ms", "cpu_percent")}})
    save(a.output / "comparison.json", {"independent_pairs": 2, "exploratory": True, "paired": paired})
    print("Graph-coverage control complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f"Graph check failed ({type(exc).__name__}); inspect retained artifacts.") from None
