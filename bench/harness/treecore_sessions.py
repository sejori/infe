#!/usr/bin/env python3
"""Counterbalanced, independent-session TreeCore benchmark (stdlib only).

Each pair contains a fresh Python and Rust server on the same pinned image/GPU.
Requests and rounds are nested observations, never independent replicates.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid

from e2e_tool_stream import run_level, pct


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def public_config(config):
    """Keep experimental settings, not local paths or device identifiers."""
    return {key: value for key, value in config.items()
            if key not in {"hf_cache", "output", "gpu"}}


def redact_log(text, hostname):
    """Remove machine identifiers from engine logs before writing artifacts."""
    text = re.sub(r"GPU-[0-9a-fA-F-]{36}", "GPU-REDACTED", text)
    text = re.sub(r"docker-[0-9a-f]{64}", "docker-REDACTED", text)
    text = re.sub(r"infe-treecore-[0-9a-f]{12}", "infe-treecore-REDACTED", text)
    text = re.sub(r"/(?:home|Users)/[^\s'\";,]+", "<local-path>", text)
    if hostname:
        text = re.sub(r"(?<![\w.-])(?:[\w.-]+@)?" + re.escape(hostname)
                      + r"(?![\w.-])", "<benchmark-host>", text)
    return text


def schedule(pairs, seed):
    if pairs < 2 or pairs % 2:
        raise ValueError("pairs must be positive and even (balanced arm order)")
    orders = [["python", "rust"], ["rust", "python"]] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return orders


def validate(rows):
    for row in rows:
        calls = row["tool_calls_parity"]
        if (row["error"] or row["ttft_ms"] is None or len(calls) != 2
                or {c["name"] for c in calls} != {"get_weather", "get_time"}
                or not all(c["args_json_ok"] and c["has_id"] for c in calls)):
            raise RuntimeError(f"Request/parity failure: {row}")


def summarize_round(rows, wall):
    validate(rows)
    itls = [x for r in rows for x in r["itl_ms"]]
    return {
        "e2e_p50": pct([r["e2e_ms"] for r in rows], .5),
        "ttft_p50": pct([r["ttft_ms"] for r in rows], .5),
        "itl_p50": pct(itls, .5), "itl_p99": pct(itls, .99),
        # First to last meaningful SSE chunk, NOT span divided by delta count.
        "stream_span_p50": pct([sum(r["itl_ms"]) for r in rows], .5),
        "deltas_per_request": statistics.mean(r["tool_deltas"] for r in rows),
        "requests_per_s": len(rows) / (wall / 1000),
    }


def cpu_usage(path):
    values = dict(line.split() for line in path.read_text().splitlines())
    return int(values["usage_usec"])


def gpu_snapshot(gpu):
    fields = "name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"
    values = command("nvidia-smi", "-i", gpu, f"--query-gpu={fields}",
                     "--format=csv,noheader,nounits").split(", ")
    return {"time": time.time(), **dict(zip(fields.split(","), values))}


def telemetry(gpu, stop, samples):
    while not stop.is_set():
        try:
            samples.append({**gpu_snapshot(gpu), "loadavg": os.getloadavg()})
        except Exception as exc:
            samples.append({"time": time.time(), "error": type(exc).__name__})
        stop.wait(1)


def run_session(a, image, pair, position, arm):
    name = f"infe-treecore-{uuid.uuid4().hex[:12]}"
    stem = a.output / f"pair-{pair:02d}-{position}-{arm}"
    base = f"http://127.0.0.1:{a.port}"
    info = {"pair": pair, "position": position, "arm": arm, "levels": [],
            "started": time.time(), "gpu_before": gpu_snapshot(a.gpu)}
    # Never evict another user's container or occupy an already-used GPU.
    processes = command("nvidia-smi", "-i", a.gpu,
                        "--query-compute-apps=pid", "--format=csv,noheader")
    if processes:
        raise RuntimeError(f"GPU {a.gpu} already has compute processes: {processes}")
    started = False
    try:
        command("docker", "run", "-d", "--name", name, "--gpus", f"device={a.gpu}",
                "--ipc=host", "-p", f"127.0.0.1:{a.port}:8000", "-v", f"{a.hf_cache}:/hf",
                "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1",
                "-e", f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={arm}",
                image, "python3", "-m", "sglang.launch_server", "--model-path", a.model,
                "--revision", a.revision, "--host", "0.0.0.0", "--port", "8000",
                "--context-length", "4096", "--mem-fraction-static", "0.85",
                "--random-seed", "42", "--tool-call-parser", "qwen")
        started = True
        deadline = time.monotonic() + a.startup_timeout
        while True:
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                    json.load(response)
                break
            except Exception:
                if time.monotonic() > deadline or command(
                        "docker", "inspect", "-f", "{{.State.Running}}", name) != "true":
                    raise RuntimeError(f"Server startup failed: {name}")
                time.sleep(2)
        # Host-side cgroup counter avoids docker exec overhead and includes all workers.
        pid = command("docker", "inspect", "-f", "{{.State.Pid}}", name)
        cgroup = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                      if line.startswith("0::"))
        cpu_path = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cpu.stat"
        cpu_usage(cpu_path)  # fail rather than silently measure just the entrypoint
        info["cpu_source"] = "container cgroup-v2 usage_usec"
        for conc in a.concurrency:
            # Prime this exact load level; retain excluded warmup data for audit.
            warmup = []
            for rnd in range(a.warmup_rounds):
                rows, wall = run_level(base, a.model, conc, 160, 300, rnd * conc)
                validate(rows)
                warmup.append({"requests": rows, "wall_ms": wall})
            samples, rounds = [], []
            stop = threading.Event()
            thread = threading.Thread(target=telemetry, args=(a.gpu, stop, samples))
            thread.start()
            try:
                for rnd in range(a.rounds):
                    before = cpu_usage(cpu_path)
                    begin = time.monotonic()
                    utc_begin = time.time()
                    rows, wall = run_level(base, a.model, conc, 160, 300, rnd * conc)
                    end = time.monotonic()
                    after = cpu_usage(cpu_path)
                    summary = summarize_round(rows, wall)
                    rounds.append({"round": rnd, "start": utc_begin, "end": time.time(),
                                   "wall_ms": wall, "cpu_usec": after - before,
                                   "cpu_wall_s": end - begin, "summary": summary, "requests": rows})
            finally:
                stop.set()
                thread.join()
            summary = {key: statistics.median(r["summary"][key] for r in rounds)
                       for key in rounds[0]["summary"]}
            summary["cpu_percent"] = (sum(r["cpu_usec"] for r in rounds) / 1e6
                                       / sum(r["cpu_wall_s"] for r in rounds) * 100)
            info["levels"].append({"concurrency": conc, "warmup": warmup, "rounds": rounds,
                                   "telemetry": samples, "summary": summary})
            save(stem.with_suffix(".json"), info)
            print(f"pair={pair} {arm} conc={conc}: {summary}", flush=True)
        info["completed"] = time.time()
        return info
    except BaseException as exc:
        info["error"] = type(exc).__name__
        raise
    finally:
        save(stem.with_suffix(".json"), info)
        if started:
            try:
                logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
                stem.with_suffix(".log").write_text(
                    redact_log(logs.stdout + logs.stderr, socket.gethostname()))
            finally:
                command("docker", "rm", "-f", name)


def comparisons(sessions):
    """Paired session effects; no request/sample-level significance tests."""
    grouped = {}
    for session in sessions:
        if "completed" not in session:
            raise ValueError("Incomplete session")
        for level in session["levels"]:
            grouped.setdefault(level["concurrency"], {}).setdefault(session["pair"], {})[
                session["arm"]] = level["summary"]
    result = {}
    for conc, pairs in grouped.items():
        result[conc] = {}
        for metric in next(iter(pairs.values()))["python"]:
            effects = [100 * (p["rust"][metric] / p["python"][metric] - 1)
                       for p in pairs.values()]
            result[conc][metric] = {
                "python_session_median": statistics.median(p["python"][metric] for p in pairs.values()),
                "rust_session_median": statistics.median(p["rust"][metric] for p in pairs.values()),
                "paired_percent_changes": effects, "median_percent_change": statistics.median(effects),
                "min_percent_change": min(effects), "max_percent_change": max(effects),
                "pairs": len(effects),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--hf-cache", type=Path, required=True)
    ap.add_argument("--revision", required=True, help="Pinned cached model commit")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image", default="lmsysorg/sglang:v0.5.19")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 64, 256])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup-rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--startup-timeout", type=int, default=600)
    a = ap.parse_args()
    orders = schedule(a.pairs, a.seed)
    if min(a.concurrency) < 1 or a.rounds < 1 or a.warmup_rounds < 1:
        ap.error("concurrency, rounds and warmup-rounds must be positive")
    a.hf_cache = a.hf_cache.resolve(strict=True)
    a.output.mkdir(parents=True, exist_ok=False)
    # Advisory exclusion for other copies of this harness. Also check compute PIDs each session.
    gpu = gpu_snapshot(a.gpu)
    # UUID is used only for the local lock; never included in artifacts.
    gpu_uuid = command("nvidia-smi", "-i", a.gpu, "--query-gpu=uuid", "--format=csv,noheader")
    lock = open(f"/tmp/infe-treecore-{gpu_uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    image_info = json.loads(command("docker", "image", "inspect", a.image))[0]
    image = image_info["Id"]
    manifest = {"config": public_config(vars(a)),
                "orders": orders, "image_id": image, "image_digests": image_info["RepoDigests"],
                "gpu": gpu, "started": time.time(),
                "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), Path(__file__).with_name("e2e_tool_stream.py")]}}
    save(a.output / "manifest.json", manifest)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sessions = []
    for pair, order in enumerate(orders):
        for position, arm in enumerate(order):
            sessions.append(run_session(a, image, pair, position, arm))
    save(a.output / "comparison.json", comparisons(sessions))
    print("Completed benchmark; artifacts saved to the requested output directory.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f"Benchmark failed ({type(exc).__name__}); inspect saved artifacts.") from None
