#!/usr/bin/env python3
"""Independent-session scheduler instrumentation A/B and separate Nsight traces.

Synthetic token-ID workloads isolate scheduling/metadata from tokenization and
text semantics. max_new_tokens and ignore_eos fix actual generation work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
from http.client import RemoteDisconnected
import hashlib
import itertools
import json
from pathlib import Path
import random
import socket
import subprocess
import threading
import time
import urllib.request
import uuid

from treecore_sessions import command, cpu_usage, gpu_snapshot, redact_log, save, telemetry


WORKLOADS = {
    "short": {"warmup": 256, "requests": 4096},
    "mixed": {"warmup": 64, "requests": 512},
    "cache_churn": {"warmup": 128, "requests": 128},
}
CASES = [("short", 8), ("short", 256), ("mixed", 64), ("cache_churn", 64)]
SEED = 20260917


def plan(workload, idx):
    rng = random.Random(SEED + idx)
    if workload == "short":
        length, output = 32, 16
        tokens = [rng.randrange(1000, 3000) for _ in range(length)]
    elif workload == "mixed":
        length = [128, 1024, 4096][idx % 3]
        output = [16, 64, 128][(idx // 3) % 3]
        tokens = [rng.randrange(1000, 3000) for _ in range(length)]
    elif workload == "cache_churn":
        # Warmup visits every family (1M prefix tokens). Measurement mixes
        # 75% hot-prefix reuse with 25% cold families, exercising both matching
        # and eviction instead of a pure prefill-bound cache-miss workload.
        family = idx % 128 if idx < 128 else (
            idx % 16 if idx % 4 != 3 else 16 + (idx // 4) % 112)
        prefix_rng = random.Random(SEED + 100000 + family)
        tokens = [prefix_rng.randrange(1000, 3000) for _ in range(8192)]
        tokens += [rng.randrange(3000, 4000) for _ in range(32)]
        output = 16
    else:
        raise ValueError(workload)
    return tokens, output


def http(base, path, body=None, timeout=600):
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data.decode()


def request_one(base, workload, idx):
    tokens, output = plan(workload, idx)
    body = {"input_ids": tokens, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": output, "ignore_eos": True}}
    request = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start, first, last = time.perf_counter(), None, None
    meta, chunks = {}, 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError("Generation returned an error")
            now = time.perf_counter()
            meta = event.get("meta_info", meta)
            if meta.get("completion_tokens", 0) > 0:
                if first is None:
                    first = now
                last = now
                chunks += 1
    end = time.perf_counter()
    if first is None or meta.get("completion_tokens") != output or meta.get("prompt_tokens") != len(tokens):
        raise RuntimeError("Generation token count mismatch")
    return {"idx": idx, "prompt_tokens": len(tokens), "completion_tokens": output,
            "cached_tokens": meta.get("cached_tokens"), "chunks": chunks,
            "ttft_ms": (first - start) * 1000, "e2e_ms": (end - start) * 1000,
            "stream_span_ms": (last - first) * 1000}


def load(base, workload, count, concurrency, offset):
    # Fixed-size closed-loop pool keeps admissions arriving as requests finish.
    # Identical request set in every arm/session; assignment/order may vary.
    counter = itertools.count()
    barrier = threading.Barrier(concurrency)
    def worker():
        rows = []
        barrier.wait()
        while True:
            idx = next(counter)
            if idx >= count:
                return rows
            rows.append(request_one(base, workload, offset + idx))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        rows = [row for future in futures for row in future.result()]
    wall = time.perf_counter() - start
    return sorted(rows, key=lambda row: row["idx"]), wall


def summary(rows, wall):
    from e2e_tool_stream import pct
    return {"requests": len(rows), "wall_s": wall, "requests_per_s": len(rows) / wall,
            "output_tokens_per_s": sum(row["completion_tokens"] for row in rows) / wall,
            "e2e_p50_ms": pct([row["e2e_ms"] for row in rows], .5),
            "e2e_p99_ms": pct([row["e2e_ms"] for row in rows], .99),
            "ttft_p50_ms": pct([row["ttft_ms"] for row in rows], .5),
            "ttft_p99_ms": pct([row["ttft_ms"] for row in rows], .99),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "cached_tokens": sum(row["cached_tokens"] or 0 for row in rows)}


def wait_ready(base, name):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            http(base, "/v1/models", timeout=2)
            return
        except Exception:
            if command("docker", "inspect", "-f", "{{.State.Running}}", name) != "true":
                break
            time.sleep(2)
    raise RuntimeError("Server failed readiness")


def run_session(a, image, pair, position, arm):
    directory = a.output / f"pair-{pair:02d}-{position}-{arm}"
    directory.mkdir()
    name = "infe-treecore-" + uuid.uuid4().hex[:12]
    base, control = f"http://127.0.0.1:{a.port}", f"http://127.0.0.1:{a.control_port}"
    info = {"pair": pair, "position": position, "arm": arm, "workloads": [],
            "gpu_before": gpu_snapshot(a.gpu)}
    started = False
    try:
        if command("nvidia-smi", "-i", a.gpu, "--query-compute-apps=pid", "--format=csv,noheader"):
            raise RuntimeError("Selected GPU has another compute process")
        args = ["docker", "run", "-d", "--name", name, "--gpus", f"device={a.gpu}",
                "--ipc=host", "-p", f"127.0.0.1:{a.port}:8000",
                "-p", f"127.0.0.1:{a.control_port}:19097", "-v", f"{a.hf_cache}:/hf",
                "-v", f"{a.plugin}:/probe:ro", "-v", f"{directory.resolve()}:/artifacts",
                "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1",
                "-e", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python",
                "-e", "INFE_PROBE_PORT=19097", "-e", "SGLANG_PLUGINS=infe_sched_probe"]
        server = ["python3", "-m", "sglang.launch_server", "--model-path", a.model,
                  "--revision", a.revision, "--context-length", "16384", "--mem-fraction-static", "0.85",
                  "--host", "0.0.0.0", "--port", "8000", "--random-seed", "42"]
        launch = server
        if arm == "trace":
            launch = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                      "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                      "--capture-range-end=stop-shutdown", "--kill=none",
                      "--force-overwrite=true", "--output=/artifacts/trace"] + server
        # Paths/arguments below are fixed strings except trusted CLI model/revision.
        # shlex.join quotes these properly; no interpolation of command text.
        import shlex
        script = ("cp -r /probe /tmp/infe-probe-src && pip install --no-deps --no-build-isolation -q /tmp/infe-probe-src && "
                  if arm != "stock" else "")
        script += "exec " + shlex.join(launch)
        command(*args, "--entrypoint", "bash", image, "-c", script)
        started = True
        wait_ready(base, name)
        if arm != "stock":
            ready = http(control, "/")
            if not ready.get("ready"):
                raise RuntimeError("Probe control unavailable")
        pid = command("docker", "inspect", "-f", "{{.State.Pid}}", name)
        cg = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
        cpu_path = Path("/sys/fs/cgroup") / cg.lstrip("/") / "cpu.stat"
        cpu_usage(cpu_path)
        server_info = http(base, "/get_server_info")
        # Save only capacity fields, never the whole server/environment metadata.
        info["capacity"] = extract_capacity(server_info)
        if arm == "trace":
            http(base, "/start_profile", {"activities": ["CUDA_PROFILER"], "with_stack": False,
                                         "record_shapes": False, "output_dir": "/artifacts"})
        cases = list(dict.fromkeys((mode, a.concurrency or concurrency)
                                   for mode, concurrency in CASES if mode in a.workloads))
        for workload, concurrency in cases:
            http(base, "/flush_cache?timeout=10", {})
            label = f"{workload}_c{concurrency}"
            settings = dict(WORKLOADS[workload])
            if workload == "short" and concurrency <= 8:
                settings = {"warmup": 64, "requests": 512}
            warm_count = min(settings["warmup"], a.pilot_count) if a.pilot_count else settings["warmup"]
            count = min(settings["requests"], a.pilot_count) if a.pilot_count else settings["requests"]
            warm, warm_wall = load(base, workload, warm_count, concurrency, 0)
            samples, stop = [], threading.Event()
            thread = threading.Thread(target=telemetry, args=(a.gpu, stop, samples))
            thread.start()
            try:
                if arm != "stock":
                    http(control, "/start", {"workload": label, "nvtx": True})
                before, start = cpu_usage(cpu_path), time.perf_counter()
                rows, wall = load(base, workload, count, concurrency, settings["warmup"])
                elapsed, after = time.perf_counter() - start, cpu_usage(cpu_path)
                phases = http(control, "/stop", {}) if arm != "stock" else None
                if phases is not None and not phases.get("phases", {}).get("schedule", {}).get("calls"):
                    raise RuntimeError("Missing scheduler timing data")
            finally:
                stop.set()
                thread.join()
            result = summary(rows, wall)
            result["cpu_percent"] = (after - before) / 1e6 / elapsed * 100
            entry = {"workload": workload, "case": label, "concurrency": concurrency,
                     "summary": result, "requests": rows,
                     "warmup": {"requests": warm, "wall_s": warm_wall},
                     "telemetry": samples, "phases": phases,
                     "cpu_usec": after - before, "cpu_wall_s": elapsed}
            info["workloads"].append(entry)
            save(directory / "session.json", info)
            print(f"pair={pair} arm={arm} case={label}: {result}", flush=True)
        if arm == "trace":
            try:
                http(base, "/stop_profile", {}, timeout=600)
            except (RemoteDisconnected, ConnectionResetError):
                # nsys stop-shutdown can exit PID 1 before HTTP responds.
                # Success still requires an actual report below.
                info["profile_stop_connection_closed"] = True
            deadline = time.monotonic() + 180
            while not list(directory.glob("*.nsys-rep")) and time.monotonic() < deadline:
                time.sleep(2)
            if not list(directory.glob("*.nsys-rep")):
                raise RuntimeError("No Nsight report produced")
        info["complete"] = True
        return info
    except BaseException as exc:
        info["error"] = type(exc).__name__
        raise
    finally:
        save(directory / "session.json", info)
        if started:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
            text = redact_log(logs.stdout + logs.stderr, socket.gethostname())
            (directory / "server.log.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_capacity(value):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"max_total_num_tokens", "max_total_tokens", "max_running_requests", "max_req_len"}:
                result[key] = child
            elif isinstance(child, (dict, list)):
                result.update(extract_capacity(child))
    elif isinstance(value, list):
        for child in value:
            result.update(extract_capacity(child))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--hf-cache", type=Path, required=True)
    ap.add_argument("--plugin", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--image", default="lmsysorg/sglang:v0.5.19")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--trace-sessions", type=int, default=0)
    ap.add_argument("--concurrency", type=int, help="Override case concurrency (pilot only)")
    ap.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    ap.add_argument("--pilot-count", type=int, default=0)
    ap.add_argument("--port", type=int, default=18096)
    ap.add_argument("--control-port", type=int, default=19097)
    a = ap.parse_args()
    if a.pairs % 2 or a.pairs < 0 or (a.concurrency is not None and a.concurrency < 1) or (not a.pairs and not a.trace_sessions):
        ap.error("Need an even nonnegative pair count or trace sessions, and positive concurrency")
    a.hf_cache, a.plugin = a.hf_cache.resolve(strict=True), a.plugin.resolve(strict=True)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    gpu_uuid = command("nvidia-smi", "-i", a.gpu, "--query-gpu=uuid", "--format=csv,noheader")
    lock = open(f"/tmp/infe-treecore-{gpu_uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for port in [a.port, a.control_port]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    image = json.loads(command("docker", "image", "inspect", a.image))[0]
    orders = [["stock", "probe"], ["probe", "stock"]] * (a.pairs // 2)
    random.Random(SEED).shuffle(orders)
    sources = [Path(__file__), Path(__file__).with_name("treecore_sessions.py"),
               a.plugin / "infe_sched_probe.py", a.plugin / "pyproject.toml"]
    manifest = {"image_id": image["Id"], "image_digests": image["RepoDigests"],
                "model": a.model, "revision": a.revision, "gpu": gpu_snapshot(a.gpu),
                "concurrency_override": a.concurrency, "cases": CASES, "workloads": WORKLOADS,
                "selected_workloads": a.workloads, "pilot_count": a.pilot_count,
                "orders": orders, "trace_sessions": a.trace_sessions, "seed": SEED,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    save(a.output / "manifest.json", manifest)
    for pair, order in enumerate(orders):
        for position, arm in enumerate(order):
            run_session(a, image["Id"], pair, position, arm)
    for repeat in range(a.trace_sessions):
        run_session(a, image["Id"], repeat, 0, "trace")
    print("Investigation sessions complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f"Investigation failed ({type(exc).__name__}); inspect retained session artifacts.") from None
