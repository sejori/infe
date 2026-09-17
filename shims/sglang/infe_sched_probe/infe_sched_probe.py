"""Diagnostic hooks only: no scheduler decisions or GPU synchronisation added.

Install the entry point, then explicitly start/stop each collection window via
the control server. CPU wall time is NOT GPU time or proof of critical-path cost.
"""
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PHASES = {
    "schedule": "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    "forward_launch": "sglang.srt.managers.scheduler.Scheduler.run_batch",
    "result": "sglang.srt.managers.scheduler.Scheduler.process_batch_result",
    "ingest": "sglang.srt.managers.scheduler.Scheduler.process_input_requests",
    "sample_launch": "sglang.srt.managers.scheduler.Scheduler.launch_batch_sample_if_needed",
    # Nested ranges: report separately, never add to their parent's cost.
    "prefill_admission": "sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill",
    "update_running": "sglang.srt.managers.scheduler.Scheduler.update_running_batch",
    "prepare_decode": "sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode",
    "prepare_extend": "sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_extend",
    "cache_match": "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.match_prefix",
    "cache_insert": "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.insert",
    "cache_evict": "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache._evict",
}
MAX_SAMPLES = 200000


class Recorder:
    def __init__(self):
        self.active = False
        self.epoch = 0
        self.stats = {}
        self.nvtx = None
        self.window_range = None
        self.started = None
        self.lock = threading.Lock()

    def start(self, label, nvtx=None):
        if self.active:
            raise ValueError("A measurement window is already active")
        if not re.fullmatch(r"(?:short|mixed|cache_churn)(?:_c[1-9][0-9]*)?", label):
            raise ValueError("Unknown workload")
        self.epoch += 1
        self.stats = {name: {"calls": 0, "wall_ns": 0, "thread_cpu_ns": 0,
                             "samples_ns": [], "active_batches": 0, "evicted_tokens": 0}
                      for name in PHASES}
        self.nvtx = nvtx
        if nvtx:
            self.window_range = nvtx.range_start("infe.window." + label)
        self.started = time.perf_counter_ns()
        self.active = True

    def stop(self):
        if not self.active:
            raise ValueError("No active measurement window")
        with self.lock:
            self.active = False
            elapsed = time.perf_counter_ns() - self.started
            self.epoch += 1  # discard calls straddling this boundary
        if self.nvtx:
            self.nvtx.range_end(self.window_range)
        return {"window_ns": elapsed, "phases": self.stats,
                "note": "Inclusive CPU wall/thread time; nested phases must not be summed."}

    def wrap(self, label):
        def around(original, *args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            epoch, nvtx = self.epoch, self.nvtx
            if nvtx:
                nvtx.range_push("infe.phase." + label)
            start_wall, start_cpu = time.perf_counter_ns(), time.thread_time_ns()
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            finally:
                wall, cpu = time.perf_counter_ns() - start_wall, time.thread_time_ns() - start_cpu
                if nvtx:
                    nvtx.range_pop()
                with self.lock:
                    if self.active and self.epoch == epoch:
                        stats = self.stats[label]
                        stats["calls"] += 1
                        stats["wall_ns"] += wall
                        stats["thread_cpu_ns"] += cpu
                        if len(stats["samples_ns"]) < MAX_SAMPLES:
                            stats["samples_ns"].append(wall)
                        if label == "schedule" and getattr(result, "batch_to_run", None) is not None:
                            stats["active_batches"] += 1
                        if label == "cache_evict":
                            stats["evicted_tokens"] += getattr(result, "num_tokens_evicted", 0)
        return around


RECORDER = Recorder()
SERVER = None


class Control(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        from sglang.srt.plugins.hook_registry import HookRegistry
        missing = [name for name, target in PHASES.items() if target not in HookRegistry._patched]
        self.reply(200, {"ready": not missing, "active": RECORDER.active,
                         "phases": list(PHASES), "missing_phases": missing})

    def reply(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/start":
                nvtx = None
                if body.get("nvtx"):
                    import torch.cuda.nvtx
                    nvtx = torch.cuda.nvtx
                RECORDER.start(body["workload"], nvtx)
                self.reply(200, {"started": True})
            elif self.path == "/stop":
                self.reply(200, RECORDER.stop())
            else:
                self.reply(404, {"error": "Unknown endpoint"})
        except Exception as exc:
            self.reply(400, {"error": type(exc).__name__})


def start_control(result, *args, **kwargs):
    global SERVER
    if SERVER is None:
        SERVER = ThreadingHTTPServer(("0.0.0.0", int(os.environ["INFE_PROBE_PORT"])), Control)
        threading.Thread(target=SERVER.serve_forever, daemon=True).start()
        print("infe scheduler probe control ready", flush=True)
    return result


def register():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name, target in PHASES.items():
        HookRegistry.register(target, RECORDER.wrap(name), HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.__init__", start_control, HookType.AFTER)
