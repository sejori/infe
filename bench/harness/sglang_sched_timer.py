"""SGLang plugin: per-step phase timing for the infe-sched M0 probe.

Hooks three methods on the SGLang Scheduler with AROUND wrappers that record
wall-clock time for each phase of the scheduling loop:

  1. get_next_batch_to_run — admission, batching, preemption decisions
  2. run_batch            — GPU forward launch (CPU-side) + overlap prep
  3. process_batch_result — result processing (decode/extend bookkeeping)

The plugin also tracks the per-step wall span (from the first phase to the
last) and reports the fraction of step time consumed by each phase, so we can
answer the kill-criterion question: is the scheduler on the critical path?

Usage (inside the SGLang container, two options):

  Option A — PYTHONPATH (no install needed; mirrors the infe-parsers shim):
    PYTHONPATH=/shims/sglang/infe_sched_probe \
    SGLANG_SCHED_TIMER_OUT=/tmp/timer.json \
    SGLANG_PLUGINS=infe_sched_probe \
    python3 -m sglang.launch_server ...

  Option B — pip install -e (if you prefer setuptools entry points):
    pip install -e /shims/sglang/infe_sched_probe
    SGLANG_SCHED_TIMER_OUT=/tmp/timer.json \
    SGLANG_PLUGINS=infe_sched_probe \
    python3 -m sglang.launch_server ...

The timer is zero-overhead when SGLANG_SCHED_TIMER_OUT is unset: the AROUND
wrapper still calls the original but takes no timestamps and writes no file.

Kill criterion (from docs/infe-kv-brief.md §2 applied to infe-sched):
  if get_next_batch_to_run is < 5% of per-step wall time AND its time fits
  within the GPU forward shadow under overlap scheduling, the scheduler is
  not on the critical path. This plugin provides the measurement; py-spy plus
  the high-admission workload provide the confirming profile.
"""
import atexit
import json
import logging
import os
import sys
import threading
import time

logger = logging.getLogger("infe.sched_probe")

# Output file path. When unset, the plugin is a no-op (hooks still apply
# but collect no data, costing ~1µs per call).
TIMER_OUT = os.environ.get("SGLANG_SCHED_TIMER_OUT", "")

# Per-phase accumulator. Thread-safe because the scheduler event loop
# runs in a single thread, but overlap mode can interleave
# run_batch of batch N+1 with process_batch_result of batch N.
_lock = threading.Lock()
_stats = {
    "get_next_batch_to_run": {"calls": 0, "total_ns": 0, "samples": []},
    "run_batch": {"calls": 0, "total_ns": 0, "samples": []},
    "process_batch_result": {"calls": 0, "total_ns": 0, "samples": []},
    "step_span": {"calls": 0, "total_ns": 0, "samples": []},
}
_step_start = None  # track per-step wall span (first phase → last phase)
_enable = bool(TIMER_OUT)
# Cap stored samples to avoid unbounded memory for very long runs.
_MAX_SAMPLES = 200000


def _record(phase, duration_ns):
    """Thread-safe accumulation of per-phase timing."""
    if not _enable:
        return
    with _lock:
        d = _stats[phase]
        d["calls"] += 1
        d["total_ns"] += duration_ns
        if len(d["samples"]) < _MAX_SAMPLES:
            d["samples"].append(duration_ns)


def _ns():
    return time.perf_counter_ns()


# ---------------------------------------------------------------------------
# Hook functions (AROUND: fn(original, *args, **kwargs) -> result)
# ---------------------------------------------------------------------------

def _hook_get_next_batch_to_run(original_fn, *args, **kwargs):
    global _step_start
    t0 = _ns()
    _step_start = t0
    result = original_fn(*args, **kwargs)
    _record("get_next_batch_to_run", _ns() - t0)
    return result


def _hook_run_batch(original_fn, *args, **kwargs):
    t0 = _ns()
    result = original_fn(*args, **kwargs)
    _record("run_batch", _ns() - t0)
    return result


def _hook_process_batch_result(original_fn, *args, **kwargs):
    global _step_start
    t0 = _ns()
    result = original_fn(*args, **kwargs)
    _record("process_batch_result", _ns() - t0)
    if _step_start is not None:
        _record("step_span", _ns() - _step_start)
        _step_start = None
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percentile(samples, p):
    """Simple percentile over sorted samples."""
    if not samples:
        return 0
    sorted_s = sorted(samples)
    k = int((len(sorted_s) - 1) * p)
    return sorted_s[k]


def _dump():
    if not _enable:
        return
    with _lock:
        report = {}
        for phase, d in _stats.items():
            calls = d["calls"]
            samples = d["samples"]
            entry = {
                "calls": calls,
                "total_ns": d["total_ns"],
                "mean_ns": d["total_ns"] // calls if calls > 0 else 0,
                "sample_count": len(samples),
            }
            if samples:
                entry["p50_ns"] = _percentile(samples, 0.50)
                entry["p90_ns"] = _percentile(samples, 0.90)
                entry["p99_ns"] = _percentile(samples, 0.99)
                entry["max_ns"] = max(samples)
            report[phase] = entry

        # Compute fractions of per-step wall time
        sched_total = _stats["get_next_batch_to_run"]["total_ns"]
        forward_total = _stats["run_batch"]["total_ns"]
        result_total = _stats["process_batch_result"]["total_ns"]
        step_total = _stats["step_span"]["total_ns"]
        denom = step_total if step_total > 0 else (
            sched_total + forward_total + result_total
        )
        report["fractions"] = {
            "sched_of_step_pct": 100.0 * sched_total / denom if denom > 0 else 0,
            "forward_of_step_pct": 100.0 * forward_total / denom if denom > 0 else 0,
            "result_of_step_pct": 100.0 * result_total / denom if denom > 0 else 0,
        }
        report["overlap_mode_note"] = (
            "Under overlap scheduling, run_batch of batch N+1 interleaves with "
            "process_batch_result of batch N. step_span measures first-phase "
            "to last-phase for one batch and is the best per-step proxy."
        )

    with open(TIMER_OUT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("infe-sched timer: wrote %s", TIMER_OUT)

    f = report["fractions"]
    s = report["get_next_batch_to_run"]
    r = report["run_batch"]
    p = report["process_batch_result"]
    print(f"[infe-sched-probe] steps={report['step_span']['calls']} "
          f"sched={f['sched_of_step_pct']:.1f}% (mean {s['mean_ns'] / 1000:.0f}µs) "
          f"forward={f['forward_of_step_pct']:.1f}% (mean {r['mean_ns'] / 1000:.0f}µs) "
          f"result={f['result_of_step_pct']:.1f}% (mean {p['mean_ns'] / 1000:.0f}µs)",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Register hooks via SGLang's plugin hook registry.
# This runs at import time — load_plugins() calls the entry-point, which is
# this module's top-level code. If not loaded via entry-point, importing this
# module directly also registers the hooks (for the PYTHONPATH approach).
# ---------------------------------------------------------------------------

try:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _HOOK_TARGETS = {
        "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run": (
            _hook_get_next_batch_to_run, HookType.AROUND,
        ),
        "sglang.srt.managers.scheduler.Scheduler.run_batch": (
            _hook_run_batch, HookType.AROUND,
        ),
        "sglang.srt.managers.scheduler.Scheduler.process_batch_result": (
            _hook_process_batch_result, HookType.AROUND,
        ),
    }

    for target, (hook, htype) in _HOOK_TARGETS.items():
        HookRegistry.register(target, hook, htype, source=None)
        logger.info("infe-sched timer: registered %s hook on %s", htype.value, target)

    atexit.register(_dump)
except ImportError:
    # SGLang not available (e.g. testing outside the engine).
    # Hooks will register when this module is imported inside the container.
    logger.warning("sglang not available; hook registration deferred")
