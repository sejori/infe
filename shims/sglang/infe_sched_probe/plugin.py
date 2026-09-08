"""Plugin entry-point for SGLang's plugin loader.

When pip-installed, setuptools exposes this module via the
``sglang.srt.plugins`` entry-point group. SGLang's ``load_plugins()``
imports it, which triggers hook registration in the parent module.

For PYTHONPATH usage (no install), importing ``infe_sched_probe`` directly
runs this code:

    PYTHONPATH=/shims/sglang/infe_sched_probe \
    SGLANG_PLUGINS=infe_sched_probe \
    python3 -m sglang.launch_server ...
"""
import logging
import os
import sys
import threading
import time
import atexit
import json

logger = logging.getLogger("infe.sched_probe")

TIMER_OUT = os.environ.get("SGLANG_SCHED_TIMER_OUT", "")
_enable = bool(TIMER_OUT)

_lock = threading.Lock()
_stats = {
    "get_next_batch_to_run": {"calls": 0, "total_ns": 0, "samples": []},
    "run_batch": {"calls": 0, "total_ns": 0, "samples": []},
    "process_batch_result": {"calls": 0, "total_ns": 0, "samples": []},
    "step_span": {"calls": 0, "total_ns": 0, "samples": []},
}
_step_start = None
_MAX_SAMPLES = 200000


def _record(phase, duration_ns):
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


def _percentile(samples, p):
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


# Register hooks. SGLang's load_plugins() imports this module, which runs
# this top-level code. For PYTHONPATH usage, importing infe_sched_probe
# also triggers registration.
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
    logger.warning("sglang not available; hook registration deferred")
