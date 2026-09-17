#!/usr/bin/env python3
"""Validate tuned diagnostic sessions and bound work outside engine cache release."""
import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from batch_result_investigation import EXTRA_PHASES, FLAGS
from scheduler_investigation import summary
from scheduler_trace import PHASES, analyse, duration, intersection, merge
from summarize_scheduler import read

CASES = {"short_c8": (64, 512), "short_c256": (256, 4096)}


def residual(window, excluded):
    """Union/subtract intervals, never sum nested inclusive phase percentages."""
    phases = window["phases_ns"]
    result = merge(phases["result"])
    occupied = merge(window["gpu_busy_ns"] +
                     sum(window["cuda_api_ns"].values(), []) +
                     sum((phases[name] for name in excluded), []))
    return 100 * (duration(result) - intersection(result, occupied)) / window["duration_ns"]


def summarise(directory):
    manifest = read(directory / "manifest.json")
    orders = manifest["orders"]
    if (sorted(orders) != [["probe", "stock"], ["stock", "probe"]] or
            manifest["trace_sessions"] != 2 or manifest["pilot_count"] or
            manifest["selected_workloads"] != ["short"] or
            manifest["extra_server_args"] != FLAGS or manifest["extra_phases"] != EXTRA_PHASES):
        raise ValueError("Unexpected diagnostic protocol")
    sessions, traces = {}, []
    for path in sorted(directory.glob("pair-*/session.json*")):
        s = read(path)
        key = (s["pair"], s["arm"])
        if key in sessions or not s.get("complete") or "error" in s:
            raise ValueError("Incomplete/duplicate session")
        if s["arm"] != "trace" and orders[s["pair"]][s["position"]] != s["arm"]:
            raise ValueError("Wrong session order")
        if [w["case"] for w in s["workloads"]] != list(CASES):
            raise ValueError("Wrong workloads")
        log = gzip.decompress((path.parent / "server.log.gz").read_bytes()).decode()
        captures = re.findall(r"Capture target decode CUDA graph begin\.[^\n]*bs=(\[[^\]]*\])", log)
        if len(captures) != 1 or max(ast.literal_eval(captures[0])) != 256:
            raise ValueError("Decode graph 256 not captured")
        for w in s["workloads"]:
            warm, count = CASES[w["case"]]
            for rows, offset, n in [(w["warmup"]["requests"], 0, warm), (w["requests"], warm, count)]:
                if [r["idx"] for r in rows] != list(range(offset, offset + n)):
                    raise ValueError("Request set mismatch")
                if any((r["prompt_tokens"], r["completion_tokens"]) != (32, 16) or
                       not 0 <= r["cached_tokens"] <= 32 for r in rows):
                    raise ValueError("Token count mismatch")
            values = summary(w["requests"], w["summary"]["wall_s"])
            values["cpu_percent"] = 100 * w["cpu_usec"] / 1e6 / w["cpu_wall_s"]
            if any(not math.isclose(v, w["summary"][k], rel_tol=1e-10) for k, v in values.items()):
                raise ValueError("Stored metrics differ from requests/CPU accounting")
            if s["arm"] == "stock":
                if w["phases"] is not None:
                    raise ValueError("Instrumented stock")
            elif not all(w["phases"]["phases"][name]["calls"] for name in EXTRA_PHASES):
                raise ValueError("Missing detailed hooks")
        sessions[key] = s
        if s["arm"] == "trace":
            evidence = read(path.parent / "derived/intervals.json.gz")
            overlap = analyse(evidence)
            if overlap != read(path.parent / "derived/overlap.json") or list(overlap) != list(CASES):
                raise ValueError("Trace summary/window mismatch")
            for window in evidence["windows"]:
                if set(window["phases_ns"]) != PHASES | set(EXTRA_PHASES) or not window["gpu_busy_ns"]:
                    raise ValueError("Trace missing expected phase/GPU evidence")
                overlap[window["case"]]["result_outside_cache_release_percent"] = residual(window, ["cache_release"])
                overlap[window["case"]]["result_outside_finish_actions_and_cache_percent"] = residual(window, ["finish_actions", "cache_release"])
            traces.append(overlap)
    if set(sessions) != {(p, arm) for p in range(2) for arm in ("stock", "probe", "trace")}:
        raise ValueError("Expected two pairs and two traces")
    cases = {}
    for i, case in enumerate(CASES):
        metrics = ["requests_per_s", "e2e_p50_ms", "e2e_p99_ms", "ttft_p50_ms", "ttft_p99_ms", "cpu_percent"]
        values = {arm: [dict(sessions[p, arm]["workloads"][i]["summary"],
                            cpu_usec_per_output_token=sessions[p, arm]["workloads"][i]["cpu_usec"] /
                            sessions[p, arm]["workloads"][i]["summary"]["completion_tokens"])
                       for p in range(2)] for arm in ["stock", "probe", "trace"]}
        metrics.append("cpu_usec_per_output_token")
        cases[case] = {"sessions": values,
            "paired_probe_effect_percent": {m: [100 * (values["probe"][p][m] / values["stock"][p][m] - 1)
                                                       for p in range(2)] for m in metrics},
            "trace_vs_median_probe_percent": {m: [100 * (v[m] / median(x[m] for x in values["probe"]) - 1)
                                                         for v in values["trace"]] for m in metrics},
            "traces": [t[case] for t in traces]}
    return {"independent_pairs": 2, "trace_sessions": 2, "diagnostic_only": True,
            "note": "Probe effects measure instrumentation. Residuals are opportunity screens, not causal speedup predictions.",
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    (a.directory / "summary.json").write_text(json.dumps(summarise(a.directory), indent=2) + "\n")
    print("Validated six sessions, fixed work, graph capture, CPU accounting and trace intervals.")
