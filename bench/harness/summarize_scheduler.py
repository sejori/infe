#!/usr/bin/env python3
"""Validate and summarise complete scheduler runs without pooling replicates."""
import argparse
import ast
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median

from scheduler_investigation import CASES, WORKLOADS, summary
from scheduler_trace import analyse


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def spread(values):
    return {"median": median(values), "min": min(values), "max": max(values), "values": values}


def validate(session):
    if not session.get("complete") or "error" in session:
        raise ValueError("Incomplete session")
    expected = [f"{mode}_c{concurrency}" for mode, concurrency in CASES]
    if [w["case"] for w in session["workloads"]] != expected:
        raise ValueError("Missing, duplicate or reordered workload")
    for w in session["workloads"]:
        mode = w["workload"]
        settings = {"warmup": 64, "requests": 512} if w["case"] == "short_c8" else WORKLOADS[mode]
        for stage, rows, offset, count in (
            ("warmup", w["warmup"]["requests"], 0, settings["warmup"]),
            ("measured", w["requests"], settings["warmup"], settings["requests"]),
        ):
            if [r["idx"] for r in rows] != list(range(offset, offset + count)):
                raise ValueError(f"Wrong {stage} request set")
            for row in rows:
                idx = row["idx"]
                prompt = 32 if mode == "short" else 8224 if mode == "cache_churn" else [128, 1024, 4096][idx % 3]
                output = [16, 64, 128][(idx // 3) % 3] if mode == "mixed" else 16
                if (row["prompt_tokens"], row["completion_tokens"]) != (prompt, output):
                    raise ValueError("Token count mismatch")
                if not 0 <= row["cached_tokens"] <= prompt:
                    raise ValueError("Invalid cache accounting")
        recomputed = summary(w["requests"], w["summary"]["wall_s"])
        recomputed["cpu_percent"] = 100 * w["cpu_usec"] / 1e6 / w["cpu_wall_s"]
        if any(not math.isclose(value, w["summary"][key], rel_tol=1e-10) for key, value in recomputed.items()):
            raise ValueError("Stored summary does not match raw data")
        if session["arm"] != "stock":
            phases = w["phases"]["phases"]
            if not phases["schedule"]["active_batches"] or not phases["forward_launch"]["calls"]:
                raise ValueError("No active batch timings")
            if mode == "cache_churn" and not phases["cache_evict"]["evicted_tokens"]:
                raise ValueError("Cache-churn workload did not evict tokens")
    capacity = session["capacity"]["max_total_num_tokens"]
    if not 0 < capacity < 128 * 8192:
        raise ValueError("Cache warmup does not exceed capacity")


def summarise(directory):
    manifest = read(directory / "manifest.json")
    if manifest["pilot_count"] or manifest["concurrency_override"]:
        raise ValueError("Pilot run is not final evidence")
    paths = sorted(directory.glob("pair-*/session.json*"))
    sessions = [read(path) for path in paths]
    for session in sessions:
        validate(session)
    by_key = {(s["pair"], s["arm"]): s for s in sessions}
    if len(by_key) != len(sessions):
        raise ValueError("Duplicate session")
    expected = {(p, arm) for p, order in enumerate(manifest["orders"]) for arm in order}
    expected |= {(p, "trace") for p in range(manifest["trace_sessions"])}
    if set(by_key) != expected:
        raise ValueError("Run does not contain every planned session")
    if len(manifest["orders"]) != 6 or manifest["trace_sessions"] != 2:
        raise ValueError("Final protocol requires six pairs and two traces")
    for pair, order in enumerate(manifest["orders"]):
        for position, arm in enumerate(order):
            if by_key[pair, arm]["position"] != position:
                raise ValueError("Session order differs from manifest")
    overlaps = {}
    for path, session in zip(paths, sessions):
        if session["arm"] == "trace":
            evidence = read(path.parent / "derived/intervals.json.gz")
            overlaps[session["pair"]] = analyse(evidence)
            if set(overlaps[session["pair"]]) != {w["case"] for w in session["workloads"]}:
                raise ValueError("Trace windows do not match workloads")
            for window in evidence["windows"]:
                if not window["gpu_busy_ns"] or not all(window["phases_ns"][name] for name in ("schedule", "forward_launch")):
                    raise ValueError("Trace window lacks GPU or scheduler activity")
    result = {"independent_pairs": 6, "diagnostic_trace_sessions": 2,
              "paired_effect": "100 * (probe / stock - 1); instrumentation, not a Rust comparison",
              "trace_comparison": "100 * (trace / median(unprofiled probe sessions) - 1); not paired",
              "cases": {}}
    for i, (mode, concurrency) in enumerate(CASES):
        case = f"{mode}_c{concurrency}"
        arms = {arm: [s["workloads"][i] for s in sessions if s["arm"] == arm]
                for arm in ("stock", "probe", "trace")}
        metrics = ("wall_s", "requests_per_s", "output_tokens_per_s", "e2e_p50_ms", "e2e_p99_ms",
                   "ttft_p50_ms", "ttft_p99_ms", "cpu_percent", "cached_tokens")
        totals = {arm: {metric: spread([w["summary"][metric] for w in values]) for metric in metrics}
                  for arm, values in arms.items()}
        paired = {metric: spread([100 * (by_key[p, "probe"]["workloads"][i]["summary"][metric] /
                                         by_key[p, "stock"]["workloads"][i]["summary"][metric] - 1)
                                  for p in range(6)]) for metric in metrics}
        phase_totals = {}
        for arm in ("probe", "trace"):
            phase_totals[arm] = {}
            for name in arms[arm][0]["phases"]["phases"]:
                phase_totals[arm][name] = {
                    label: spread([100 * w["phases"]["phases"][name][key] / w["phases"]["window_ns"] for w in arms[arm]])
                    for key, label in (("wall_ns", "wall_percent"), ("thread_cpu_ns", "thread_cpu_percent"))}
                phase_totals[arm][name].update({key: spread([w["phases"]["phases"][name][key] for w in arms[arm]])
                                              for key in ("calls", "active_batches", "evicted_tokens")})
        result["cases"][case] = {
            "sessions": totals, "paired_probe_effect_percent": paired,
            "phase_percent_of_window_and_counts": phase_totals,
            "trace_effect_percent": {m: [100 * (w["summary"][m] / totals["probe"][m]["median"] - 1)
                                          for w in arms["trace"]] for m in metrics},
            "trace_overlap": [overlaps[p][case] for p in range(2)],
        }
    return result


def summarise_graph_check(directory):
    manifest = read(directory / "manifest.json")
    expected_orders = [["default", "decode_graph_256"], ["decode_graph_256", "default"]]
    if manifest["orders"] != expected_orders:
        raise ValueError("Unexpected exploratory graph order")
    sessions = [read(p) for p in sorted(directory.glob("pair-*/session.json*"))]
    if len(sessions) != 4:
        raise ValueError("Graph control needs four sessions")
    by_key = {}
    for s in sessions:
        key = (s["pair"], s["configuration"])
        if key in by_key or not s.get("complete") or "error" in s or len(s["workloads"]) != 1:
            raise ValueError("Invalid graph session")
        if s["configuration"] != expected_orders[s["pair"]][s["position"]]:
            raise ValueError("Wrong graph session order")
        if s["extra_server_args"] != manifest["configurations"][s["configuration"]] or s["arm"] != "stock":
            raise ValueError("Wrong graph intervention")
        w = s["workloads"][0]
        if w["case"] != "short_c256" or w["phases"] is not None:
            raise ValueError("Wrong graph workload or unexpected probe")
        for rows, offset, count in [(w["requests"], 256, 4096), (w["warmup"]["requests"], 0, 256)]:
            if [r["idx"] for r in rows] != list(range(offset, offset + count)):
                raise ValueError("Graph request set mismatch")
            if any((r["prompt_tokens"], r["completion_tokens"]) != (32, 16) for r in rows):
                raise ValueError("Graph token count mismatch")
        recomputed = summary(w["requests"], w["summary"]["wall_s"])
        recomputed["cpu_percent"] = 100 * w["cpu_usec"] / 1e6 / w["cpu_wall_s"]
        if any(not math.isclose(v, w["summary"][k], rel_tol=1e-10) for k, v in recomputed.items()):
            raise ValueError("Graph metrics do not match raw data")
        log = directory / f"pair-{s['pair']:02d}-{s['position']}-stock/server.log.gz"
        captures = re.findall(r"Capture target decode CUDA graph begin\.[^\n]*bs=(\[[^\]]*\])", gzip.decompress(log.read_bytes()).decode())
        target = 256 if s["configuration"] == "decode_graph_256" else 24
        if len(captures) != 1 or max(ast.literal_eval(captures[0])) != target:
            raise ValueError("Requested decode graph coverage was not captured")
        by_key[key] = w["summary"]
    paired = []
    for pair in range(2):
        values = {name: by_key[pair, name] for name in expected_orders[pair]}
        paired.append({"pair": pair, "summaries": values,
                       "graph_effect_percent": {key: 100 * (values["decode_graph_256"][key] / values["default"][key] - 1)
                                                for key in ("wall_s", "requests_per_s", "e2e_p50_ms", "ttft_p50_ms", "cpu_percent")}})
    return {"independent_pairs": 2, "exploratory": True, "paired": paired}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    result = summarise(a.directory)
    (a.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    graph = a.directory / "graph-check"
    if graph.exists():
        (graph / "comparison.json").write_text(json.dumps(summarise_graph_check(graph), indent=2) + "\n")
        print("Validated graph-control requests, metrics, order and captured batch sizes.")
    print("Validated all sessions, token counts, cache pressure and trace windows; wrote summary.json.")


if __name__ == "__main__":
    main()
