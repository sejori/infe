#!/usr/bin/env python3
"""Export privacy-safe interval evidence and CPU/GPU overlap from Nsight SQLite.

Only relative timestamps, whitelisted phase labels and GPU busy intervals leave
the raw trace. No host, device, process, thread, path or container identifiers.
"""
import argparse
import gzip
import json
from pathlib import Path
import re
import sqlite3


PHASES = {"schedule", "forward_launch", "result", "ingest", "sample_launch",
          "prefill_admission", "update_running", "prepare_decode", "prepare_extend",
          "cache_match", "cache_insert", "cache_evict"}


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def clip(intervals, start, end):
    return [[max(a, start), min(b, end)] for a, b in intervals if b > start and a < end]


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    """Intersection duration of two sorted, disjoint interval lists."""
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def analyse(evidence):
    result = {}
    for window in evidence["windows"]:
        width = window["duration_ns"]
        gpu = merge(window["gpu_busy_ns"])
        api = {name: merge(values) for name, values in window.get("cuda_api_ns", {}).items()}
        gpu_or_api = merge(gpu + sum(api.values(), []))
        phases = window["phases_ns"]
        groups = dict(phases)
        groups["non_launch_bookkeeping"] = sum((phases.get(name, []) for name in ("schedule", "result", "ingest")), [])
        metrics = {}
        for name, intervals in groups.items():
            union = merge(intervals)
            wall = duration(union)
            overlap = intersection(union, gpu)
            metrics[name] = {"ranges": len(intervals), "wall_ns": wall,
                             "gpu_overlap_ns": overlap, "gpu_uncovered_ns": wall - overlap,
                             "wall_percent_of_window": 100 * wall / width,
                             "gpu_uncovered_percent_of_window": 100 * (wall - overlap) / width,
                             "gpu_uncovered_outside_cuda_api_percent_of_window":
                                 100 * (wall - intersection(union, gpu_or_api)) / width,
                             "cuda_api_percent_of_window": {
                                 category: 100 * intersection(union, values) / width
                                 for category, values in api.items()}}
        result[window["case"]] = {"window_ns": width, "gpu_busy_ns": duration(gpu),
                                  "gpu_busy_percent": 100 * duration(gpu) / width, "phases": metrics}
    return result


def extract(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables or "NVTX_EVENTS" not in tables:
        raise ValueError("Trace lacks GPU kernels or NVTX ranges")
    gpu = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            gpu.extend(con.execute(f"SELECT start,end FROM {table}").fetchall())
    gpu = merge(gpu)
    if not gpu:
        raise ValueError("No GPU activity")
    ranges = con.execute("SELECT n.start,n.end,COALESCE(n.text,s.value),n.globalTid FROM NVTX_EVENTS n "
                         "LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL").fetchall()
    windows = [(start, end, name.removeprefix("infe.window.")) for start, end, name, _ in ranges
               if name and re.fullmatch(r"infe\.window\.(short|mixed|cache_churn)(?:_c[1-9][0-9]*)?", name)]
    if not windows or len({case for _, _, case in windows}) != len(windows):
        raise ValueError("Missing or duplicate measurement windows")
    phases = {name: [] for name in sorted(PHASES)}
    scheduler_threads = set()
    for start, end, label, tid in ranges:
        if label and label.startswith("infe.phase."):
            name = label.removeprefix("infe.phase.")
            if name not in PHASES:
                raise ValueError("Unexpected phase label")
            phases[name].append([start, end])
            scheduler_threads.add(tid)
    if len(scheduler_threads) != 1:
        raise ValueError("Expected one scheduler thread for this TP=1 protocol")
    # Match API calls to the actual phase thread internally; never export its ID.
    api = {name: [] for name in ("synchronise", "memcpy", "launch", "other")}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for start, end, name in con.execute(
            f"SELECT r.start,r.end,s.value FROM {table} r JOIN StringIds s ON r.nameId=s.id WHERE r.globalTid=?",
            (next(iter(scheduler_threads)),),
        ):
            category = ("synchronise" if "Synchronize" in name else "memcpy" if "Memcpy" in name
                        else "launch" if "Launch" in name else "other")
            api[category].append([start, end])
    if not any(api.values()):
        raise ValueError("No CUDA API calls on the scheduler thread")
    evidence = {"units": "nanoseconds relative to each window start",
                "gpu_activity": "union of kernels, memcpy and memset across captured streams",
                "windows": []}
    for start, end, case in sorted(windows):
        if end <= start:
            raise ValueError("Empty measurement window")
        def relative(intervals):
            return [[a - start, b - start] for a, b in clip(intervals, start, end)]
        evidence["windows"].append({"case": case, "duration_ns": end - start,
                                    "gpu_busy_ns": relative(gpu),
                                    "cuda_api_ns": {name: relative(values) for name, values in api.items()},
                                    "phases_ns": {name: relative(values) for name, values in phases.items()}})
    con.close()
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    evidence = extract(a.sqlite)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "intervals.json.gz").write_bytes(gzip.compress(json.dumps(evidence).encode(), mtime=0))
    (a.output / "overlap.json").write_text(json.dumps(analyse(evidence), indent=2) + "\n")
    print("Exported anonymised interval evidence and overlap summary.")


if __name__ == "__main__":
    main()
