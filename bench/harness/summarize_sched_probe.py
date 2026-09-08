#!/usr/bin/env python3
"""Summarize the infe-sched M0 probe results.

Reads sched_probe_*.json (and .cpu.txt, .timer.json sidecars) and prints:
  1. Per-mode, per-arm e2e metrics (TTFT, ITL, e2e, stream_span, CPU%)
  2. Timer fractions (probe arm only): what % of step time is scheduling vs forward vs result
  3. Stock-vs-probe comparison (is the timer plugin itself adding overhead?)
  4. Kill-criterion verdict: is get_next_batch_to_run < 5% of step time?

Usage:
    python3 summarize_sched_probe.py 'sched_probe_*.json'

Note: the glob matches .timer.json files too. Files without a "levels" key
(timer sidecars) are loaded as timer data, not as e2e reports.
"""
import glob
import json
import statistics
import sys
import os
from collections import defaultdict


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr(xs):
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def load_reports(pattern):
    rows = defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        # Skip timer sidecar files — they are loaded explicitly below.
        if ".timer." in f or ".cpu." in f:
            continue

        r = json.load(open(f))
        basename = f.rsplit(".", 1)[0]  # strip .json
        cpu_file = basename + ".cpu.txt"
        timer_file = basename + ".timer.json"

        # Skip files that don't have the e2e report structure (timer data, etc.)
        if "levels" not in r:
            continue

        cpu = []
        try:
            cpu = [float(x.strip().rstrip("%")) for x in open(cpu_file) if x.strip()]
        except FileNotFoundError:
            pass

        timer = None
        try:
            timer = json.load(open(timer_file))
        except FileNotFoundError:
            pass

        mode = r.get("mode", "unknown")
        arm = r.get("arm", "unknown")
        engine = r.get("engine", "unknown")

        for lv in r["levels"]:
            s = lv["summary"]
            k = (mode, arm, s["concurrency"])
            d = rows[k]
            d.setdefault("ttft_p50", [])
            d.setdefault("itl_p50", [])
            d.setdefault("itl_p99", [])
            d.setdefault("e2e_p50", [])
            d.setdefault("chunks_per_s", [])
            d.setdefault("stream_span_ms", [])
            d.setdefault("cpu", [])
            d.setdefault("errors", 0)
            d["errors"] += s.get("errors", 0)
            d["cpu"].extend(cpu)
            for m in ("ttft_p50", "itl_p50", "itl_p99", "e2e_p50", "chunks_per_s"):
                if s.get(m) is not None:
                    d[m].append(s[m])
            for req in lv.get("requests", []):
                req_itls = req.get("itl_ms", [])
                if req_itls:
                    d["stream_span_ms"].append(sum(req_itls))

        # Attach timer data (probe arm only) at the run level, not per-conc
        if timer:
            rows[("_timer", mode, arm)] = timer

    return rows


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "sched_probe_*.json"
    rows = load_reports(pattern)

    # Print e2e table
    print(f"{'mode':16} {'arm':6} {'conc':>4} {'ttft_p50':>9} {'itl_p50':>8} "
          f"{'itl_p99':>8} {'e2e_p50':>8} {'chunks/s':>8} {'cpu%':>5} {'err':>4} "
          f"{'stream_span':>12}")
    print("-" * 100)

    modes = sorted(set(k[0] for k in rows if not k[0].startswith("_")))

    for mode in modes:
        for arm in ["stock", "probe"]:
            for (m, a, c), d in sorted(rows.items()):
                if m != mode or a != arm or m.startswith("_"):
                    continue
                ss = d.get("stream_span_ms", [])
                print(
                    f"{mode:16} {arm:6} {c:4d} "
                    f"{med(d['ttft_p50']):9.1f} {med(d['itl_p50']):8.2f} "
                    f"{med(d['itl_p99']):8.1f} {med(d['e2e_p50']):8.0f} "
                    f"{med(d['chunks_per_s']):8.0f} {med(d['cpu']):5.0f} "
                    f"{d['errors']:4d} "
                    f"{med(ss) if ss else 0:12.1f}"
                )

    # Timer fractions (probe arm only)
    print("\n=== Phase Timing (probe arm) ===")
    for mode in modes:
        key = ("_timer", mode, "probe")
        timer = rows.get(key)
        if not timer:
            continue
        f = timer.get("fractions", {})
        s = timer.get("get_next_batch_to_run", {})
        fwd = timer.get("run_batch", {})
        res = timer.get("process_batch_result", {})
        steps = timer.get("step_span", {}).get("calls", 0)

        print(f"\n  mode={mode}")
        print(f"    steps recorded:       {steps}")
        print(f"    get_next_batch_to_run: {f.get('sched_of_step_pct', 0):5.1f}% of step "
              f"(mean {s.get('mean_ns', 0) / 1000:.0f}µs, "
              f"p50 {s.get('p50_ns', 0) / 1000:.0f}µs, "
              f"p99 {s.get('p99_ns', 0) / 1000:.0f}µs)")
        print(f"    run_batch:             {f.get('forward_of_step_pct', 0):5.1f}% of step "
              f"(mean {fwd.get('mean_ns', 0) / 1000:.0f}µs)")
        print(f"    process_batch_result:  {f.get('result_of_step_pct', 0):5.1f}% of step "
              f"(mean {res.get('mean_ns', 0) / 1000:.0f}µs)")

        sched_pct = f.get("sched_of_step_pct", 0)
        forward_pct = f.get("forward_of_step_pct", 0)
        if sched_pct < 5.0:
            print(f"    >>> KILL CRITERION MET: sched={sched_pct:.1f}% < 5%")
            print(f"        The scheduler is NOT on the critical path.")
            print(f"        The forward pass ({forward_pct:.1f}%) dominates step time.")
            print(f"        Do not build infe-sched.")
        else:
            print(f"    >>> sched={sched_pct:.1f}% >= 5% — scheduler is a meaningful fraction")
            print(f"        Further investigation warranted (check overlap scheduling fit)")

    # Stock vs probe comparison
    print("\n=== Stock vs Probe (timer overhead check) ===")
    baseline = "stock"
    for mode in modes:
        for (m, a, c), d in sorted(rows.items()):
            if m != mode or a != "probe" or m.startswith("_"):
                continue
            s_key = (mode, "stock", c)
            s = rows.get(s_key)
            if not s:
                continue
            print(f"\n  mode={mode} conc={c}")
            for metric in ("ttft_p50", "itl_p50", "e2e_p50"):
                sm, im = med(s[metric]), med(d[metric])
                if sm > 0:
                    print(f"    {metric:12} stock={sm:8.2f}  probe={im:8.2f}  "
                          f"delta={100 * (im - sm) / sm:+6.1f}%")

            s_span = med(s.get("stream_span_ms", [])) if s.get("stream_span_ms") else None
            d_span = med(d.get("stream_span_ms", [])) if d.get("stream_span_ms") else None
            if s_span and d_span and s_span > 0:
                print(f"    {'stream_span':12} stock={s_span:8.2f}  probe={d_span:8.2f}  "
                      f"delta={100 * (d_span - s_span) / s_span:+6.1f}%  "
                      f"(delta-count invariant)")

            s_cpu = med(s.get("cpu", []))
            d_cpu = med(d.get("cpu", []))
            print(f"    {'cpu%':12} stock={s_cpu:8.0f}  probe={d_cpu:8.0f}")


if __name__ == "__main__":
    main()
