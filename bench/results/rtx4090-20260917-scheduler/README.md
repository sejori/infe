# Scheduler investigation: RTX 4090, 2026-09-17

The [final findings](../../../docs/infe-sched-m0-findings.md) close the proposed
general Rust scheduling/admission investigation without a prototype. They also
identify substantial high-concurrency execution/output work; this is not a
blanket finding that CPU-side optimisation cannot help.

The [prespecified protocol](../../../docs/infe-sched-m0-plan.md) comprises six
counterbalanced stock/probe pairs and two additional CUDA/NVTX trace sessions.
Every server is fresh; all 73,472 measured requests completed with exact planned
token counts. The experiment uses SGLang v0.5.19, Qwen2.5-1.5B-Instruct, TP=1,
one RTX 4090, Python TreeCore and default overlap scheduling. The probe adds
instrumentation only. CPU values are percentages of one core, measured with
window-scoped cgroup deltas.

| Case | Stock median requests/s | Inclusive scheduling, probe median | Scheduling without simultaneous GPU activity, two traces |
|---|---:|---:|---:|
| Short c8 | 88.32 | 3.81% | 1.85% / 1.98% |
| Short c256 | 519.78 | 7.03% | 3.57% / 4.08% |
| Mixed c64 | 18.79 | 2.22% | 0.66% / 0.65% |
| Cache churn c64 | 14.33 | 2.30% | 0.44% / 0.36% |

Trace intervals are diagnostic, not causal speedup estimates. At concurrency
256, result processing exposes about 12% and forward/launch work about 24%,
with material profiler distortion. Read the full report before interpreting
the scheduling table as an engine-wide ceiling.

A separate two-pair unprofiled control raises decode CUDA graph coverage from
the resolved default maximum of 24 to 256. Throughput improves 11.4% / 13.0%
and E2E p50 falls 10.3% / 13.9%, while TTFT p99 rises 18.0% / 12.6%. All 16,384
additional measured requests passed token-count validation. This exploratory
control is retained in `graph-check/`, without mixing it into the original
six-pair comparison.

## Files and reproduction

- `manifest.json`: original configuration, balanced order, image/model pins and executed source hashes.
- `provenance.json`: additional hardware/tool versions and analysis source hashes, recorded after the run.
- `pair-*/session.json.gz`: all measured and excluded warmup requests, counters, telemetry and phase samples.
- `pair-*/server.log.gz`: scrubbed server logs, including resolved graph capture sizes.
- `pair-*-trace/derived/intervals.json.gz`: relative GPU, phase and scheduler-thread CUDA-API intervals.
- `pair-*-trace/derived/overlap.json`: per-trace overlap summaries.
- `summary.json`: six-pair effects, raw session metric distributions and both trace results.
- `graph-check/`: separate exploratory graph-coverage control selected after inspecting the traces.

Regenerate `summary.json` from the compressed sessions and interval evidence:

```bash
python3 bench/harness/summarize_scheduler.py bench/results/rtx4090-20260917-scheduler
python3 -m unittest discover -s bench/harness -p 'test_*.py'
```

The summarizer checks the complete schedule, every request's token counts,
summary accounting, real cache evictions, cache capacity and trace windows.
It also regenerates the graph comparison and verifies actual capture sizes in
the logs against the requested configurations.
It does not treat requests or method calls as independent replicates. Nested
phase durations are not added to their parents.

Raw Nsight reports and SQLite exports remain private. The public interval
schema contains only relative times and fixed phase/API-category names; host,
device, process, thread, path and container identifiers are discarded. No
generated text or input token IDs are included. Functional pilots are excluded.
