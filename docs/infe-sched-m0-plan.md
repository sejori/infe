# Final scheduler investigation protocol (2026-09-17)

Question: does Python scheduling or batch preparation expose enough CPU work
on the serving critical path to justify a native component? The prior parser
and TreeCore experiments did not demonstrate a win; they do not establish that
all CPU-side optimisation is futile.

## Fixed experiment

SGLang v0.5.19 (immutable image ID recorded), Qwen2.5-1.5B-Instruct pinned at
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, one RTX 4090, TP=1, context 16384,
memory fraction 0.85, Python TreeCore, seed 42, default overlap scheduling.
Synthetic token IDs deliberately remove tokenizer and language-quality effects.
Generation uses temperature 0 and `ignore_eos=true`; every response must have
the exact planned input/output token counts. This is a stress test, not a
representative customer trace or quality evaluation.

Six independent stock/probe pairs, with three orders of each direction shuffled
using seed 20260917. Every arm starts a fresh container. The probe changes only
instrumentation, never scheduling decisions. Four cases run in each session:

| Case | Concurrency | Input / output tokens | Warmup requests | Measured requests |
|---|---:|---|---:|---:|
| Short, low concurrency | 8 | 32 / 16 | 64 | 512 |
| Short, admission pressure | 256 | 32 / 16 | 256 | 4096 |
| Mixed lengths | 64 | 128/1024/4096 / 16/64/128 | 64 | 512 |
| Long-prefix reuse and churn | 64 | 8192 shared + 32 unique / 16 | 128 | 128 |

Use a fixed-size closed-loop client pool and identical deterministic request
sets across arms. Flush the prefix cache between cases, then warm up. For churn,
warmup visits 128 distinct 8192-token prefixes (1,048,576 prefix tokens), above
the pilot's 619,443-token cache capacity. Measurement mixes 75% hot-prefix
requests with 25% cold-family requests. Report actual cached and evicted tokens;
do not infer cache pressure merely from prompt length.

CPU counters and telemetry bracket only measured work. Explicit probe start/stop
excludes warmup and idle periods between cases. Per-session measurements are
independent replicates; individual requests and method invocations are not.
The stock/probe comparison measures instrumentation overhead, not Rust speedup.

## CPU and GPU attribution

The installed SGLang entry-point plugin records wall time and thread CPU time
for scheduling, forward launch, result processing, request ingestion and sample
launch. Nested ranges identify admission, running-batch updates, decode/extend
preparation, prefix matching/insertion and eviction. Do not add nested ranges
to parent totals. Hook registration is verified before collecting data.

Two further fresh sessions run under Nsight Systems with CUDA/NVTX tracing,
CUDA graph node tracing, CPU sampling disabled, and explicit measurement-window
NVTX ranges. CUDA profiler start/stop controls the trace; no synchronisation is
inserted into the scheduling loop. GPU kernels/copies/memsets share the trace
clock with CPU NVTX ranges. Intersect intervals to measure the portion of each
CPU range during which no GPU activity was recorded. This is an upper bound on
potentially removable exposed work, not proof of causation or guaranteed speedup.
Report profiler overhead and separate these diagnostics from the unprofiled A/B.

## Decision criterion, specified before the full run

Proceed to a bounded Rust prototype only if a named scheduling/bookkeeping
component plausibly exposes enough removable work for approximately 5% useful
end-to-end improvement at a tested operating point, and its cost is not simply
GPU execution/waiting, instrumentation overhead, client starvation or work better
removed with an engine synchronisation/graph fix. A high CPU percentage alone
does not qualify. Below that opportunity threshold, stop this component track.
If traces or workload checks fail, repair the measurement rather than treating
missing data as a negative result. Retain limitations and smaller possible gains.

## Reproduction and publication

Run `bench/harness/scheduler_investigation.py --help`. The default six pairs plus
`--trace-sessions 2` implement this protocol. `--pilot-count` and a concurrency
override are functional checks only, excluded from the final dataset.

Publish sanitised request/timer data, measured GPU specifications, derived trace
intervals and analysis. Raw Nsight reports/SQLite databases can contain machine
identity metadata and must remain local. No hostnames, SSH identities, home
paths, physical GPU UUIDs or container identifiers belong in public artifacts.

The earlier `feat/infe-sched-m0-probe` implementation is superseded: it did not
establish independent sessions or actual GPU overlap, relied on unregistered
PYTHONPATH plugin loading, and attempted to copy an exit-written timer before
exit. Its TreeCore regression claim was also withdrawn.
