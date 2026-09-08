# infe-sched M0 — profiling plan and kill criterion (2026-09-08)

## Context

Two components have been measured and killed:

| component      | outcome                          | root cause                                     |
|----------------|----------------------------------|------------------------------------------------|
| `infe-parsers` | parity, no win (round 5)         | parsing is on the SSE path, not between batches |
| `infe-kv`      | native impl that ships is slower | radix-cache CPU is ~2% of a decode step        |

The common pattern: decode dominates, and the CPU-side work around it is
single-digit percent of the step. `infe-sched` (BRIEF §6.3) sits in the same
place. The prior is that it will also be flat. **Profile first, with a kill
criterion** — the same methodology that worked twice before.

## The question

**What fraction of per-step wall time does the scheduler consume, and does
it spill past the GPU forward under overlap scheduling?**

If the scheduling decision (admission + batching + preemption) is <5% of
per-step wall time *and* fits inside the GPU forward shadow under overlap
scheduling, the scheduler is not on the critical path and building a Rust
replacement would not help — exactly as `infe-kv` showed.

## The seams (verified against SGLang v0.5.19 source)

### SGLang: plugin hook registry (not `--scheduler-cls`)

SGLang has **no `--scheduler-cls` equivalent**. The `Scheduler` class is
5,836 lines (plus 3,727 in `schedule_batch.py`, 1,519 in `schedule_policy.py`,
~13,000 across `scheduler_components/`). There is no flag to swap it out.

**But** SGLang has a **plugin hook registry** (`sglang/srt/plugins/`) that
can AROUND-hook or REPLACE any function/method/class in the codebase:

- Loaded via setuptools entry_points in the `sglang.srt.plugins` group
- Can also be loaded via PYTHONPATH + `SGLANG_PLUGINS=<name>` env var
- `HookType.AROUND` wraps any method: `fn(original_fn, *args, **kwargs) -> result`
- `load_plugins()` is called at scheduler startup (scheduler.py:5750), before
  the event loop starts

This means: **we can hook `Scheduler.get_next_batch_to_run`, `Scheduler.run_batch`,
and `Scheduler.process_batch_result` without forking SGLang**. That is the seam,
and it is more surgical than vLLM's (wrap one method vs replace the whole class).

### vLLM: `--scheduler-cls` exists, but is higher-friction

vLLM 0.28 has a real pluggable seam: `--scheduler-cls` accepts a class path,
resolved via qualname, loading a `SchedulerInterface` subclass. Used by
vllm-spyre. ABC churn is ~6 changes/half-year — relatively stable. But vLLM
is secondary (no GPU benchmarks on this machine against vLLM for this probe).

### The hot path per step

```
event_loop → ingest_requests → get_next_batch_to_run → run_batch → process_batch_result
```

- `get_next_batch_to_run` (line 3497): the scheduling decision. Calls
  `get_new_batch_prefill` → `PrefillAdder.add_one_req()` per request. This is
  admission, chunked prefill budgets, preemption, priority.
- `run_batch` (line 4193): GPU forward launch (CPU-side). In overlap mode, the
  result processing of batch N overlaps with the forward of batch N+1.
- `process_batch_result` (line 4538): result bookkeeping (decode/extend,
  detokenization, metrics, router load updates).

### Existing instrumentation in SGLang

- `SGLANG_RECORD_STEP_TIME=1`: records per-step wall time by batch size,
  available via `/get_server_info` endpoint
- `--enable-forward-pass-metrics`: ZMQ IPC stream of per-iteration metrics
- `--enable-mfu-metrics`: TFLOPS and memory bandwidth utilization
- `MetricsReporter.step_time_dict`: per-batch-size step time, accumulated
  in metrics_reporter.py

These tell us the *total* step time. They do **not** break down *which phase*
consumes the time. The phase-timer plugin fills that gap.

## The profiling kit

### 1. Phase-timer plugin (`shims/sglang/infe_sched_probe/`)

AROUND hooks on the three scheduler methods, recording wall-clock time. Zero
overhead when `SGLANG_SCHED_TIMER_OUT` is unset. At exit, dumps JSON with:

- Per-phase: calls, total_ns, mean_ns, p50/p90/p99, max
- Fractions: sched % / forward % / result % of per-step wall time
- `step_span`: wall time from `get_next_batch_to_run` entry to
  `process_batch_result` exit for one batch (the best per-step proxy under
  overlap scheduling)

Loaded via PYTHONPATH inside the container — no pip install needed, mirrors
the infe-parsers shim approach.

### 2. High-admission workload (`bench/harness/e2e_high_admission.py`)

Three modes that stress the scheduler differently from the existing
`e2e_tool_stream.py`:

| mode              | what it stresses                                               |
|-------------------|----------------------------------------------------------------|
| `mixed_lengths`  | Varying prompt + output lengths; mixed prefill+decode every step; long shared prefix for prefix-cache pressure |
| `short_burst`     | Many short requests in a tight window (admission under queue pressure) |
| `tool_call`       | The standard tool-call workload (baseline/control, matches earlier rounds) |

`mixed_lengths` is the key mode: varied `max_tokens` per request (32–200)
forces the scheduler to batch decode steps with different completion times,
keeping the prefill admission path active every step. The long shared prefix
(500 tokens of `"You are a helpful assistant. "` repeated) creates
prefix-cache pressure that exercises eviction and `match_prefix`.

### 3. Run driver (`bench/harness/run_sched_probe.sh`)

Runs two arms × three modes, interleaved:

- **stock**: no plugin, no timer (control)
- **probe**: timer plugin loaded, timer output captured

Same image (v0.5.19), same flags, same model (Qwen2.5-1.5B-Instruct). The
only difference is the plugin — this tests both the scheduling fraction *and*
whether the timer plugin itself adds overhead. Concurrency levels 8/64/256.

## Kill criterion (written down before running)

**If `get_next_batch_to_run` accounts for <5% of per-step wall time** under
any of the three workloads *and* the time fits within the GPU forward shadow
(i.e., `sched_of_step_pct` < `forward_of_step_pct` overlap), **stop and
write it up.** The scheduler is not on the critical path.

If the timer plugin shows <1% overhead vs stock (expected — it is 2 ×
`perf_counter_ns()` per call), the measurement is trustworthy.

If the scheduler is ≥5% of step time on any workload, it warrants further
investigation: examine whether the time is in prefix matching, prefill tile
budgeting, or preemption, and whether that specific sub-phase could benefit
from a Rust port. But the prior is that it will be flat — two components
have now confirmed that decode dominates at this scale.

## What to run

```bash
# On the 4090 machine (inside INFE_BENCH_DIR):
INFE_REPO=/path/to/infe PORT=18000 \
  bash bench/harness/run_sched_probe.sh 0 8 64 256

# Then summarize:
cd $INFE_BENCH_DIR/results && \
  python3 $INFE_REPO/bench/harness/summarize_sched_probe.py 'sched_probe_*.json'
```

## Prior art check

- **SGLang has not tried a native scheduler.** Git history of `schedule_policy.py`
  shows only refactor and feature commits — no Rust/Cython/C++ scheduler
  attempts. The Rust work in SGLang is limited to the router and the TreeCore
  (both already shipped).
- **vLLM's `--scheduler-cls` seam** is used by vllm-spyre for hardware
  constraints, not for performance. No known vLLM Rust scheduler exists.
- **The Dynamo runtime** (NVIDIA) includes a scheduler, but it is a separate
  serving framework (not a drop-in engine component).

Unlike `infe-kv`, where SGLang's own C++/Rust attempts gave us the answer
before we started, there is no prior native scheduler to learn from. This
makes the measurement more necessary, not less — but the prior from two dead
components is that CPU-side work at this scale is not the bottleneck.

## Deliverables

1. This document (plan + kill criterion)
2. `bench/harness/sglang_sched_timer.py` — the timer plugin source
3. `shims/sglang/infe_sched_probe/` — installable plugin package
4. `bench/harness/e2e_high_admission.py` — high-admission workload
5. `bench/harness/run_sched_probe.sh` — run driver (stock vs probe × 3 modes)
6. `bench/harness/summarize_sched_probe.py` — results summarizer
7. After the run: `docs/infe-sched-m0-findings.md` with the verdict
