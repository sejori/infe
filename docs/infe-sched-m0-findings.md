# Scheduler investigation — final measurement (2026-09-17)

## Decision

**Finish M0 without implementing the proposed Rust admission/batch-preparation
component. Keep the high-concurrency execution/output path as a separate
opportunity.** Scheduling exposes at most 4.08% of a traced window, and the
individual preparation/admission/cache ranges are smaller. That does not justify
a general scheduler replacement against the approximately 5% useful end-to-end
opportunity criterion. It does not establish that every possible scheduler
change has less than 5% impact.

There is meaningful CPU-side headroom at concurrency 256: result processing and
forward/launch work expose much more time than batch selection. A separate
two-pair control increasing decode graph coverage improved throughput by
11.4% / 13.0% with the existing Python scheduler, although tail TTFT worsened.
Result/output processing could warrant a narrower follow-up after graph tuning;
this investigation has not isolated a portable Rust boundary or measured a Rust
speedup there. Calling the whole native-code experiment a proven dead end would
overstate the evidence.

The [dataset and reproduction notes](../bench/results/rtx4090-20260917-scheduler/README.md)
retain every planned full session and the derived trace evidence.

## Method and scope

The [protocol](infe-sched-m0-plan.md) was fixed before the full run: six
counterbalanced stock/probe pairs and two additional traced sessions, each
starting a fresh SGLang server. The probe is instrumentation only; no Rust
scheduler or changed scheduling decision is being compared. Requests and
individual method calls are nested observations, not independent replicates.

The environment is SGLang v0.5.19, Qwen2.5-1.5B-Instruct at the recorded model
revision, one RTX 4090, driver 590.48.01, and a Threadripper PRO 7975WX with 64
logical CPUs. TP=1, context length 16384, static memory fraction 0.85, Python
TreeCore, seed 42, default overlap scheduling and CUDA graphs. Client and engine
share the host; there is no CPU affinity or GPU-clock pinning. The manifest pins
the image digest and executed runner/plugin source hashes. Nsight Systems is
2026.4.1.191, with CUDA/NVTX and graph-node tracing, and CPU sampling disabled.

The four cases are short requests at concurrency 8/256, mixed input/output
lengths at concurrency 64, and long-prefix reuse/churn at concurrency 64. They
use synthetic token IDs, fixed output lengths and `ignore_eos=true`. Counts are
validated for every response, but this is not a text-quality or customer-traffic
evaluation. The closed-loop client uses the same request set in each session;
actual arrival order, batching and cache hits can vary.

Cache-churn warmup visits 128 distinct 8192-token prefixes: 1,048,576 prefix
tokens against 619,443 tokens of cache capacity. Measurement mixes 75% hot reuse
and 25% cold-family requests. Actual evictions are checked. Different cache hits
mean identical requested input/output counts do not imply identical GPU work;
the cache-churn A/B is therefore a workload-sensitivity diagnostic as well as
an instrumentation check.

## Unprofiled results and instrumentation calibration

All twelve unprofiled sessions completed: 62,976 measured requests, with no
request failures or token-count mismatches. Effects below are the median of six
paired `100 * (probe / stock - 1)` values, followed by their full range. Negative
latency changes mean the instrumented arm was faster; these are observer/run
effects, not evidence of a Rust speedup. No requests are pooled into a
significance test, and a small median is not a demonstration of equivalence.

| Case | Stock req/s, median | Stock E2E p50, ms | Probe E2E p50 effect | Probe window-duration effect |
|---|---:|---:|---:|---:|
| Short c8 | 88.32 | 90.27 | −0.18% [−0.59, +0.30] | −0.10% [−0.58, +0.42] |
| Short c256 | 519.78 | 476.70 | −1.45% [−8.56, +2.55] | −1.20% [−7.67, +1.62] |
| Mixed c64 | 18.79 | 3031.71 | −0.26% [−1.31, +1.22] | +0.20% [−0.43, +0.83] |
| Cache churn c64 | 14.33 | 4392.02 | +1.41% [−3.31, +7.36] | +1.37% [−3.66, +7.29] |

Inclusive scheduling wall time is 3.81%, 7.03%, 2.22% and 2.30% of the measured
window respectively (medians across six probe sessions). The high-concurrency
case therefore needs GPU-overlap evidence; its inclusive cost alone is large
enough to attract attention.

In cache churn, measured cached tokens range from 734,502 to 802,658 across the
twelve sessions. Probe sessions actually evict 267,021–315,997 tokens during
measurement. Matching, insertion and eviction take median inclusive wall
fractions of 0.37%, 0.46% and 0.22% respectively. This verifies real cache
pressure without treating it as proof of a hot cache-management CPU loop.

## CUDA overlap results

Both fresh traced sessions completed, adding 10,496 measured requests. Each
cell gives the two trace results, as percentages of that trace's measurement
window. Scheduling includes its nested admission/preparation/cache work;
"bookkeeping" is the union of scheduling, result processing and ingestion.

| Case | GPU active | Scheduling GPU-uncovered | Bookkeeping GPU-uncovered | Result GPU-uncovered | Forward/launch GPU-uncovered |
|---|---:|---:|---:|---:|---:|
| Short c8 | 91.06 / 90.89 | 1.85 / 1.98 | 3.00 / 3.16 | 0.87 / 0.89 | 5.40 / 5.37 |
| Short c256 | 57.52 / 56.52 | 3.57 / 4.08 | 16.09 / 16.75 | 12.08 / 12.17 | 23.88 / 24.39 |
| Mixed c64 | 87.15 / 87.38 | 0.66 / 0.65 | 1.93 / 2.07 | 1.22 / 1.38 | 10.51 / 10.18 |
| Cache churn c64 | 94.36 / 95.02 | 0.44 / 0.36 | 0.84 / 0.73 | 0.28 / 0.27 | 2.32 / 2.39 |

This does **not** support calling all CPU-side work irrelevant. At concurrency
256, result processing alone has 11.13% / 11.22% of the window outside both GPU
activity and CUDA API calls; forward/launch has 19.37% / 19.91%. Those are real
places to investigate. Batch selection is smaller: excluding CUDA API calls
leaves 3.16% / 3.60% in this case. Nested decode preparation exposes only
0.53% / 0.55%, and extend preparation 0.77% / 0.89%, even before API exclusion.

The low-concurrency result timer illustrates why attribution matters: CUDA
synchronisation alone accounts for 55.39% / 54.00% of that window inside result
processing, while the entire result range exposes less than 0.9%. At concurrency
256, the corresponding synchronisation fraction is only about 0.02%; its result
processing cost must not be dismissed as the same GPU wait.

Tracing changes window duration versus the median of six unprofiled probe
sessions by +5.95% / +6.20% (short c8), +12.65% / +16.52% (short c256),
+7.17% / +6.96% (mixed), and +10.99% / +6.97% (churn). These are descriptive,
unpaired comparisons, not isolated profiler-overhead estimates: arrival order,
batching and cache hits vary, especially in churn. The high-concurrency profile
is meaningfully perturbed and needs an unprofiled intervention before making a
speedup claim. Trace-container CPU also includes profiler overhead.

Startup logs show default decode graph captures only for batch sizes
`[1, 2, 4, 8, 12, 16, 24]`, while the high-concurrency workload runs larger
batches. Prefill graphs cover token counts through 2048. This identifies a
configuration control to test before choosing a Rust boundary: increase decode
graph coverage with `--cuda-graph-max-bs-decode 256`.

## Exploratory graph-coverage control

After inspecting the traces, run two further balanced pairs on the same GPU,
image and model: default/graph-256, then graph-256/default. Each arm starts a
fresh server, uses the uninstrumented stock scheduler, and runs only short c256
with 256 warmup and 4096 measured requests. All 16,384 measured requests passed
the fixed token-count checks. Logs confirm decode graph captures through 24
for the default configuration and through 256 for the intervention. Memory
fraction and other server settings are unchanged.

| Pair | Default → graph-256 req/s | Throughput effect | E2E p50 effect | TTFT p99, default → graph-256 |
|---|---:|---:|---:|---:|
| Default first | 536.51 → 597.54 | +11.38% | −10.33% | 272.87 → 321.89 ms (+17.97%) |
| Graph-256 first | 547.84 → 619.01 | +12.99% | −13.91% | 259.94 → 292.65 ms (+12.59%) |

This is an actionable configuration result, with a tail-latency tradeoff. It is
exploratory: two pairs, one synthetic operating point, chosen after viewing the
profiles. It is not a six-pair confirmation, a blanket recommendation for all
loads, or a Rust speedup. The graph change was not re-profiled, so the original
12% result-processing exposure must not be assumed to remain after tuning.
That path needs isolation and measurement under the tuned configuration before
proposing a native output/bookkeeping component.

The [four-session dataset](../bench/results/rtx4090-20260917-scheduler/graph-check/comparison.json)
is separate from the original 14-session protocol. Reproduce with:

```bash
python3 bench/harness/scheduler_graph_check.py \
  --gpu 1 --hf-cache "$HOME/infe-bench/hf" \
  --plugin "$PWD/shims/sglang/infe_sched_probe" \
  --output "$HOME/infe-bench/scheduler-graph-check"
```

The plugin directory is mounted by the shared runner but not installed or
enabled in these stock sessions. The wrapper appends only the recorded decode
graph flag to the otherwise unchanged server command. The original runner
source remains identical to the full experiment's manifest hash.

## How to interpret the profiles

Unprofiled probe sessions record inclusive wall and thread CPU time. Nested
admission, batch preparation and cache ranges must not be added to their parent
scheduling/result totals. `run_batch` is a CPU method containing model execution
and CUDA calls; its duration is not a measurement of GPU execution time.

Nsight provides a shared clock for CPU ranges and GPU kernels/copies/memsets.
The analysis unions GPU activity across streams and intersects it with each
CPU range. "GPU-uncovered" means no captured GPU activity occurred during
that part of the range. It estimates directly exposed work in these executions;
it is not a causal speedup estimate or a mathematical bound on engine-wide
improvements, queueing effects or different scheduling policies. Client
starvation and instrumentation can contribute to the exposed intervals.

The secondary CUDA-API analysis matches calls to the scheduler thread, then
discards its identifier. It separates synchronisation, memcpy, launch and other
API time. Excluding GPU activity and CUDA API intervals together avoids treating
those native calls as removable Python bookkeeping. Their durations are not
added separately when they overlap.

Result processing contains explicit `copy_done.synchronize()` calls in the
[pinned engine's batch result processor](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/managers/scheduler_components/batch_result_processor.py).
An inclusive result-processing timer, even with high thread CPU consumption,
therefore cannot establish a large Python loop that Rust would eliminate.

## Limits and publication

This experiment does not cover vLLM scheduling, multiple GPUs, MoE, speculative
decoding, disaggregation, priority/fairness policies, or sustained preemption
pressure. It does not exclude smaller gains or a CPU bottleneck on another
model/hardware combination. A future optimisation claim needs a named removable
cost and an independently replicated end-to-end comparison on its target load.

The public data contains request timings, token/cache counts, window-scoped CPU
counters, telemetry, scrubbed logs and relative trace intervals. Raw Nsight
reports and SQLite databases remain private because they retain machine and
process identity metadata. Functional pilots are excluded by design.
