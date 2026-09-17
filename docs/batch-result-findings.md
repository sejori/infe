# Batch-result ownership: tuned diagnostic findings

The proposed state-owning batch boundary is implemented as a standalone Python
control and replay harness. **Do not port it to Rust yet.** With decode graphs
through batch size 256, its plausible remaining opportunity falls below the
prespecified approximately 5% serving-improvement screen. This is a scoped
negative result for these workloads, not evidence that native serving is exhausted.

## What ran

SGLang 0.5.19, pinned image digest and Qwen2.5-1.5B-Instruct revision from the
[manifest](../bench/results/rtx4090-20260917-batch-results/manifest.json), TP=1,
RTX 4090, static memory fraction 0.85, default overlap scheduling and
`--cuda-graph-max-bs-decode 256` in every arm. All six logs confirm graph capture
through 256. Two balanced fresh stock/probe pairs calibrate the additional hooks;
two separate fresh Nsight sessions provide interval evidence. Each session uses
32-token synthetic prompts and fixed 16-token streaming outputs: c8 has 64 warmup
and 512 measured requests; c256 has 256 warmup and 4,096 measured requests.
No optional generation features were enabled. A small functional pilot and a
preflight port-bind failure are excluded; no measured sessions were discarded.

Additional hooks separate normalization, completion checks, finish actions, cache
release, output accumulation, payload packing and sender handoff. They observe
existing Python execution, without changing decisions. Both ordinary probe and
Nsight instrumentation can perturb scheduling. CPU cost is whole-container cgroup
CPU time over each measured window, excluding warmup/startup. Raw per-request
latency/token counts and CPU accounting are retained and revalidated.

## Attribution and decision

Percentages below are fractions of each **profiled wall-clock window** where the
phase overlaps neither GPU activity nor scheduler-thread CUDA APIs. Nested phases
are inclusive and must not be added. They screen opportunities; they are not
causal critical-path measurements or predicted speedups.

| Work at c256 | Trace 1 | Trace 2 |
|---|---:|---:|
| All result processing | 6.90% | 7.63% |
| Cache release, inside result processing | 3.43% | 3.22% |
| Result processing outside cache release (interval subtraction) | 3.47% | 4.41% |
| Result processing outside finish actions and cache release | 3.09% | 3.94% |
| Token normalization | 0.046% | 0.046% |
| Completion checks | 0.323% | 0.442% |
| Entire output streamer, including packing and send | 0.948% | 1.290% |
| Payload packing alone | 0.027% | 0.030% |
| Sender handoff alone | 0.402% | 0.473% |

At c8, all result processing contributes only 0.54–0.56% outside GPU/CUDA work.
At c256, even assigning **all** non-cache result work to the candidate stage
leaves 3.47–4.41%, before paying for input packing, output consumption and a native
boundary. Removing 4.41% of elapsed time perfectly would imply only about 4.6%
more throughput under a simple serial model. Real overlap changes can differ;
these measurements justify declining the port, not a universal upper bound.

The candidate Python owner does not own cache release. Including that 3.2–3.4%
would require a materially broader cache/allocator ownership experiment, with
engine-specific side effects. The earlier TreeCore comparison tested a different
boundary and does not settle this question. Cache release is the next local
substage to inspect if this investigation continues; do not label it a demonstrated
Rust opportunity.

The calibrated probe changes c8 throughput by −0.21%/−0.02%, and c256 throughput
by +5.21%/−1.70% across the two pairs. This variation prevents a precise claim about
probe overhead. Nsight throughput is 4.9–5.1% below the median ordinary probe at
c8 and 1.9–4.7% below it at c256. Those comparisons are unpaired. The detailed
[summary](../bench/results/rtx4090-20260917-batch-results/summary.json) includes
both pairs, CPU microseconds per output token, median/tail latency and trace
perturbation. There is **no Python-control or Rust serving speedup claim** here.

## What is implemented

[ABI v1 and the Python control](../python/infe-batch-results/README.md) own request
history, completion state and output cursors across calls. The interface uses
packed numeric batch columns, generation-stamped handles, ordered steps and
exactly-once completion/release notifications. Validation errors do not partially
commit a batch. Unsupported generation features are explicitly outside the ABI.
The engine retains actual cache release and acknowledges completed lifetimes.

The deterministic synthetic replay passed against the actual pinned SGLang
completion method: **640 admissions, 1,280 steps, 640 terminal records, 8,948 ignored
late/finished rows and 10,240 engine-state comparisons**, with 3,820 emitted tokens.
[Recorded result](../bench/results/rtx4090-20260917-batch-results/engine-replay.json).
Cancellation is checked at an already-observed terminal boundary. This is not
live-engine integration, captured GPU batch replay, or full HTTP parity. Unit tests
also cover malformed atomic batches, token/length ties and stale slot reuse.

## Existing native alternative

The pinned image already contains `sglang.srt.rust_extensions._server`.
`SGLANG_RUST_SERVER=1` selects a Rust API/tokenization/detokenization stack embedded
in the scheduler process. Its `RustServer.push_generation` accepts a whole output
batch, using a columnar message header and raw numeric buffers. Scheduler request
bookkeeping remains Python. Relevant installed sources are
`sglang/srt/rust_server/server.py` and
`sglang/srt/managers/scheduler_components/output_streamer.py`.

This is worth reusing before building a second native output transport. It also
changes thread placement/CPU affinity, so its A/B would measure an architectural
bundle. Extension presence and source behavior were inspected; this investigation
did not launch or benchmark that server. Scheduler-side output work here is only
0.9–1.3%; downstream API/detokenization costs are outside those scheduler ranges.
A future output-heavy/text-heavy workload should profile that broader path first,
then compare existing native serving with balanced sessions and output parity.

## Reproduction and privacy

Run `batch_result_investigation.py --gpu <index> --hf-cache <cache>
--base-plugin shims/sglang/infe_sched_probe --output <new-directory>` on the
benchmark machine. Defaults run two pairs and two traces; the driver injects
additional phases into a copied plugin and leaves the original plugin unchanged.
Export each private Nsight report with the image's `nsys export --type sqlite`,
then use the driver's `--sqlite <file> --output <derived-directory>` mode.

Run `python3 bench/harness/summarize_batch_results.py
bench/results/rtx4090-20260917-batch-results` to reproduce the published summary.
It verifies the session set/order, requests, token totals, graph capture, metrics,
CPU accounting and derived interval summaries. Source hashes identify the run
scripts and generated probe; the analysis records its own hash. The generated
probe can be reconstructed from the base plugin plus the driver.

Public traces contain only whitelisted phase labels and relative timing intervals.
No raw Nsight/SQLite reports, generated text, request IDs, machine identifiers or
SSH details are included. Client address/port pairs and server process numbers are redacted from logs.
Session JSON is losslessly gzip-compressed for storage.
