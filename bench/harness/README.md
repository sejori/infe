# A/B benchmark harness

## TreeCore independent-session reproduction

For SGLang issue [#38536](https://github.com/sgl-project/sglang/issues/38536), use
`treecore_sessions.py`. The old `run_ab_docker.sh` runs one session per arm;
its rounds and CPU samples are not independent backend replicates.

```bash
python3 bench/harness/treecore_sessions.py \
  --gpu 1 --hf-cache "$HOME/infe-bench/hf" \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output bench/results/treecore-sessions
```

Requires Linux, Python stdlib, Docker, NVIDIA container support, readable host
cgroup-v2 CPU counters, and the image/model already cached. Pick an available
GPU and port. The harness checks for compute processes before each session,
uses unique container names, and removes only its own containers. Its advisory
GPU lock coordinates other copies of this harness, not arbitrary GPU jobs.

The default protocol is fixed before collecting data:

- Six pairs / twelve fresh server sessions on one GPU. Three pairs run Python
  first, three Rust first, shuffled using the recorded seed. Within-session
  rounds are not independent replicates. Both arms use the same resolved image
  ID (default tag v0.5.19), model revision, server seed, and stock `qwen` parser.
- Concurrency 8, 64, 256, in that order in every session. Two discarded warmup
  rounds at each level, then three measured rounds. This tests a warmed cache;
  it deliberately differs from Round 6's four-request initial warmup. Raw warmup
  requests are retained. No optional early stopping based on the effect size.
- Whole-container CPU is the cgroup CPU-time difference over each measured
  round. The level CPU percentage is total CPU time / total measured wall time,
  expressed as percent of one core. No startup/warmup samples or main-PID fallback.
- GPU temperature, clocks, power, utilisation and host load are sampled during
  each measured level. Telemetry samples are diagnostic, not independent units.
  Measurement windows are short, so telemetry may be sparse. Other host users
  can still introduce CPU contention; this is not a dedicated-machine test.
- Each session contributes one median-of-rounds value per latency metric/load.
  The comparison preserves every pair's Rust/Python percentage change and shows
  its median and range. It does not pool requests into a significance test or
  claim that a nonsignificant/small result establishes equivalence.
- HTTP failures or missing/malformed tool calls abort the experiment and retain
  partial data and server logs. Successful parity checks cover two expected tool
  names, valid argument objects, and IDs; they do not validate semantic answers.

`stream_span_p50` is the median time from the first to last meaningful SSE chunk
(sum of inter-chunk intervals), not time divided by chunk count. ITL measures SSE
chunks, not necessarily model tokens. Delta counts remain visible for comparison.

Artifacts: `manifest.json` (schedule/config/image/hardware/harness hashes), one
JSON and server log per session, and `comparison.json` (paired session results).
Published metadata omits hostnames, SSH identities, local cache/output paths,
GPU UUIDs and Docker identifiers. The UUID is used only for the local GPU lock.
Engine logs are scrubbed for machine identifiers before being saved. Keep
hardware specifications and measurements for reproducibility.
Keep the full set together. Do not feed these files to `summarize_ab.py`, which
handles the legacy schema. The legacy summarizer now leaves CPU unavailable
unless the input explicitly supplies a window-scoped value.

Render a complete run (also accepts archived `pair-*.json.gz` files):

```bash
python3 bench/harness/summarize_treecore.py bench/results/treecore-sessions
```

If the latency regression survives independent sessions, profile the scheduler
and TreeCore before choosing an optimisation. If it does not, report failure to
reproduce under this warmed protocol. A long-prefix study is a separate workload,
not a substitute for reproducing the original short-prefix observation.

Run harness checks with:

```bash
python3 -m unittest discover -s bench/harness -p 'test_treecore_sessions.py'
```

## Purpose

Measure `infe-parsers` (Rust) vs stock Python parsers in vLLM and SGLang,
following the BRIEF's success criterion: a reproducible A/B report on the
same model, same hardware, same engine commit, showing the engine with the
component enabled versus the stock Python path.

## Layers

### Layer 1: Microbenchmark (in-Crate, Criterion)

Pure parser throughput — no engine, no PyO3. Measures ns/chunk and
deltas/sec for each dialect across input shapes (single chunk, multi-chunk,
plain content, concurrent streams).

**Mocked inference model:** The benchmark uses `MockTokenStream`, a token
generator inspired by `inference-lab`'s `serve::engine` `TokenEvent`
pipeline. Instead of a real GPU decode loop, it emits pre-split text chunks
that simulate per-token decode output — including tool-call markers split
across token boundaries, JSON arguments, and reasoning blocks.

This isolates the parser's CPU cost from the engine. It runs in CI (GitHub
Actions) with reduced sample sizes and measurement times so it completes
in minutes, not hours.

Location: `crates/infe-parsers/benches/parse_stream.rs`

**Benchmarks:**
- `hermes/single_tool_call` — one Hermes tool call streamed across 10 chunks
- `hermes_plain/no_tool_calls` — 18 chunks of plain content (pass-through path)
- `llama3_json/single_tool_call` — one Llama-3 JSON tool call across 6 chunks
- `deepseek_reasoning/reasoning_block` — reasoning block + content across 15 chunks
- `concurrent/hermes_streams/{64,256,1024}` — N concurrent parsers fed in one step

The concurrent-stream benchmark is the proxy for the ITL p99 claim: if
parsing 1024 streams in one step is cheap, the batched approach wins over
per-token Python crossings.

### Layer 2: PyO3 Crossing Cost (M0 Deliverable)

Measures the cost of one `feed()` call through PyO3 vs one equivalent Python
method call. This quantifies the per-call overhead and validates the
batch-vs-per-token thesis.

Script: `bench/harness/pyo3_crossing.py` (to be written)

### Layer 3: End-to-End A/B (M1 Deliverable)

Runs the full engine with tool-heavy traffic, comparing stock vs infe-parsers.
This uses `inference-lab` as the mocked inference server:

- `inference-lab --serve --enable-directives` provides an OpenAI-compatible
  API that emits scripted tool-call responses via the `<<respond:...>>`
  directive system. This gives deterministic, reproducible tool-call traffic
  without GPU costs.
- The infe-parsers shim plugs into the same `TokenEvent` pipeline, replacing
  the stock Python parser path.
- Load generator: `inference-lab`'s built-in workload generator, or
  `vllm bench serve` against the inference-lab server.
- Concurrency levels: 64, 256, 1024 concurrent streams
- Metrics: ITL p50/p99, API-server CPU%, throughput (tokens/sec)

Config: `bench/scenarios/tool_heavy.yaml` (to be written)

### Layer 4: Parity Matrix (CI, nightly)

Conformance fixtures run against pinned engine versions in CI, and against
engine `main` nightly. Produces a per-dialect pass/fail matrix. Drift is
reported, not failed.

## Report format

Raw Criterion JSON is committed to `bench/results/`. The A/B report uses the
llm-d Benchmark Report 0.2.1 schema with `cfg_id` hashes for stack and load
config (BRIEF §13). Comparability follows the kubernetes-sigs/inference-perf
`comparability.md` checklist.

## Negative results

If the improvement is unmeasurable at GPU-bound operating points, the report
says so plainly (BRIEF §11). The claim is CPU-side; pick latency-bound points
for the headline and show no regression elsewhere.
