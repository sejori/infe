# A/B benchmark harness

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
