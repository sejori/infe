# A/B benchmark harness

## Purpose

Measure `infe-parsers` (Rust) vs stock Python parsers in vLLM and SGLang,
following the BRIEF's success criterion: a reproducible A/B report on the
same model, same hardware, same engine commit, showing the engine with the
component enabled versus the stock Python path.

## Layers

### Layer 1: Microbenchmark (in-Crate, Criterion)

Pure parser throughput — no engine, no PyO3. Measures ns/byte and deltas/sec
for each dialect across input shapes (single chunk, multi-chunk, plain content).

Location: `crates/infe-parsers/benches/parse_stream.rs`

This is the "ceiling" — if the Rust parser is not faster than Python here,
it won't be faster end-to-end.

### Layer 2: PyO3 Crossing Cost (M0 Deliverable)

Measures the cost of one `feed()` call through PyO3 vs one equivalent Python
method call. This quantifies the per-call overhead and validates the
batch-vs-per-token thesis.

Script: `bench/harness/pyo3_crossing.py` (to be written)

### Layer 3: End-to-End A/B (M1 Deliverable)

Runs the full engine with tool-heavy traffic, comparing stock vs infe-parsers.

- Load generator: `vllm bench serve` (widest datasets, SLO sweep)
- Datasets: ShareGPT-style chat, tool-call-heavy
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
