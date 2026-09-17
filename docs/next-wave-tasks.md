# Next-wave task list — handover (updated 2026-09-17)

## Where the project is

`infe-parsers` is **done and closed**: correct, drop-in on both engines, dual-published, and it does not make
either engine faster. Round 5 scored M1 honestly — criteria (1) drop-in, (2) parity and (4) published are met;
(3) measured improvement is not. Full numbers and reasoning: `docs/review-2026-09-04.md` §"Round 5".

Do not spend more rounds tuning it for speed. Parsing sits on the SSE path, not between GPU batches; the
component's real value was proving manifest -> crate -> wheel -> shim -> conformance -> A/B end to end on two
engines, which it did.

## `infe-kv`: deprioritised; Round 6 interpretation corrected (2026-09-17)

**Do not build it without a measured opportunity.** Round 6 observed higher Rust
latency, but one session per backend in fixed order cannot establish a backend
regression. Its whole-run CPU medians cannot establish a CPU saving or blocking
mechanism. The earlier claims of independent confirmation and 6% lower CPU are
withdrawn. See `docs/infe-kv-m0-findings.md` for the correction and
`bench/harness/treecore_sessions.py` for the independent-session reproduction.

The [completed follow-up](../bench/results/rtx4090-20260917-treecore-sessions/README.md)
used six counterbalanced pairs on an RTX 4090: median paired E2E effects −0.33%,
−1.03%, +2.38% at concurrency 8/64/256. The original 11–12% slowdown was not
reproduced under this warmed protocol. No CPU-saving or blocking mechanism was
established. Recommend closing upstream #38536 as unconfirmed, retaining the data.

## `infe-sched`: M0 complete; general scheduler replacement not justified

The [final investigation](infe-sched-m0-findings.md) ran six independent
stock/probe pairs and two CUDA/NVTX traces across short, mixed-length and
long-prefix churn workloads. Scheduling exposed at most 4.08% of a measured
trace window; individual admission/preparation/cache ranges were smaller.
Do not implement the proposed Rust admission/batch-preparation component on
this evidence. The old `feat/infe-sched-m0-probe` branch is superseded.

This is not a blanket negative result for CPU-side optimisation. At concurrency
256, result processing exposed about 12% and forward/launch work about 24% of
the traced window. Tracing perturbed that workload, so these are diagnostic
opportunities, not promised speedups. Default decode graph coverage stopped at
batch size 24; the separate graph-coverage control is recorded in the findings.
Increasing coverage to 256 improved throughput by 11.4% / 13.0% in two balanced
pairs, while TTFT p99 worsened by 18.0% / 12.6%. This is a measured configuration
lever, with a tradeoff, rather than a measured Rust benefit.

The next performance work, if resumed, should validate graph coverage and its
tail-latency tradeoff on the target workload, then re-profile and isolate
result/output processing under the tuned configuration. It needs
a concrete removable cost and a bounded interface before choosing Rust. Do not
restart a broad engine or scheduler rewrite merely because CPU utilisation is
high. Larger gains on other hardware, models or scheduling policies remain
untested.

| Component | Status |
|---|---|
| `infe-parsers` | Functional/parity work complete; no demonstrated serving speedup |
| `infe-kv` | Not built; original TreeCore regression unconfirmed after independent sessions |
| `infe-sched` | M0 complete; no general replacement justified; execution/output opportunity retained |

The small engineering items below do not block this performance verdict.

## Carried-over items (small, not blocking)

| # | Where | What |
|---|---|---|
| B7 | `crates/infe-parsers/src/types.rs` | Done: thread-local xorshift IDs are already implemented and tested. |
| D4b | CPU measurement | TreeCore reproduction now uses cgroup counter deltas inside each measured round. Legacy whole-run samples must not be used for per-level CPU or independent-sample tests; a shorter sampling interval alone does not fix the experimental unit. |
| D7 | `bench/harness/run_ab_docker.sh` | The sampler exits only when the container disappears, so the driver kills it before `wait`. A `--stop-file`/`--duration` would be cleaner. |
| C | conformance | 14 Rust fixtures, all synthetic. Both round-4 blockers were *shim* bugs that no Rust fixture can catch; the off-GPU probes (`bench/harness/parity_probe_*.py`, `probe_feed_trace.py`) caught both in ~1 min each. Promote them to a CI job that runs inside the pinned engine images — that is the highest-value testing work outstanding. |

## Two habits worth keeping

1. **Probe off-GPU before every run.** The probes caught the round-4 double-feed and the trailing-`{}` bug in
   about a minute each, before any GPU time. Every regression so far was visible without a GPU.
2. **Trust the delta-count-invariant metric.** ITL falls whenever an arm emits more chunks, with nothing
   getting faster; rounds 3 and 4 both produced double-digit "wins" that `stream_span` later showed were flat.

---

# Appendix — historical task list (infe-parsers, rounds 1-5)

## A. Make the infe arms run at all (engine-side, small)

| # | File | Change | Evidence |
|---|---|---|---|
| A1 | `shims/vllm/infe_parsers/__init__.py` | Import `DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall` with a try/except: `vllm.entrypoints.generate.base.protocol` (main) -> fallback `vllm.entrypoints.openai.engine.protocol` (<=0.28.0). Drop the inline import inside `extract_tool_calls`. | `diff -u shims/vllm/infe_parsers/__init__.py bench/results/rtx4090-20260904/scratch/vllm_shim.py` |
| A2 | same | `DeltaMessage(tool_calls=delta_tool_calls)` — always a list; 0.28 rejects `None` with a pydantic error on every stream. | same diff |
| A3 | `shims/sglang/infe_parsers/__init__.py` | Implement abstract `structure_info(self)` on the detector (mirror `HermesDetector`). Without it `FunctionCallParser` cannot instantiate the class. | `diff -u shims/sglang/infe_parsers/__init__.py bench/results/rtx4090-20260904/scratch/sglang_shim.py` |
| A4 | new `python/infe-parsers/python/infe_parsers/shims/sglang/launch.py` | File-based launcher (not `-c`/stdin — multiprocessing spawn re-imports `__main__` from path). | `bench/results/rtx4090-20260904/scratch/launch_sglang.py` |
| A5 | `shims/*/infe_parsers/` | Rename: these packages shadow the wheel if ever on `sys.path`. Moved inside the wheel as `infe_parsers.shims.vllm` / `infe_parsers.shims.sglang`. | review §3 |

**Acceptance**: `vllm serve ... --tool-call-parser infe_hermes --tool-parser-plugin infe_parsers_vllm` and
`python -m infe_parsers_sglang.launch ... --tool-call-parser infe_hermes` both serve a streamed tool call with HTTP 200.

## B. Make the Rust parser output-compatible (the real work)

All items B1-B8 are **done** as of round 5. Kept for reference.

| # | File | Change | Status |
|---|---|---|---|
| B1 | `hermes.rs::extract_name` | Return the `arguments` sub-object, not the wrapper. | done |
| B2 | `hermes.rs` streaming | Emit `arguments_fragment` as the diff (partial-JSON semantics, matching vLLM's `hermes_tool_parser.py`). | done (round 3) |
| B3 | `types.rs` / all dialects | Assign `state.id` and `state.index` on first delta per call. | done |
| B4 | `parser.rs` | `finish()` closes an open tool call the way stock does. | done |
| B5 | reasoning | `deepseek_reasoning` through engines' reasoning interfaces. | pending (not blocking) |
| B6 | `hermes.rs` | Marker-less continuation calls (bare `{` after completed call). | done |
| B7 | `types.rs::make_tool_call_id` | Random ids, not index-derived. | done |
| B8 | SGLang shim | Nameless argument fragments forwarded, not dropped. | done |

## C. Conformance that would have caught B

- Mine fixtures from `vllm/tests/tool_use/` and `tests/entrypoints/openai/tool_parsers/` and SGLang `test/srt/function_call/`.
- Fixture `expected_tool_calls[].arguments` must be set and asserted (currently `null` -> skipped).
- Promote off-GPU parity probes to a CI job running inside pinned engine images.

## D. Harness and CI

- `bench/harness/{e2e_tool_stream.py, run_ab_docker.sh, summarize_ab.py}` are committed.
- Round 6: harness updated to support `rust` and `python` SGLang arms for the infe-kv probe.
- CI: fmt/clippy/test/conformance jobs all green.
- Still needed: Python-level conformance job testing shims end-to-end against a live engine.
- `infe-core` is unused; either use it or stop listing it as a dependency.

## E. Reproduce the benchmark (any Linux box with an NVIDIA GPU, Docker, nvidia-container-toolkit)

```
export INFE_BENCH_DIR=~/infe-bench; mkdir -p $INFE_BENCH_DIR/{hf,wheels,shims,results}; chmod 777 $INFE_BENCH_DIR/hf
docker pull vllm/vllm-openai:latest; docker pull lmsysorg/sglang:latest          # 0.28.0 / 0.5.18 on 2026-09-04
docker run --rm -v $PWD:/io -v $INFE_BENCH_DIR/wheels:/out -w /io/python/infe-parsers ghcr.io/pyo3/maturin:latest build --release --out /out
docker run --rm --user $(id -u):$(id -g) -v $INFE_BENCH_DIR/hf:/hf -e HF_HOME=/hf --entrypoint python3 vllm/vllm-openai:latest \
  -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
cp bench/harness/{e2e_tool_stream.py,run_ab_docker.sh} $INFE_BENCH_DIR/; cp bench/results/rtx4090-20260904/scratch/* $INFE_BENCH_DIR/shims/
for spec in "vllm stock 18001" "vllm infe 18001" "sglang stock 18000" "sglang infe 18000"; do set -- $spec
  PORT=$3 ROUNDS=3 $INFE_BENCH_DIR/run_ab_docker.sh $1 $2 <gpu-index> 8 64 256; done
(cd $INFE_BENCH_DIR/results && python3 /path/to/bench/harness/summarize_ab.py "*.json")
```
