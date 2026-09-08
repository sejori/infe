# Next-wave task list — handover (updated 2026-09-08)

## Where the project is

`infe-parsers` is **done and closed**: correct, drop-in on both engines, dual-published, and it does not make
either engine faster. Round 5 scored M1 honestly — criteria (1) drop-in, (2) parity and (4) published are met;
(3) measured improvement is not. Full numbers and reasoning: `docs/review-2026-09-04.md` §"Round 5".

Do not spend more rounds tuning it for speed. Parsing sits on the SSE path, not between GPU batches; the
component's real value was proving manifest -> crate -> wheel -> shim -> conformance -> A/B end to end on two
engines, which it did.

## `infe-kv`: killed at M0, and confirmed by Round 6 (2026-09-08)

**Do not build it.** M0 answered the kill criterion from SGLang's own PRs (C++ tree closed with zero end-to-end
win despite a 229x insert-finalisation microbenchmark; Rust replacement merged with no published end-to-end
numbers). **Round 6 confirmed it independently on our hardware**: the shipped Rust TreeCore is *slower* than
Python — `stream_span` +7.6 % @conc64, +9.1 % @conc256, e2e ~+11 % — while using 6 % less CPU. The
engine-version control (0.5.18 vs 0.5.19 Python) was flat and all arms had identical deltas/req, so the
comparison is clean. Full write-up, setup corrections and limitations: `docs/infe-kv-m0-findings.md` §"Round 6".

Two things to extract rather than discard:
- **Worth filing upstream.** SGLang's Rust TreeCore end-to-end benchmarking is an explicitly "planned
  follow-up"; we have a clean three-arm measurement of a 7.6-9.1 % regression with a pinned repro.
  **Filed: SGLang issue #38536** (https://github.com/sgl-project/sglang/issues/38536)
- **The ranking lesson, now twice-confirmed** (see below).

## `infe-sched`: M0 profiling kit BUILT — awaiting benchmark run

The profiling kit is complete and ready for the 4090. Full plan and kill criterion:
`docs/infe-sched-m0-plan.md`.

**What was built:**
- `shims/sglang/infe_sched_probe/` — SGLang plugin that hooks `get_next_batch_to_run`,
  `run_batch`, `process_batch_result` with AROUND timers. Loaded via PYTHONPATH, no fork.
- `bench/harness/e2e_high_admission.py` — three workload modes (mixed_lengths, short_burst,
  tool_call) that stress the scheduler in ways e2e_tool_stream.py does not
- `bench/harness/run_sched_probe.sh` — run driver: stock vs probe × 3 modes, interleaved
- `bench/harness/summarize_sched_probe.py` — results summarizer with kill-criterion verdict
- `bench/harness/sglang_sched_timer.py` — reference implementation (same code as the plugin)

**To run on the 4090:**
```bash
INFE_REPO=/path/to/infe PORT=18000 \
  bash bench/harness/run_sched_probe.sh 0 8 64 256
```

**Kill criterion (written before running):** if `get_next_batch_to_run` is <5% of per-step wall
time under any workload AND fits inside the GPU forward shadow under overlap scheduling, stop.
The scheduler is not on the critical path.

**Prior art check:** No native scheduler exists in either engine — SGLang's Rust work is
limited to the router and TreeCore. vLLM's `--scheduler-cls` seam is used for hardware
constraints (vllm-spyre), not performance. There is no prior implementation to learn from,
unlike infe-kv.

## What next — read this before picking up `infe-sched`

Two components have now been measured end-to-end and **both were flat or negative**:

| component | outcome | why |
|---|---|---|
| `infe-parsers` | parity, no win (round 5) | parsing is on the SSE path, not between GPU batches |
| `infe-kv` | not built; the native impl that ships is a regression (round 6) | radix-cache CPU is ~2 % of a decode step |

The common cause is visible in BRIEF §4's own numbers: decode dominates, and the CPU-side work around it is
single-digit percent of the step. `infe-sched` (BRIEF §6.3) sits in the same place. **The prior is that it will
also be flat.** So before any implementation:

1. **Profile first, with a written kill criterion.** The M0 pattern worked twice and cost days, not weeks.
2. **Check whether the engines already tried it.** Both times the answer was in the engine's own repo — SGLang
   had shipped a C++ *and* a Rust radix tree before we started. Search vLLM/SGLang PRs for a native scheduler
   before writing one. **Done for infe-sched: no prior native scheduler exists in either engine.**
3. **Consider re-aiming the thesis.** BRIEF §5.1's boundary rule assumed the win comes from moving a hot CPU
   loop off Python. Two rounds say the loops around decode are not hot enough for that on a small dense model.
   Testing it properly likely needs either (a) a regime where CPU genuinely binds — very large batch, many tiny
   requests, CPU-bound preprocessing, or disaggregated prefill where the scheduler runs hot — or (b) a target
   other than per-step CPU: correctness, portability or memory, which `infe-parsers` did actually deliver.

## Carried-over items (small, not blocking)

| # | Where | What |
|---|---|---|
| B7 | `crates/infe-parsers/src/types.rs` | Ids are index-derived, so every request's first call shares an id. Unique within a message (what the API requires) but collides across a conversation. Use a random, seedable generator. |
| D4b | `bench/harness/cpu_sampler.py` | Now cgroup-wide (all container PIDs), verified at 101% on a single-threaded busy loop. Still only 5-13 samples per run; drop the interval to 0.25s or lengthen runs before quoting CPU. |
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
| B7 | `types.rs::make_tool_call_id` | Random ids, not index-derived. | pending (small) |
| B8 | SGLang shim | Nameless argument fragments forwarded, not dropped. | done |

## C. Conformance that would have caught B

- Mine fixtures from `vllm/tests/tool_use/` and `tests/entrypoints/openai/tool_parsers/` and SGLang `test/srt/function_call/`.
- Fixture `expected_tool_calls[].arguments` must be set and asserted (currently `null` -> skipped).
- Promote off-GPU parity probes to a CI job running inside pinned engine images.

## D. Harness and CI

- `bench/harness/{e2e_tool_stream.py, run_ab_docker.sh, summarize_ab.py}` are committed.
- Round 6: harness updated to support `rust` and `python` SGLang arms for the infe-kv probe.
- infe-sched M0: `run_sched_probe.sh`, `e2e_high_admission.py`, `summarize_sched_probe.py`, `shims/sglang/infe_sched_probe/` added.
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

### infe-sched M0 probe (requires no wheel — the plugin is pure Python)

```
INFE_REPO=/path/to/infe PORT=18000 bash bench/harness/run_sched_probe.sh 0 8 64 256
cd $INFE_BENCH_DIR/results && python3 $INFE_REPO/bench/harness/summarize_sched_probe.py 'sched_probe_*.json'
```
