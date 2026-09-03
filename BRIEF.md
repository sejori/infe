# infe — project brief

Working name: **infe** (project codename in the design docs: Robotnik). Written 2026-09-03.
Audience: the engineer or agent who will write the first code. Everything here is self-contained; the evidence
behind each claim is in `docs/research/` (01–08) and the design rationale in `docs/design/`.

## 1. Mission

Build the CPU-side data plane of an LLM inference engine as **discrete, tested, benchmarked Rust components**
that can be **swapped into vLLM and/or SGLang** without forking either engine, and **prove with published
benchmarks** that running an engine with one or all of these components improves performance without changing
outputs.

The deliverable is not the components alone. It is the components **plus the proof**.

## 2. Goal and success criterion

**Goal.** For each component, and for the full set together, a reproducible A/B report on the same model, same
hardware, same engine commit, showing the engine with the component enabled versus the stock Python path.

**Success = all of the following, per component, per engine:**

1. **Drop-in.** Enabled by an existing engine flag, env var or registry hook at engine start. No engine fork,
   no patch to engine source. Stock behaviour is one flag away.
2. **Parity.** Same outputs as the stock path on the conformance suite (bit-identical where the engine is run in
   deterministic mode; token-identical greedy otherwise) and no accuracy regression on a small lm-eval slice.
3. **Measured improvement.** A statistically defended improvement on at least one primary metric at at least one
   named operating point (see §7), with the raw reports published alongside.
4. **Published as both a crate and a wheel**, with a manifest in the registry directory and a conformance suite
   that runs in CI against pinned engine versions.

If a component cannot show (3) it still ships if (1), (2) and (4) hold, but the report must say so plainly.
Negative results are results; they are what stops the project drifting into unmeasured abstraction.

## 3. Non-goals

- Rewriting vLLM or SGLang, or any part of model execution: model definitions, model runner, CUDA-graph capture,
  torch.compile, sampling kernels, attention kernels. Those stay in Python/torch. See `docs/design/02-rust-components.md` §"Why not".
- A new inference engine, a new serving framework, or a Kubernetes operator.
- Config translation between engines and versions (the earlier "package manager" track in
  `docs/design/00-thesis.md` §1–8). It is a separate project; this brief does not depend on it.
- Runtime hot-swap of a component inside a live process. "Swap" means: select at engine start via flag; A/B means
  two deployments or two runs, never a live switch.
- Kernels. FlashInfer, sgl-kernel, the HF Kernel Hub and the engines' own registries already cover that layer.

## 4. Why this, in five facts

Measured from both repos on 2026-09-03 (`docs/research/06`, `07`):

| Fact | Number |
|---|---|
| Largest single files | vLLM `gpu_model_runner.py` 7,738 lines; SGLang `scheduler.py` 5,702; SGLang `schedule_batch.py` 3,662; vLLM `scheduler.py` 3,123 |
| Commit rate into those files | 35–85 commits/month each, sustained for 18 months |
| Duplicated CPU-side subsystems | scheduler, KV manager/prefix cache, model runner, sampler, API server, tool parsers (52 files vs 43), reasoning parsers, LoRA, multimodal, spec-decode |
| Interfaces that have *cooled* (safe to target) | vLLM KV connector ABC: 33 → 13 → 6 commits per half-year; vLLM scheduler ABC: ~6 per half |
| Interfaces that are *accelerating* (avoid) | vLLM attention backend ABC: 14 → 32 → 41 → 21 in two months; SGLang attention ABC re-accelerating |

And the structural argument (`docs/design/02-rust-components.md`): the reason these files are monolithic is that
in Python every abstraction on the CPU-side hot path costs tokens/sec. Rust removes that cost, which is why every
successful decomposition of this layer so far (HF tokenizers, llguidance, sglang-router, NVIDIA Dynamo runtime,
NVIDIA kvbm) is a Rust core with Python bindings. This project makes that pattern deliberate and measured.

Convergence context (`docs/research/08`): vLLM (PyTorch Foundation; Inferact, $150M) and SGLang (LMSYS; RadixArk,
$100M) are converging feature-for-feature and will not merge. Neither will build a neutral component layer.
Both consume Rust-built dependencies already.

## 5. Architecture

### 5.1 The boundary rule (non-negotiable)

**One call per engine step, arrays in, arrays out.** The PyO3 crossing costs GIL acquisition and object conversion.
A component pays for itself only if the whole hot loop lives on the Rust side. Therefore:

- Contracts are step-granular: "schedule this step", "allocate/free/match for this batch", "consume this token
  chunk", never per-request or per-token Python calls.
- Data crosses as DLPack / Arrow / numpy views over pre-allocated buffers, not Python objects.
- Rust releases the GIL for the duration of the call.
- The crossing cost itself is a benchmarked number (M0) and every contract is checked against it.

### 5.2 Anatomy of a component

```
registry/<component>/manifest.yaml     contract version, capabilities, supported engine versions, conformance ref
crates/infe-<component>/               Rust core; no Python; criterion benches; published to crates.io
python/infe_<component>/               PyO3 bindings via maturin (abi3); published to PyPI
shims/vllm/infe_<component>/           thin adapter registered through vLLM's existing seam
shims/sglang/infe_<component>/         thin adapter registered through SGLang's existing seam (or documented gap)
conformance/<component>/               fixtures mined from both engines' own tests; both shims must pass
bench/<component>/                     A/B scenarios, expected operating points, result reports
```

The shim is the only engine-version-specific code. It is small, regenerated per engine release, and pinned to an
engine version range in the manifest. When an engine restructures (vLLM deleted all V0 attention backends in one
commit; SGLang relocated 93 files in one week — `docs/research/07` §4), the crate and conformance suite survive;
only the shim is redone.

### 5.3 Shared crate

`crates/infe-core`: step contract traits, DLPack/Arrow buffer helpers, error types, tracing hooks, a
`StepTimer` that every component uses so that CPU-side step time is reported identically.

## 6. Components, in priority order

Ranked by contract stability × duplication × independence from the GPU. Do them in this order; do not start #3
before #2's A/B report exists.

### 6.1 `infe-parsers` — tool-call and reasoning stream parsers

- **Scope.** The union of both engines' tool-call parsers (vLLM 52 files, SGLang 43) and reasoning parsers, behind
  one streaming interface: feed token-text chunks, receive structured deltas (OpenAI tool_call / reasoning_content
  deltas). Per-model dialects (Hermes, Llama-3.x JSON, Mistral, Qwen, DeepSeek, GLM, Kimi, gpt-oss Harmony, …) are
  data-driven where possible, code where not.
- **Engine seams.** vLLM: `--tool-call-parser` + `--tool-parser-plugin <file>` and `--reasoning-parser` +
  `--reasoning-parser-plugin <file>` `[verify exact flag names at M0]`. SGLang: `--tool-call-parser`,
  `--reasoning-parser`; register via the 2026 `sglang.srt.plugins` hook registry `[verify]`.
- **Hypothesis.** Small performance gain (parsers sit on the streaming path in the API-server process, not the GPU
  loop): lower ITL p99 jitter and API-server CPU time at high concurrency with tool-heavy workloads. Main value is
  parity, and proving the manifest → crate → wheel → shim → conformance → bench pipeline on a piece nobody will
  fight over.
- **Measure.** ITL p50/p99 and API-server CPU utilisation at 64/256/1024 concurrent streams on a tool-call-heavy
  dataset; conformance pass-rate per dialect per engine (the "parity matrix").
- **Accept.** 100% pass on fixtures mined from both engines' parser tests; both shims registered without patching;
  published parity matrix.

### 6.2 `infe-kv` — KV block manager and prefix cache

- **Scope.** Block allocator (paged), prefix cache with both strategies the engines use (vLLM hash-block, SGLang
  radix tree), eviction policies, reference counting, and the bookkeeping for prefill/decode disaggregation and
  offload tiers. Pure data structures over block ids; never touches GPU memory itself.
- **Engine seams.** vLLM: the KV connector seam (`--kv-transfer-config` with `kv_connector` +
  `kv_connector_module_path`; NVIDIA kvbm plugs in at `kvbm.vllm_integration.connector` — this is the proven path
  and the interface that has cooled). **Honest caveat:** the connector seam is for transfer/offload/cross-request
  reuse; replacing vLLM's *in-engine* allocator (`v1/core/kv_cache_manager.py`) is not flag-selectable today and
  needs an upstream RFC. So M2 ships as a connector first (prefix reuse across tiers, offload), and the allocator
  replacement is an upstream proposal with our benchmark as the argument. SGLang: HiCache / disaggregation
  interface (`disaggregation/base/conn.py`, rising churn) plus `radix_cache.py` has no seam; expect a documented gap
  or a plugin-hook path `[verify]`.
- **Hypothesis.** At high request arrival rates and heavy prefix sharing (multi-turn, RAG, agents), per-step
  allocation/free/match time on the Python side is a measurable fraction of step time, and the GPU sits idle for it.
  Rust reduces CPU step overhead and the GPU-idle fraction, which shows up as throughput at small-to-medium batch
  and TTFT under load.
- **Measure.** Scheduler/step CPU time (via `StepTimer` and the engines' own step metrics), GPU-idle fraction
  (torch profiler / nsys), TTFT p50/p99 and goodput at 1×, 2×, 4× the saturating arrival rate, prefix hit-rate
  parity with stock.
- **Accept.** Parity on prefix-hit decisions on the conformance replay; measured reduction in step CPU time; the
  A/B report; an RFC draft for the allocator seam in vLLM.

### 6.3 `infe-sched` — scheduler policy

- **Scope.** Admission, batching (chunked prefill budgets), preemption, priority, spec-decode-aware token budgets.
  Operates on request metadata + `infe-kv` state; produces the step plan as arrays.
- **Engine seams.** vLLM: `--scheduler-cls` (SchedulerConfig, used by out-of-tree schedulers; ABC churn ~6/half).
  SGLang: no seam; 5,702-line scheduler; shim = plugin-hook or documented gap `[verify]`.
- **Hypothesis.** Same as 6.2, larger effect, higher coupling. Expect to lag new engine features (new spec-decode
  modes, new disaggregation shapes); the manifest must say which engine features the component does not support.
- **Measure/accept.** As 6.2, plus scheduling-decision parity on replayed traces.

### 6.4 `infe-attn-meta` — attention metadata builders (last, maybe never)

Builds block tables, cu_seqlens, slot mappings for the attention kernels. Couples to CUDA-graph capture protocol
and to interfaces that are accelerating. Only attempt after 6.1–6.3 have published reports and if the profiler
shows metadata construction as a top CPU cost.

### Already Rust/C++ — register, don't rewrite

tokenizers (Rust), llguidance (Rust) / xgrammar (C++), NIXL / Mooncake (C++), sglang-router (Rust), kvbm (Rust).
The registry should list them with manifests so the "all modules" run can enable them consistently.

## 7. The proof: benchmark methodology

This section is the product. Treat it as a spec.

### 7.1 Protocol

- **Fixed:** model (+ revision, precision), engine (+ exact commit), hardware (SKU × count × interconnect, driver,
  CUDA), container image, all engine flags except the one that enables the component.
- **Two arms:** `stock` and `infe`. Same image, same launch, one flag differs. Both arms restarted fresh per run.
- **Runs:** ≥5 per arm per operating point, interleaved (stock, infe, stock, infe, …) to defeat thermal/clock drift.
  Warmup excluded. Report median and IQR; claim improvement only when the bootstrap CI of the difference excludes zero.
- **Workload:** fixed `ignore_eos`, fixed output length, fixed sampling params, open-loop arrival at named rates,
  datasets: ShareGPT-style chat, long-prefix multi-turn (for `infe-kv`), tool-call-heavy (for `infe-parsers`),
  and trace replay where available. Lift the comparability checklist from kubernetes-sigs/inference-perf
  `comparability.md`.
- **Operating points:** at minimum three arrival rates (below, at, above saturation) × two batch regimes
  (latency-bound small batch; throughput-bound large batch). Improvements are expected in the first regime;
  the second must show no regression.
- **Load generator:** one of `vllm bench serve`, inference-perf, or AIPerf, pinned; never a hand-rolled client.
- **Artefact:** llm-d Benchmark Report 0.2.1 schema plus a hardware block and an `infe` block (component,
  version, manifest hash). Raw JSON committed under `bench/results/`. A `bench/diff` tool renders the A/B table.

### 7.2 Metrics

| Metric | Why |
|---|---|
| CPU step time (p50/p99) via `StepTimer` and engine step metrics | the direct claim |
| GPU-idle fraction per step (torch profiler / nsys) | the mechanism |
| TTFT, TPOT/ITL p50/p99 | user-visible latency |
| Throughput (tok/s) and goodput under SLO | capacity |
| API-server CPU % (parsers) | process-level cost |
| Prefix hit-rate, preemption count, batch-size histogram | parity of decisions, not just outputs |

### 7.3 Parity (correctness) gate — runs before any perf number is reported

- Deterministic mode where the engine supports it (`VLLM_BATCH_INVARIANT=1` or equivalent) → bit-identical outputs.
- Otherwise greedy, fixed seed → token-identical outputs on the conformance prompt set.
- Decision parity: replayed request traces must produce the same scheduling / allocation / parse decisions as stock,
  or a documented, intentional difference.
- Small lm-eval slice (e.g. GSM8K subset + a tool-calling suite) → no regression beyond noise.

### 7.4 Hardware

Single-GPU is sufficient for `infe-parsers`. One 8-GPU H100/H200-class node is sufficient for `infe-kv` and
`infe-sched` A/Bs. Multi-node disaggregation is out of scope for the first reports.

## 8. Repository layout

```
infe/
  BRIEF.md                      this file
  docs/design/                  thesis, package model, rust-components rationale
  docs/research/                01–08 evidence (engines, packaging, verification, orchestration, internals, churn, governance)
  registry/                     one dir per component: manifest.yaml, README
  crates/
    infe-core/                  contracts, buffers, StepTimer
    infe-parsers/
    infe-kv/
    infe-sched/
  python/
    infe_parsers/ infe_kv/ infe_sched/     maturin projects, thin
  shims/
    vllm/                       one package per component, pinned to a vLLM version range
    sglang/
  conformance/                  fixtures + runners; mined from engine tests; engine-agnostic format
  bench/
    harness/                    A/B runner, report schema, diff tool
    scenarios/                  named operating points and datasets
    results/                    committed raw reports
  .github/workflows/            rust CI (fmt/clippy/test/criterion), maturin wheels matrix, conformance vs pinned engines
```

## 9. Milestones

- **M0 — Foundations (weeks 1–2).** Pin current vLLM and SGLang releases and container images. Build the A/B
  harness and run *stock vs stock* to establish the noise floor. Microbenchmark the PyO3 crossing cost for
  step-granular calls with DLPack buffers; publish the number. `infe-core` with the step contract and `StepTimer`.
  Manifest schema v0. Verify every `[verify]` item in this brief and update it.
- **M1 — `infe-parsers` (weeks 3–6).** Crate, wheel, both shims, conformance mined from both engines, parity matrix,
  streaming A/B report. Pipeline proven end to end.
- **M2 — `infe-kv` (weeks 6–12).** Connector-seam integration in vLLM, SGLang path or documented gap, decision-parity
  replay, A/B report at the three arrival rates, allocator-seam RFC draft for vLLM.
- **M3 — `infe-sched` (weeks 12–18).** vLLM via `--scheduler-cls`, SGLang path or gap, trace-parity, A/B report.
- **M4 — All-modules report (weeks 18–20).** Engine with every component enabled vs stock, both engines, published
  with raw data. Upstream RFCs opened with the reports attached.

## 10. Engineering standards

- Rust stable, edition 2024; `cargo fmt`, `clippy -D warnings`, `cargo test`, criterion benches per crate.
- PyO3 + maturin, abi3 wheels, `manylinux_2_28` + macOS arm64 for dev; Linux x86_64 is the only supported prod target.
- No per-token or per-request Python calls in any hot path (§5.1). Reviewers reject PRs that add one.
- Every crate has a `README` with: contract version, what it replaces in each engine, and its last A/B result.
- Every shim pins an engine version range in `manifest.yaml`; CI runs conformance against the pinned engine
  containers, nightly against engine `main`, and reports drift rather than failing the build.
- Bench results are committed, never summarised without the raw JSON.
- Public repo hygiene: no customer data, prompts or org names in fixtures or reports.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Improvement unmeasurable at GPU-bound operating points | Report it. The claim is CPU-side; pick latency-bound points for the headline and show no regression elsewhere |
| PyO3 crossing eats the gain | M0 measures it first; contracts are step-granular by rule |
| Engine restructures break shims | Shims are small and pinned; crate + conformance survive; nightly drift job |
| The allocator/scheduler seam does not exist in SGLang | Ship the vLLM path, document the SGLang gap, use the report to argue for the seam upstream |
| Components lag engine features (new spec-decode modes etc.) | Manifest declares unsupported features; engine falls back to stock; never silently degrade |
| Scope creep into kernels/runner | §3 non-goals; 6.4 gated on profiler evidence |

## 12. Open questions for M0

1. Exact current flag names for vLLM tool/reasoning parser plugins and SGLang's plugin hook registry.
2. Whether SGLang's 2026 plugin system offers any hook into `radix_cache` / scheduler, or only kernels/attention.
3. Whether vLLM CI builds Rust today (SGLang does, in-tree for the router).
4. Which load generator: `vllm bench serve` (widest datasets, SLO sweep) vs AIPerf (warmup, trace replay,
   server-Prometheus capture). Pick one, wrap it.
5. Deterministic-mode availability per engine at the pinned versions, for the bit-identical parity gate.

## 13. Reference facts (verified 2026-09-03)

- Engine seams, vLLM: entry-point groups `vllm.general_plugins`, `vllm.platform_plugins`, io_processor, stat_logger,
  endpoint, logits_processors; class-path strings for `--scheduler-cls`, attention `CUSTOM`, executor backend;
  `--kv-transfer-config {kv_connector, kv_connector_module_path}`; `ModelRegistry.register_model`.
- Engine seams, SGLang: `sglang.srt.platforms`, `sglang.srt.plugins` hook registry (2026), `sglang.kernels`
  registry (RFC #29630, July 2026; `KernelSpec`, `FormatSignature`, `CapabilityRequirement`, `PlatformInfo`),
  `SGLANG_EXTERNAL_MODEL_PACKAGE`; no scheduler seam.
- Shared pinned deps at both engines' `main`: torch 2.13.0, flashinfer 0.6.18, tilelang 0.1.12, quack-kernels 0.6.4,
  cutlass-dsl 4.6.2; both use xgrammar/llguidance, compressed-tensors, transformers v5, Mooncake/NIXL/LMCache.
- Existing Rust prior art to study before writing a line: NVIDIA `kvbm` (`pip install kvbm`), Dynamo runtime,
  sglang-router, HF tokenizers, llguidance, pydantic-core (for PyO3 granularity lessons).
- Benchmark schema: llm-d Benchmark Report 0.2.1 (`cfg_id` hashes for stack and load). Comparability checklist:
  kubernetes-sigs/inference-perf `comparability.md`.
