# 03 — Verification and canary: benchmark, correctness, promote

Research for Robotnik (detect → plan → emit config → deploy → **verify** → diff → promote/rollback).
Date: 2026-09-03. Sources inline. Items marked **[unverified]** were not confirmed from a primary source.

Method note: GitHub metadata (archived flag, last push, stars) was pulled via `gh api` on 2026-09-03 and is
authoritative over dates quoted in third-party blog posts.

---

## 0. Headline findings

1. **Benchmark clients have converged on the same metric vocabulary** (TTFT, TPOT, ITL, E2EL, req/s, tok/s,
   goodput-under-SLO) and on OpenAI-compatible endpoints, but **not on a result schema**. The only cross-tool
   schema in the wild is llm-d's Benchmark Report (BR 0.2.1), which inference-perf now emits natively.
2. **Cross-tool numbers are not comparable by default.** inference-perf's own comparability guide:
   "Two benchmarking tools pointed at the same server with their default settings do not produce comparable
   numbers." Output-length pinning, `ignore_eos`, sampling params, open- vs closed-loop, warmup, and token
   counting (client vs server) all differ.
3. **No tool natively A/B's two endpoints.** The Rust `vllm-bench` had `--compare a.json b.json` and is now
   archived (2026-08-03). llm-d-benchmark runs two harness pods in parallel; everything else is "run twice, diff
   yourself."
4. **"Engine parity" is not an established practice.** Each engine gates its *own* releases on task accuracy
   thresholds (vLLM: lm-eval GSM8K/GPQA/AIME + BFCL; TRT-LLM: hypothesis-tested thresholds; SGLang: sgl-eval).
   Distribution-level comparison (KL on logprobs) exists only in llama.cpp (quant vs FP16, same engine) and in
   ad-hoc scripts. Nobody ships a "does vLLM 0.12 match vLLM 0.11 for this model" tool.
5. **Canary machinery for LLMs is weight-on-HTTPRoute, manually advanced.** KServe LLMInferenceService,
   GAIE InferencePool rollout, llm-d blue-green, KubeRay incremental upgrade, Together, Baseten, Anyscale all
   shift traffic; **none gates promotion on a benchmark or eval result.** The only metric-gated promoters are
   generic (Argo Rollouts AnalysisTemplate, Flagger, Kargo) and have no LLM awareness.
6. **Versioned, keyed result stores exist but are closed or single-purpose**: InferenceX (Postgres, natural key
   over model/hardware/framework/precision/ISL/OSL/concurrency), vLLM perf-eval → perf.vllm.ai (keyed by commit,
   model, hardware), GKE Inference Quickstart profiles (model, model server, server version, accelerator). None is
   a reusable library.

---

## (a) Benchmarking an LLM serving deployment

### Comparison table

| Tool | Owner / status (2026-09-03) | Endpoint scope | Load shapes | Metrics | Output | Native A/B? |
|---|---|---|---|---|---|---|
| `vllm bench serve/throughput/latency/sweep` | vLLM, in-tree, active | OpenAI-compat (`openai`, `openai-chat`, `openai-embeddings`, `openai-audio`, `vllm-rerank`, `tgi`) | request-rate, max-concurrency, burstiness; datasets: random, ShareGPT, sonnet, HF (VisionArena, MMVU, InstructCoder, AIMO, MT-Bench, GSM8K, ASR), SpecBench/SPEED-Bench, BFCL, prefix-repetition | TTFT, TPOT, ITL, E2EL (mean/median/p99, configurable percentiles), req/s, in/out tok/s, `--goodput ttft:X,tpot:Y,e2el:Z` | `--save-result` / `--output-json` / `--save-detailed` / `--append-result`; JSON | No (sweep has `plot`/`plot_pareto` with filter-by) |
| `vllm-bench` (Rust) | vLLM; **archived 2026-08-03**, folded into main repo | same as above | same datasets | same + peak concurrency, goodput | JSON, same schema as Python | **Yes**: `--compare a.json b.json` |
| GuideLLM | vllm-project (ex-Neural Magic), 1.6k★, pushed 2026-09-03 | OpenAI-compat | `synchronous`, `throughput`, `concurrent`, `constant`, `poisson`, `sweep` (sync → saturate → interpolate) | TTFT, ITL, TPOT, req/s, in/out/total tok/s, full distributions | JSON, YAML, CSV, HTML, PLOT; per-request prompts/outputs retained (reservoir `sample_size`) | No documented compare |
| AIPerf | NVIDIA ai-dynamo, 631★, pushed 2026-09-03; **replaces GenAI-Perf** (officially "phased out") | OpenAI chat/completions/embeddings/audio/images, NIM rankings, plugin arch | concurrency, request-rate, rate+max-concurrency, fixed schedule, **trace replay** (Mooncake, BurstGPT, ShareGPT, Baseten, SageMaker, Weka/TraceLab agentic), multi-turn, dedicated **warmup** | TTFT, ITL, TPOT, E2E, throughput, goodput, HTTP trace (DNS/TCP/TLS), timeslice; **scrapes server Prometheus** | JSON + CSV, pydantic "profile export" | Sweeps/grid + Bayesian adaptive search → Pareto; no two-endpoint diff |
| inference-perf | kubernetes-sigs (WG-Serving; WG concluded Feb 2026), 238★, pushed 2026-09-02 | vLLM/SGLang/TGI verified; any OpenAI-compat | constant, poisson, concurrent, multi-stage sweeps, OTel trace replay, Weka trace replay, synthetic agentic, multimodal | TTFT, TPOT, ITL, **NTPOT**, throughput, **goodput**; client vs server token counts + mismatch counter | `summary_lifecycle_metrics.json`, `stage_N_…`, `per_request_…`, `config.yaml`; **BR 0.2.1 partial per stage** | Side-by-side via llm-d-benchmark; `docs/comparability.md` maps settings to vllm bench / AIPerf |
| SGLang `bench_serving` | sgl-project, in-tree | `sglang` native + OpenAI-compat (vLLM, LMDeploy, TRT-LLM, Truss), embeddings | request-rate, max-concurrency; ShareGPT, random, generated-shared-prefix, **Mooncake trace**, agentic-trace, MMMU, SPEED-Bench | TTFT, ITL, TPOT, throughput, per-modality tokens, spec-decode accept length | JSONL appended per run; `--output-details` per-request arrays | No; no goodput flag |
| `trtllm-bench throughput/latency` | NVIDIA | **engine-specific**, offline (Python API, no HTTP) | dataset file | throughput, latency | text/JSON | No |
| HF `inference-benchmarker` | HF, Rust, 171★, last push 2026-06-09 | OpenAI-compat / TGI | sweep-style | TTFT, ITL, throughput | JSON | No |
| LLMPerf | ray-project; **archived, last push 2024-12-09** | OpenAI-compat + vendor APIs | fixed concurrency | TTFT, ITL, throughput | JSON | No |
| k6 / Locust | generic | any | any | no token metrics unless you instrument SSE yourself | — | — |
| MLPerf Inference v6.0 | MLCommons, Apr 2026 | reference impls + LoadGen | Offline/Server scenarios, fixed datasets | samples/s, tokens/s under latency bounds; accuracy constraints (closed division) | per-submission dirs + `summary_results.json` | No (leaderboard) |
| InferenceX (née InferenceMAX) | SemiAnalysis, 1.6k★, pushed 2026-09-03 | vLLM, SGLang, TRT-LLM containers | ISL/OSL × concurrency sweeps, AgentX multi-turn/1M-ctx | tput/GPU vs interactivity (tok/s/user), TTFT, TPOT, E2EL, cost | Postgres + public dashboard | Cross-engine by design (same workload, same hardware) |

Sources: vLLM bench CLI <https://docs.vllm.ai/en/latest/benchmarking/cli/>, sweeps <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>, serve_sla <https://docs.vllm.ai/en/latest/cli/bench/sweep/serve_sla/>, auto_tune <https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md>, vllm-bench <https://github.com/vllm-project/vllm-bench>, GuideLLM outputs <https://github.com/vllm-project/guidellm/blob/main/docs/guides/outputs.md>, AIPerf <https://github.com/ai-dynamo/aiperf>, GenAI-Perf deprecation <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html>, inference-perf <https://github.com/kubernetes-sigs/inference-perf>, comparability <https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/comparability.md>, reports <https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/reports.md>, BR0.2 <https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/br_v0_2.md>, WG-Serving wrap-up <https://www.cncf.io/blog/2026/02/26/kubernetes-wg-serving-concludes-following-successful-advancement-of-ai-inference-support/>, SGLang bench_serving <https://docs.sglang.io/developer_guide/bench_serving.html>, trtllm-bench <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-bench.html>, inference-benchmarker <https://github.com/huggingface/inference-benchmarker>, llmperf <https://github.com/ray-project/llmperf>, MLPerf v6.0 <https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/>, results repo <https://github.com/mlcommons/inference_results_v6.0>, InferenceX <https://github.com/SemiAnalysisAI/InferenceX>.

### Notes per tool

**vLLM bench.** Subcommands `serve`, `throughput`, `latency`, `sweep serve`, `sweep serve_sla`, `sweep plot`,
`sweep plot_pareto`, `mm-processor`. `serve_sla` searches for the max request rate / concurrency that satisfies a
JSON constraint set like `{"p99_e2el_ms": "<=500"}`, with linked variables (`max_num_seqs=max_concurrency`) —
i.e. it already does SLO-attainment search over *server × client* params. `benchmarks/auto_tune/auto_tune.sh`
grid-searches `max-num-seqs` × `max-num-batched-tokens` under E2E-latency and prefix-hit-rate constraints.
RFC #35639 (opened 2026-03-01, in progress) proposes `vllm bench eval`: one JSONL record per run with
`metadata` / `accuracy` (lm_eval) / `performance` (from Prometheus histograms) / `environment` (full
`vllm collect-env`), explicitly to accumulate across runs for regression detection.
<https://github.com/vllm-project/vllm/issues/35639>

**vLLM's own perf + accuracy CI** (blog 2026-07-16): nightly 17 model×hardware combos (DeepSeek V4, gpt-oss,
Kimi K2.5, Qwen3.5, GLM 5.1, Gemma 4, Nemotron 3 …) on H200/B200/MI300X/MI355X; perf via vllm-bench; accuracy via
lm-eval GSM8K/GPQA/AIME and BFCL; sample-level inspection dashboard; results in an internal DB behind
perf.vllm.ai (and hud.pytorch.org). Release candidates must pass three gates: full CI, perf benchmarks, accuracy
eval. A CI-analyzer bot bisects nightly failures and opens auto-revert PRs (~1.5/day, ~70% accurate). Workload
recipes are public in `vllm-project/perf-eval` (YAML per (model, hardware); results tagged by `VLLM_COMMIT`;
`NIGHTLY` flag pairs adjacent builds for the `/nightly` view).
<https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality>, <https://github.com/vllm-project/perf-eval>,
<https://github.com/vllm-project/perf-dashboard> (WIP), <https://hud.pytorch.org/benchmark/llms?repoName=vllm-project/vllm>

**GuideLLM.** `sweep` = synchronous run (floor latency) → throughput run (max RPS) → N interpolated rates.
Modal's stopwatch and llm-d-benchmark both wrap it. Console "Benchmarks Metadata" prints server details but the
docs do not enumerate engine version / hardware fields **[unverified whether `vllm` version is captured]**.
<https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference>

**AIPerf.** The most complete client: only one of the four majors with explicit warmup, real trace replay with
timestamps (Mooncake JSONL), and Prometheus co-collection. Also the engine behind NVIDIA Dynamo's online
profiler (2–4 h) vs AI Configurator offline estimate (~30 s).
<https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference>,
<https://docs.nvidia.com/aiperf/benchmark-modes/trace-replay-with-mooncake-traces>

**inference-perf / llm-d-benchmark / BR 0.2.1.** llm-d-benchmark (66★, pushed 2026-09-03) is a harness-agnostic
orchestrator (`-l inference-perf|guidellm|vllm-benchmark`) that stands up a stack, runs harness pods, collects
native output and composes a **Benchmark Report** (`benchmark-report/llmd_benchmark_report/br_v0_2_1_json_schema.json`).
BR 0.2.1 example shape:

```yaml
version: "0.2.1"
run: { uid, eid (experiment id), time: {start,end,duration}, description }
scenario:
  stack:                       # one entry per component (vllm-svc-0, epp-0, ...)
  - metadata: { cfg_id: <hash>, label, schema_version }
    native: { args: {--tensor-parallel-size: 8, --kv-transfer-config: ...}, envars: {...} }
  load:
    metadata: { cfg_id: <hash> }
    native: { <full inference-perf config> }
    standardized: { concurrency, input_seq_len:{distribution,value}, output_seq_len, prefix, multi_turn, rate_qps, tool, tool_version }
results:
  request_performance: { aggregate: { requests, latency (p0p1..p99p9), throughput } }
  observability: { components: [{ aggregate: {gpu_utilization, kv_cache_usage, running_requests,...}, time_series }] , epp_dispatch_latency, drop_rate }
  component_health: { replica_health: [{healthy, restarts, logs}] }
```

inference-perf writes only what it can vouch for (`run.uid/eid/time` + `results`) as a partial and expects a
composer to `yq`-merge the `scenario.stack` partial. This is **the closest existing thing to Robotnik's
"verified artifact keyed by config"** — note `cfg_id` hashes for both stack and load. llm-d-benchmark also has
"post-deployment validation" smoketests that check deployed pods match the scenario (resources, parallelism,
env, probes, routing, vLLM flags).
<https://github.com/llm-d/llm-d-benchmark>, <https://llm-d.ai/docs/architecture/Components/benchmark>

**SGLang.** No public nightly perf dashboard was found in docs **[unverified — may exist internally]**.
Accuracy CI uses `sgl-eval` (17★, pushed 2026-09-02): NeMo-Skills graders, GSM8K/AIME25/multichoice,
`pass@1[avg-of-16]` with confidence intervals, per-sample JSONL with provenance (model, endpoint, sampling),
presets to replay a config, partial runs flagged to prevent invalid comparisons.
<https://github.com/sgl-project/sgl-eval>, <https://docs.sglang.io/developer_guide/benchmark_and_profiling.html>

**TensorRT-LLM.** `trtllm-bench` is offline/engine-specific. For online, NVIDIA's docs point to
`trtllm-serve` + a generic client (AIPerf) **[unverified which client the docs currently recommend]**.
<https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html>

**InferenceX.** Self-hosted GitHub Actions runners per GPU SKU (`.github/configs/runners.yaml`, `runners/` launch
scripts using Docker or SLURM). Result natural key: `(workflow_run_id, config_id, benchmark_type, isl, osl, conc,
offload_mode)`; `config_id` normalises model, hardware, framework, precision, spec-decode method, disagg,
topology (tp/pp/dcp, prefill_tp/decode_*). Throughput rows carry `tput_per_gpu`, `input_tput_per_gpu`, latencies.
Stored in Postgres with `ON CONFLICT` upsert on the natural key; provenance (git metadata) immutable per
source run. Apache-2.0; you can add your own hardware. Frameworks: SGLang, vLLM, TRT-LLM; hardware H100→GB300
NVL72, MI300X→MI355X, TPU/Trainium "coming".
<https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/docs/results-and-ingestion.md>,
<https://inferencex.semianalysis.com/blog/inferencemax-open-source-inference-benchmarking>,
<https://vllm.ai/blog/2025-10-09-blackwell-inferencemax>

**GKE Inference Quickstart.** Google's internal benchmark DB exposed as `gcloud container ai profiles list` /
`... benchmarks list`: rows keyed by model, model server, **model server version**, accelerator, instance type;
metrics NTPOT, TTFT, output tok/s, cost per M input/output tokens at several pricing models; single replica,
saturating, ISL median 108 / OSL median 132; built on inference-perf. Read-only; you cannot contribute rows.
<https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/inference-quickstart>

**Modal LLM Engineer's Almanac / stopwatch.** Provisions vLLM/SGLang/TRT-LLM on Modal, drives with GuideLLM
(throughput / synchronous / constant), saves `results.json` locally; "three nineties" heuristic (p90 TTFT +
p90 ITL × output tokens ≈ p90 TTLT).
<https://modal.com/llm-almanac/how-to-benchmark>, <https://github.com/modal-labs/stopwatch>

**Optimum-benchmark / LLM-Perf leaderboard.** Library still pushed 2026-05-26 but the leaderboard dataset viewer is
broken (schema cast error), rows shown are CPU/onnxruntime/openvino. Not relevant to GPU serving engines.
<https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard>

**AI-Hypercomputer/inference-benchmark** archived 2026-03-11 (superseded by inference-perf).

**Client-side measurement bias.** arXiv 2605.24217 models how single-process asyncio clients (GIL) inflate TTFT/TPOT at
high QPS and proposes multi-process clients + NTPOT. Relevant: Python clients (vllm bench, GuideLLM, inference-perf)
vs Rust/multiprocess (AIPerf multiprocess, old vllm-bench). <https://arxiv.org/abs/2605.24217>

### Workload shapes available today
- Fixed synthetic ISL/OSL with distributions (all tools; inference-perf `standardized` block records them).
- Dataset-driven (ShareGPT everywhere; HF datasets in vllm bench; multimodal in vllm bench / inference-perf / AIPerf).
- Shared-prefix / prefix-cache-aware synthetic (SGLang generated-shared-prefix, vllm prefix-repetition, inference-perf shared_prefix).
- Trace replay with timestamps: AIPerf (Mooncake, BurstGPT, Azure-style), inference-perf (OTel, Weka), SGLang (mooncake, agentic-trace).
- Agentic multi-turn: InferenceX AgentX, AIPerf TraceLab, inference-perf synthetic agentic.

---

## (b) Correctness / quality regression between two engine versions or configs

### Task-accuracy harnesses over an OpenAI-compatible endpoint

| Tool | Endpoint support | Notes for engine-vs-engine use |
|---|---|---|
| lm-evaluation-harness (13.9k★) | `local-completions` (needs `logprobs`+`echo` for loglikelihood/MCQ tasks such as MMLU), `local-chat-completions` (generative only) | Versioned tasks; vLLM CI uses it with per-model YAML thresholds (e.g. `Qwen3.5-35B-A3B-FP8-DEP2.yaml`, accuracy 0.86) and `tests/evals/gsm8k/gsm8k_eval.py` <https://github.com/vllm-project/vllm/blob/main/tests/evals/gsm8k/gsm8k_eval.py>, <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md> |
| lighteval (2.5k★) | in-process vLLM backend, plus TGI / HF endpoints / OpenAI / LiteLLM | HF-centric; fine for endpoint evals <https://huggingface.co/docs/lighteval/main/en/use-vllm-as-backend> |
| Inspect (UK AISI, 2.7k★) | `openai-api/…`, `vllm`, `sglang`, `ollama`, `llama-cpp-python` providers with `--model-base-url` | Rich agentic evals; **no built-in two-log diff documented** <https://inspect.aisi.org.uk/providers.html> |
| sgl-eval (17★) | any OpenAI-compat via `--base-url` | pass@1 avg-of-16 + CI; per-sample JSONL with provenance; presets <https://github.com/sgl-project/sgl-eval> |
| TRT-LLM accuracy tests | offline LLM API | Reference accuracies per (model, quant spec) in YAML; thresholds via **two-sample hypothesis testing** (α=0.05, β=0.2, σ, min detectable regression θ) rather than fixed cutoffs <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/README.md> |
| promptfoo (24.8k★) | any OpenAI-compat provider | Multiple providers side-by-side in one grid; deterministic + model-graded assertions; `--pass-rate-threshold`; non-zero exit for CI <https://developers.openai.com/cookbook/examples/evaluation/moving-from-openai-evals-to-promptfoo> |
| deepeval (18k★), ragas (15.6k★) | app-level via Python | pytest-style CI gates (deepeval); RAG metrics (ragas). Application, not engine, layer |
| tool-eval-bench (326★) | vLLM, SGLang, llama.cpp, LiteLLM | 69 core + 19 hard-mode tool-calling scenarios, SQLite results; scores absolute quality, "not designed for direct parity comparisons" but usable as a tool-call regression suite across engines <https://github.com/SeraphimSerapis/tool-eval-bench> |

### Distribution-level parity (same model, two engines/quantisations)

- **llama.cpp `llama-perplexity --kl-divergence-base <file> --kl-divergence`**: record full logits from FP16 to a
  binary file (11 GiB for Llama-2, 37 GiB for Llama-3 on WikiText-2), then compute KL, top-1 agreement, etc. for
  the quantised model. Same engine, different weights. <https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md>
- **vLLM `tests/models` `check_logprobs_close`**: vLLM's top-k logprobs must be inside HF's top-k and vice versa
  (plus greedy-text equality tests). Engine-vs-reference-implementation, per model, in vLLM's own CI. Not
  exposed as a CLI. <https://docs.vllm.ai/en/v0.9.2/contributing/model/tests.html>
- **Ad-hoc cross-engine KL**: `rawsh` gist (last active 2026-07-27) compares vLLM and SGLang against an HF
  reference on DeepSeek-R1-Distill-Qwen-1.5B, 90 prompts × 8 completions: MAE ≈ 0.015–0.017, Pearson 0.9998,
  KL ≈ 1.8e-4–2.7e-4; vLLM without inductor agreed best. <https://gist.github.com/rawsh/245b3ddd466911d744b2d1b9f409d21b>
- **`vllm-benchmark-suite` (53★)**: `--quality kl --quality-ref http://reference:8000` computes KL between a
  candidate and a reference endpoint using token-level logprobs from the OpenAI API; explicitly kept separate from
  the perf score. Small project. <https://github.com/notaDestroyer/vllm-benchmark-suite>
- **llm-compressor #2646 "cheap KLD metric"**: full-vocab logprob extraction through vLLM takes ~64 h for
  WikiText; proposed extracting pre-lm_head hidden states (~30× less data). Closed "not planned" (2026).
  <https://github.com/vllm-project/llm-compressor/issues/2646>
- **Quantisation acceptance convention**: Red Hat / llm-compressor publish ">99% recovery" vs BF16 on lm-eval
  tasks; "Give Me BF16 or Give Me Death" (500k evals over Llama-3.1) found well-tuned INT8 W8A8 within 1–3%.
  <https://arxiv.org/abs/2411.02355>, <https://github.com/vllm-project/llm-compressor>
- **Determinism as a precondition**: vLLM `VLLM_BATCH_INVARIANT=1` (beta; NVIDIA CC≥8.0, Intel XPU) makes output
  independent of batch composition, at a perf cost; from Thinking Machines' batch-invariant kernels
  (1,000 runs → bitwise-identical). Without it, greedy-output diffing across two engines is noisy by construction.
  <https://docs.vllm.ai/en/latest/features/batch_invariance/>, <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>

### Cautionary examples of engine-version drift
- **vLLM V0 → V1 (ServiceNow, RL training)**: discrepancy surfaced only through trainer metrics
  (clip rate, KL new/old, entropy, reward), not through direct checks. Fixes: `logprobs-mode=processed_logprobs`,
  disable prefix caching + async scheduling, fp32 final projection. Recommendation: "fix backend correctness
  first, then add corrections." <https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections>
- **FP8 FA3 on Hopper**: long-context NIAH accuracy fell 91% → 13% due to accumulation precision; fixed with
  two-level FP32 accumulation, at prefill cost. A pure config/kernel change with a catastrophic quality effect
  that a TTFT/TPOT benchmark would never show. <https://vllm.ai/blog/2026-04-22-fp8-kvcache>
- **Managed inference**: "the OpenAI-compatible API is a contract about request and response shape, not about
  weights, precision, kernel, or sampler defaults." <https://futureagi.com/blog/evaluating-fireworks-together-inference-2026/>

**Answer to "is there an established practice for engine v2 ≈ engine v1 distribution?"** No. Established
practice is (1) task-accuracy thresholds per (model, quant) inside each engine's CI, with TRT-LLM the only one
using statistical thresholds; (2) top-k logprob containment vs HF in vLLM's model tests; (3) ">99% recovery" for
quantised checkpoints. Cross-engine KL is done by individuals, is expensive at full vocab, and is limited by API
top-k logprob exposure (`logprobs`/`prompt_logprobs` availability differs per engine and per API mode).

---

## (c) Canary / promotion of an inference config in production

| System | Mechanism | Promotion | Metric-gated? | LLM-aware? | Source |
|---|---|---|---|---|---|
| KServe `InferenceService` (predictive) | `canaryTrafficPercent` between last-good and new revision | manual (set to 100) | no | no | <https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary> |
| KServe `LLMInferenceService` (v0.16+, prod-ready v0.17 on llm-d + GIE 1.3.0; RawDeployment canary in v0.20) | `spec.router.route.{group,weight}` → weighted `backendRefs` on each member's HTTPRoute; oldest route wins | manual patches; promote by `serving.kserve.io/stop=true` on old; group members must share model name + LoRA set | no | routing via llm-d EPP, but rollout is plain weights | <https://github.com/kserve/website/blob/main/docs/model-serving/generative-inference/llmisvc/canary-rollout.md>, <https://kserve.github.io/website/blog/kserve-0.17-release> |
| Gateway API Inference Extension "InferencePool Rollout" | second InferencePool + HTTPRoute `backendRefs.weight` (e.g. 90/10 → 100/0); use cases: node/accelerator, base model, **model-server framework version** | manual `kubectl edit`; `helm uninstall` old pool | no | pools are inference-aware; rollout logic isn't | release-1.3 `site-src/guides/inferencepool-rollout.md` (404 on main site 2026-09-03) <https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/release-1.3/site-src/guides/inferencepool-rollout.md> |
| llm-d rollouts | Rolling (single pool, random exposure), Blue-Green (two pools + HTTPRoute 1→5→10→50→100), LoRA via `InferenceModelRewrite` | manual | no | notes long-running requests can't migrate and cold pods need warmup | <https://llm-d.ai/docs/dev/operations/rollouts> |
| Argo Rollouts + Gateway API plugin | patches HTTPRoute weights; `AnalysisTemplate` providers Prometheus/Datadog/Web/**Job** (exit code 0/1) → Successful/Failed/Inconclusive → continue/abort/pause | automatic or with pause steps | **yes** (generic) | no; InferencePool backendRef support **[unverified — plugin resolves backendRefs by Service name]** | <https://argo-rollouts.readthedocs.io/en/stable/analysis/job/>, <https://rollouts-plugin-trafficrouter-gatewayapi.readthedocs.io/> |
| Kargo | Stage `verification` reuses Argo AnalysisTemplates; freight not promotable downstream until verified | automatic | **yes** (generic) | no | <https://docs.kargo.io/user-guide/how-to-guides/verification> |
| Flagger | Prometheus checks + 8 webhooks (`confirm-rollout`, `pre-rollout`, `rollout`, `confirm-traffic-increase`, `confirm-promotion`, `post-rollout`, `rollback`, `event`); bundled loadtester (hey/ghz/k6/custom); 2xx = advance | automatic | **yes** (generic) | no; Gateway API HTTPRoute supported, InferencePool **[unverified/unlikely]** | <https://docs.flagger.app/usage/webhooks> |
| Iter8 | SLO validation / A/B/n experiments | — | yes | no | **dormant: last push 2024-08-01** |
| Seldon Core 2 Experiments | candidates with weights; `mirror` (shadow, responses discarded); sticky `x-seldon-route` | manual | no | no | <https://docs.seldon.ai/seldon-core-2/user-guide/experiment> |
| KubeRay `NewClusterWithIncrementalUpgrade` | new RayCluster; HTTPRoute weights stepped by `stepSizePercent` every `intervalSeconds`; `maxSurgePercent` capacity | time-based, automatic; rollback by reverting spec (KubeRay ≥1.7.0, feature gate) | no | designed for LLM-scale clusters | <https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice-incremental-upgrade.html> |
| Anyscale Services | new cluster per version; gradual shift by default; `--canary-percent` to pause; `--max-surge-percent`; multi-version weighted split (private beta) | automatic unless paused; auto-rollback on unhealthy canary | health only | no | <https://docs.anyscale.com/services/update> |
| NVIDIA Dynamo operator | rolling update triggered by `spec.restart.id`, `Sequential`/`Parallel` order; old/new worker DCDs coexist with hash labels; **single-node non-Grove only**; no traffic split; rollback not addressed | n/a | no | yes (DGD) | <https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/rolling-update> |
| Baseten canary (May 2025) | traffic ramps in 10 equal steps over a user-chosen window; "hit cancel" to revert | manual | no | platform-level | <https://www.baseten.co/blog/canary-deployments-on-baseten/> |
| Together AI (blog 2026-08-17) | multiple deployments per endpoint; A/B = control + ≤20 variants with fixed % (95/5 → 80/20 → 50/50); shadow traffic; canary / rolling / blue-green / auto-rollback | manual blue-green then delete experiment | platform metrics only; "deliberately doesn't measure quality", logs serving deployment in metadata for your analytics | platform-level | <https://www.together.ai/blog/a-b-test-models-in-production>, <https://www.together.ai/dedicated-model-inference> |

**Perf-gated promotion:** no LLM-specific implementation was found. The nearest things are
(i) vLLM's release process — RC must pass CI + perf + accuracy gates, but that gates the *engine release*, not
your deployment (<https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality>);
(ii) Dynamo `DynamoGraphDeploymentRequest` with `autoApply: true` — profiles (AIPerf online or AI Configurator
offline) to pick prefill/decode TP for a TTFT/ITL SLA and then generates and applies the DGD; this is
plan→deploy, not verify→promote (<https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/profiler/overview.md>);
(iii) generic Argo Rollouts Job-analysis / Flagger webhook / Kargo verification, where you would wrap a
benchmark or eval as a Job that exits 0/1.

Fireworks/Modal engine-upgrade write-ups: nothing substantive found beyond marketing pages **[gap]**.

---

## Versioned, comparable benchmark artifacts — what exists

| Store | Key | Open? | Reusable as a library? |
|---|---|---|---|
| InferenceX (Postgres) | `(workflow_run_id, config_id, benchmark_type, isl, osl, conc, offload_mode)`; config_id ⊇ model, hardware, framework, precision, spec-decode, disagg, topology | code Apache-2.0, DB is theirs | ingestion pipeline is coupled to their app; schema is readable |
| vLLM perf-eval → perf.vllm.ai / hud.pytorch.org | vLLM commit × workload YAML (model, hardware) × client params; nightly pairs | recipes public; DB internal (ClickHouse/Databricks per blog) | no |
| GKE Inference Quickstart profiles | model, model server, **server version**, accelerator, instance type | read-only via gcloud | no |
| llm-d Benchmark Report 0.2.1 | `scenario.stack[].metadata.cfg_id` + `scenario.load.metadata.cfg_id` + `run.eid` | yes (JSON Schema + pydantic, vendored into inference-perf) | **yes** — the only portable schema |
| MLPerf `inference_results_vX.Y` repos | submitter/system/benchmark/scenario + `summary_results.json` | yes | not for continuous use |
| sgl-eval JSONL, `vllm bench eval` JSONL (RFC) | provenance per run | yes | partial |
| LLM-Perf leaderboard (optimum-benchmark) | model × backend × hardware | dataset broken | no |

---

## Implications for Robotnik

1. **Wrap, don't write, the load generator.** Standardise on one OpenAI-compatible client and treat the others as
   pluggable: AIPerf if you need warmup, trace replay and server-Prometheus capture; `vllm bench serve` /
   `serve_sla` if you want SLO-attainment search and the widest dataset set for free. Do not build a k6/Locust path.
2. **Adopt BR 0.2.1 as the artifact schema (or a strict superset).** It already has `cfg_id` hashes for stack
   and load and an experiment id; Robotnik's (model, engine, engine-version, hardware, config) key maps onto
   `scenario.stack[].native.args/envars` + a hardware block you must add. Emit partials the way inference-perf
   does and compose.
3. **Comparability is a config-normalisation problem Robotnik must own.** Pin `ignore_eos`, fixed output length,
   sampling params, open/closed loop, warmup exclusion, and token-count provenance in the plan step; refuse to
   diff two reports whose `scenario.load.standardized` blocks differ. Lift the checklist from inference-perf's
   `comparability.md`.
4. **The diff engine is the thing nobody ships.** A/B of two reports (ΔTTFT/ΔTPOT/Δgoodput with noise bands,
   Pareto shift) exists only as the archived `vllm-bench --compare`. Build it; keep it schema-driven.
5. **Correctness gate = task accuracy with statistical thresholds, not KL.** Copy TRT-LLM's hypothesis-test
   framing (α, β, σ, θ → sample size and threshold) on top of lm-eval `local-completions` / sgl-eval, plus a
   tool-calling suite (tool-eval-bench or BFCL) because tool-call formatting is where engine upgrades break
   silently. Store reference accuracies per (model, quant) like TRT-LLM's YAML.
6. **Offer distribution parity as an optional, expensive tier.** Top-k logprob containment on a fixed prompt
   set (vLLM's `check_logprobs_close` idea) over the API is cheap and engine-agnostic; full-vocab KL is not
   (64 h/WikiText through vLLM). Require `VLLM_BATCH_INVARIANT=1` / equivalent when diffing greedy text,
   otherwise flag results as noisy.
7. **Promotion should target HTTPRoute weights, not a specific platform.** KServe LLMISvc, GAIE InferencePool,
   llm-d blue-green and KubeRay all reduce to "two backends, one HTTPRoute, weights". Robotnik can emit the
   route patch and the second pool/DGD and leave the gateway to the user; add native adapters for Dynamo's
   `spec.restart.id` (no split possible) and for Together/Anyscale APIs later.
8. **Perf-gated promotion has no incumbent; plug into Argo Rollouts / Flagger / Kargo rather than replacing
   them.** Ship Robotnik `verify` as a Job with exit-code semantics and as an HTTP endpoint returning 2xx/non-2xx,
   which is exactly what AnalysisTemplate Job provider and Flagger `confirm-promotion` consume.
9. **Budget for warmup and long-request drain in the verify step** (llm-d and GAIE docs both call these out);
   record `component_health` (restarts) the way BR does so a "green" benchmark on a restarting pod is caught.
10. **Reuse public references for expectation-setting.** InferenceX (same engine, same model, same GPU) and GKE
    profiles give a sanity band for "is my tok/s/GPU plausible"; vLLM perf-eval YAML recipes give ready-made
    workloads per model. None can be a Robotnik backend, but all can seed defaults.
