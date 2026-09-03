# 05 — Synthesis: what exists, and why there is no package manager for inference serving

Answers the two questions asked on 2026-09-03. Evidence in 01–04.

## Q1. What tooling already exists for Robotnik to build on?

Sorted by what Robotnik would *wrap* vs *build*.

### Wrap (exists, good enough)

| Need | Use | Notes |
|---|---|---|
| Engine arg schemas | vLLM `EngineArgs`/`FlexibleArgumentParser`, SGLang `ServerArgs`, TRT-LLM `LlmArgs` (Pydantic, fields tagged stable/beta/prototype) | Load from the pinned version at emit time; validation by delegation (LMI, LocalAI already do this) |
| Native config file formats to emit | vLLM `--config` YAML, SGLang `--config` YAML, TRT-LLM `--config`/`extra_llm_api_options`, llama-server presets, Ollama Modelfile | Files diff and check in; CLI strings don't |
| Model-intrinsic facts | HF `config.json`, `generation_config.json`, `chat_template.jinja`, GGUF KV, ModelPack `config` | Never re-declare these |
| Ground-truth intent→flags pairs | vLLM recipes (187 YAML, `hardware_overrides`, `min_vllm_version`, JSON export), Dynamo recipes + AIConfigurator, NVIDIA deployment guides, InferenceX daily matrix | Seed data for the mapping and for tests |
| Implicit cross-engine mappings | llm-d same-guide-three-engines overlays; GIE per-engine metric table; GPUStack dual-backend catalog; LMI `vllm_rb_properties.py` remap table; Xinference version shims; KServe `sort -V` gates | Mine as fixtures |
| Load generator | `vllm bench serve` / `serve_sla`, AIPerf, inference-perf | All OpenAI-compatible; pick one, plug others |
| Result artefact schema | llm-d Benchmark Report 0.2.1 (`cfg_id` hashes for stack + load) | Only portable schema; add a hardware block |
| Accuracy gate | lm-eval `local-completions`, sgl-eval, BFCL / tool-eval-bench; TRT-LLM's hypothesis-test thresholds | Task accuracy, not KL |
| Promotion mechanics | second InferencePool/LLMISVC/DGD + HTTPRoute weights; Argo Rollouts AnalysisTemplate, Flagger `confirm-promotion`, Kargo | Robotnik `verify` = Job exit code / HTTP 2xx |
| Deployment targets | KServe `LLMInferenceServiceConfig` baseRefs, llm-d Kustomize, Dynamo DGD, Ray `LLMConfig`, LMI `serving.properties`, GPUStack `InferenceBackend`, RamaLama `--generate kube` | Emit, don't replace |
| Adapter architecture | Renovate (117 managers, mostly extract-only), aqua (`version_overrides`, 89% of 2,108 packages), OpenRewrite/GritQL fixture-per-recipe tests, devcontainer scenario matrix | Proven shapes |

### Build (does not exist anywhere)

1. **A canonical serving schema with per-engine, per-version emitters.** 11 engines and 21 platforms surveyed: nobody ships a typed config that fans out to two engines. Every "multi-engine" layer is `enum backend` + raw passthrough (Dynamo, GPUStack, Xinference, Ray's one-value enum) or vLLM-only (KServe, KAITO, production-stack). LMI's `option.*` core is the closest and is AWS-internal.
2. **A rename/behaviour ledger per engine.** vLLM shipped ~18 minors in 12 months with removals in most (`--guided-decoding-backend` → `--structured-outputs-config.backend` @0.12; `VLLM_ATTENTION_BACKEND` env → `--attention-backend` @0.13; V0 removal 0.10→0.11); SGLang renames without aliases (`--enable-deepep-waterfill` → `--enable-waterfill` @0.5.16). Today this knowledge is shell (`sort -V` in KServe), Python shims (Xinference), Kustomize patches (llm-d) and prose (LMI breaking-changes page).
3. **A benchmark diff engine.** A/B of two reports with noise bands existed only as the Rust `vllm-bench --compare`, archived 2026-08-03.
4. **Perf-gated promotion.** All platforms reduce to HTTPRoute weights advanced by hand or timer; none gates on a benchmark or eval.
5. **A promotion record**: canonical config + emitted native config + engine version + hardware + benchmark result + verdict. vLLM `--generate-sweep`, Dynamo DGDR profiler, dstack Presets and GPUStack all *suggest* configs; none records "X on Y@Z scored S, promoted".
6. **A feature-parity matrix with pass-rates** across engines (Deno/Bun-vs-Node style). Nothing like it exists for inference.

### Closest competitors to watch

- **dstack Presets** (Aug 2026): LLM agent benchmarks across engines and exports a service YAML. Agentic search, not mapping; output is still a shell string.
- **NVIDIA Dynamo DGDR + AIConfigurator**: `backend: auto` picks an engine by simulated SLA. Solver, not translator; NVIDIA-shaped.
- **KAITO autoUpgrade**: the best version-management UX (maintenance windows, drift status), vLLM-only, Azure-shaped.
- **SageMaker LMI**: the only shipped multi-backend portable key set; no tooling around it.

## Q2. Why is there no centralised package manager for inference serving?

The evidence in 02 §3 supports a structural answer, not a "nobody got round to it" answer.

**The package isn't a package. It's a tuple.** npm works because the artifact is the same bytes on ~6 target triples and tests run on any laptop. A serving config is `(model × precision × engine × engine-version × GPU SKU × count × interconnect)` and its *contents* change per target: `tp`, `max-model-len`, `gpu-memory-utilization`, attention backend and quant format are not hardware-neutral. Every project that put config in the package was forced to add hardware as a key (NIM profiles are named `<backend>-<precision>-tp<N>`; vLLM recipes have `hardware_overrides`; Dynamo recipes are directories per SKU). Every project that wanted portability (HF Hub, GGUF, ModelPack, KitOps, Docker model-spec) left config out on purpose. ModelPack's own text says runtimes need "startup parameters" and then defines no field.

**The stable unit became the container, not the config.** Flag surfaces churn too fast to be a registry target: vLLM has 200–250 engine args across a dozen config classes and shipped 18 minors in a year; the SGLang cookbook versioned its YAML per engine release and was archived within months; NIM restructured profiles three times in two years; TGI went to maintenance. HF Endpoints and NIM both now pin the *engine image* and treat flags as ephemeral. A cross-engine schema has to be re-derived on every engine release, which is what a recipes repo does by hand.

**Verification is the scarce resource.** A config is only known-good on the GPU it ran on, so coverage is a sparse `verified` map (vLLM recipes) or vendor-funded (NIM "certified"; NIM 3.0 explicitly isn't). Without a cheap verification oracle a community registry can't accumulate trust the way npm's test suites do.

**The energy went elsewhere.** Weights are huge, so the standards effort went to blob movement (OCI/ModelPack, image volumes GA in K8s 1.36) and that boundary is now shared by Docker, Red Hat, Ant and CNCF. Kubernetes operators absorbed the config problem into their own CRD/Helm release cycles (KServe, llm-d, KAITO, Dynamo, Ray), so config is versioned with the operator, not the model, and isn't shareable across operators. WG Serving closed in Feb 2026 without a packaging standard.

**The consumer is a platform engineer with a text editor.** The de facto registry is GitHub plus an HF repo id as primary key (vLLM recipes: 187 YAMLs; Dynamo recipes; llm-d guides). Nobody has been forced to build a resolver because nothing consumes the config programmatically.

**Vendors cooperate on weights and monetise configs.** Docker/Red Hat/Ant/Jozu converged on ModelPack for the neutral part. Tuned per-hardware configs are the benchmark advantage (NIM, Baseten Engine Builder, Together), so nobody donates them.

**And config is increasingly derived, not authored.** NIM's memory-aware selector, KAITO's SKU rules and `max-model-len` binary search, AIConfigurator's simulation, vLLM recipes' `vram_minimum_gb` formula: the trend is `f(model, hardware, SLA) → config`. A static registry can't hold a function.

**Where local tools succeeded, the hardware axis had collapsed.** Ollama, LM Studio, Docker Model Runner and RamaLama are real package managers because the only question is "does it fit in one box". Robotnik's problem is the multi-GPU, multi-node, multi-engine one where nothing has won.

### What this means for the wedge

Do not build a registry of configs. Build a **resolver + translator over a versioned evidence base**: canonical intent (model, precision, parallelism strategy, memory *outcome*, features) → per-engine, per-version emitter → parse/start check → benchmark → recorded verdict keyed by the full tuple. Adopt existing keys (HF path + revision, engine semver, hardware tuple as NIM/vLLM-recipes model it). Be honest that portability is a translation with a confidence level ("verified on / estimated for / unknown"), because every honest system ended up with a per-SKU matrix. The rename ledger and the promotion record are the two artefacts nobody has, and both compound: every migration Robotnik runs adds evidence.
