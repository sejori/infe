# 04 — Orchestration layers above the engine, and cross-domain prior art

Compiled 2026-09-03 from six sub-surveys (llm-d/KServe; KAITO/Ray Serve; vLLM production-stack/GIE; managed
PaaS layers; dependency/version managers; codemod/migration tooling). Facts verified against GitHub API,
raw source and live docs on the day unless flagged `[unverified]`. Source URLs inline.

Legend for "config flow": **T** = typed schema, **PT** = raw passthrough (args/env/command), **IMG** = engine
version lives only in a container image tag.

---

## Part A — Orchestration / deployment layers

### A.0 One-page summary

| Layer | Deployment unit | Abstracts engine? | Engine version = ? | Upgrade as first-class op? | Config flow |
|---|---|---|---|---|---|
| NVIDIA Dynamo | DGD / DCD CRDs; DGDR (`backend: auto\|vllm\|sglang\|trtllm`, `sla`, `workload`) | Enum + profiler search, not mapping | IMG | Rolling restart (`spec.restart.id`); no traffic split | PT (args, SGLang YAML via ConfigMap) |
| llm-d v0.9 | No CRD any more: Kustomize overlays of Deployment/LWS/DisaggregatedSet + router Helm | No — one overlay dir per engine; label `llm-d.ai/engine-type` | IMG, pinned once per engine in Kustomize Components | "change the image tag"; blue/green via HTTPRoute | PT (`command`/`args`/`env`) |
| KServe v0.20 | `LLMInferenceService` v1alpha2 (+ `LLMInferenceServiceConfig` baseRefs); legacy `ServingRuntime` | Effectively vLLM-only; ~6 typed parallelism ints | IMG (`ghcr.io/llm-d/llm-d-cuda:v0.9.0`) | `rolloutStrategy{maxSurge,maxUnavailable}` + route `group/weight` canary | T subset → Go-templated bash → `VLLM_ADDITIONAL_ARGS` + `$@` |
| KAITO v0.12 | `Workspace` (`kaito.sh/v1beta1`), `InferenceSet`, `MultiRoleInference` | vLLM \| transformers only | Controller-embedded `base_images.yaml` (not in Workspace) | **Yes**: `InferenceSet.autoUpgrade{InPlace\|Surge, maintenanceWindow}`, drift counter | Hardware-derived flags (TP/PP/DP from SKU, `max-model-len` binary-searched) + PT ConfigMap |
| Ray Serve LLM (Ray 2.58) | `LLMConfig` Pydantic; `RayService.serveConfigV2` on K8s | `llm_engine` enum has ONE value (vLLM); SGLang via private `server_cls`, different kwarg vocabulary | `ray[llm]` pin → Ray image tag | KubeRay `upgradeStrategy: NewCluster \| NewClusterWithIncrementalUpgrade` | T Ray side + untyped `engine_kwargs` dict |
| vLLM production-stack 0.1.12 | Helm `servingEngineSpec.modelSpec[]` or `VLLMRuntime` CRD v1alpha1 | No (vLLM hardcoded) | IMG (`tag: latest` default) | RollingUpdate default; no hooks | T subset (`vllmConfig.*`) → flags + `extraArgs[]` + `env[]`; Helm and CRD schemas disagree |
| Gateway API Inference Ext v1.6 | `InferencePool` (GA v1); EPP/`InferenceObjective` moved to llm-d-router | Agnostic by *protocol* (OpenAI API + 3 gauges); metric-name table per engine | Not owned | Documented as "second pool + HTTPRoute weights" | n/a (no engine args at all) |
| SageMaker LMI (DJL) | `serving.properties` / `OPTION_*` env | **Yes — shared `option.*` core across vLLM & TRT-LLM** but different images | LMI number ≈ vLLM pin (v15=0.8.4 … v28=0.26) | None; prose "breaking changes" page | T core + validated PT (parsed by bundled vLLM's own argparser) |
| SageMaker vLLM DLC | `SM_VLLM_*` env → `--flag` | No | IMG | None | PT, zero validation |
| SkyServe | task YAML + `service:` block | No (`setup:`/`run:` shell) | Whatever `pip install`/image says | `sky serve update --mode rolling\|blue_green` | PT shell |
| dstack | `type: service` YAML | No (`image` + `commands`) | IMG | Rolling on property diff; **Presets** = LLM agent benchmarks across engines and exports YAML | PT shell |
| Modal | Python `@app.server` / `@app.cls` | No | pip pin in image | `modal deploy --strategy`, `app rollback` | user code |
| Baseten Truss | `config.yaml` | TRT-LLM typed (`trt_llm_config.py`), vLLM/SGLang via `docker_server` | tied to truss release | environments + promotion | T (single engine) |
| Anyscale | `service.yaml` + Ray `LLMConfig` | No | image tag | canary %, rollback `[unverified]` | PT `engine_kwargs` |
| HF Inference Endpoints | endpoint object, `custom_image` | Engine selector (TGI/vLLM/SGLang/llama.cpp); TGI in maintenance since 2025-12 | image URL | none; engine not changeable in place | per-engine UI fields + env |
| Vertex Model Garden | `serving_container_args` on `Model.upload` | container URI = engine (vLLM, vLLM-optimized, Hex-LLM TPU) | date-stamped `_RCxx` tags | redeploy | PT |
| Azure AI Foundry | AzureML online-deployment YAML | HF path exposes 6 runtimes incl. vLLM/SGLang/TEI | curated env | platform | PT |
| RHOAI 3.x | KServe ServingRuntime / LLMISVC | vLLM CUDA/ROCm/Gaudi/Spyre + TGIS/OpenVINO | pinned per RHOAI release (3.3.x = vLLM 0.13.0) | platform upgrade | PT args |
| GPUStack 2.2 | `Model{backend, backend_version, backend_parameters[]}` + `InferenceBackend{version_configs}` | **Yes — backend & backend_version first-class**, but native args; catalog has separate spec per backend | per-version runner image | edits don't touch running instances | PT + a few derived flags |
| Harbor | `.env` of `HARBOR_<SVC>_*` + compose fragments | No — namespace per engine | `HARBOR_VLLM_VERSION` | rebuild | PT strings |
| Xinference 3.3 | `xinference launch --model-engine …` + model-family JSON | **Yes — engine chosen at launch; compatibility computed** (`match_json`) | per-launch virtualenv pin; version-gated shims in `vllm/core.py` | none | T core + PT kwargs |

### A.1 Kubernetes-native stacks (detail)

**llm-d** — `ModelService` CRD is archived (2025-07); replaced first by a Helm chart (`llm-d-modelservice` 0.4.16,
with `modelCommand: vllmServe` synthesising `vllm serve` from `parallelism{tensor,data}` then appending free-form
`args`) and since v0.7 by plain Kustomize manifests. Base deployment is literally `image: REPLACE_MODEL_SERVER_IMAGE,
command: [], args: []`. Engine pins live in Kustomize Components (`gpu-vllm/release` → `vllm/vllm-openai:v0.26.0`,
`gpu-sglang/release` → `lmsysorg/sglang:v0.5.16`). Version-coupled flags leak in as patches (`env USER=llm-d`
"required with vLLM v0.20.0+ due to torch 2.11"; TRT-LLM pinned `1.3.0rc23` because router metrics need ≥1.3.0rc12).
The same guide (pd-disaggregation) exists for vLLM/SGLang/TRT-LLM with equivalent intent — an implicit cross-engine
mapping worth mining: `--kv-transfer-config NixlConnector` ≙ `--disaggregation-mode/--disaggregation-transfer-backend=nixl`
≙ TRT-LLM `extra_llm_api_options`. `llm-d-inference-scheduler` was renamed `llm-d-router` (v0.10.0, 2026-08-17).
https://github.com/llm-d/llm-d/releases · https://github.com/llm-d/llm-d-model-service ·
https://github.com/llm-d-incubation/llm-d-modelservice · https://github.com/llm-d/llm-d-router

**KServe** — `LLMInferenceService` production since v0.17 (2026-03); v0.20 (2026-08-06) serves v1alpha1+v1alpha2.
Topology inferred from shape (`worker` → LeaderWorkerSet; `prefill` → P/D; else Deployment). Controller injects
well-known `LLMInferenceServiceConfig` templates, then merges `baseRefs` in order, then the spec — a reusable
"profile inheritance" model. Every template is a bash wrapper around `exec vllm serve /mnt/models` that does
**runtime version detection inside the container** (`sort -V` on `$VLLM_VERSION` to gate `--shutdown-timeout` ≥0.18
and `OffloadingConnector` ≥0.22). That is exactly the "flag introduced at version X" data a porting tool needs, encoded
as shell. Legacy `ServingRuntime` selection: explicit `runtime` → autoSelect by `supportedModelFormats{name,version}`
→ `priority`. https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview ·
https://raw.githubusercontent.com/kserve/kserve/master/config/llmisvcconfig/config-llm-template.yaml

**KAITO** — the most interesting upgrade story. vLLM version is *not* in the Workspace; it lives in a `go:embed`
`base_images.yaml` (`tag: 0.5.0, runtimeVersion: {vllm: 0.25.1}`; comment says 0.26.0 — in-repo inconsistency).
Maintainer runbook for a bump: requirements → image → ConfigMap mirror → regenerate 532-arch allowlist → re-sync
reasoning/tool-parser maps → smoke DeepGEMM/FlashInfer/LMCache "historically broken across vLLM bumps".
`InferenceSet.autoUpgrade{enabled, strategy: InPlace|Surge, maintenanceWindow{schedule, duration}}` with drift
reported in status. Controller derives TP/PP/DP from SKU (3-tier rule), dtype from compute capability, probes
`gpu-memory-utilization` and binary-searches `max-model-len`. User overrides = ConfigMap of dash-flags appended verbatim.
https://raw.githubusercontent.com/kaito-project/kaito/main/presets/workspace/models/base_images.yaml ·
https://raw.githubusercontent.com/kaito-project/kaito/main/presets/workspace/models/vllm_version_upgrade.md ·
https://raw.githubusercontent.com/kaito-project/kaito/main/docs/proposals/20260507-auto-upgrade-base-image.md

**Ray Serve LLM** — `LLMEngine` enum has a single member (`vLLM`). SGLang path uses a private `server_cls` and
`engine_kwargs` keys change vocabulary (`tensor_parallel_size` vs `tp_size`; `model_source` vs `model_path`), so
`llm_engine` is not a real switch. No validator against `AsyncEngineArgs`; errors surface at engine construction.
vLLM pinned by `ray[llm]` (`vllm[audio]==0.27.0` on master; 2.58.0 shipped 0.26.0). Engine bump = Ray bump = image change;
KubeRay decides rollout (`NewCluster` blue/green default, `NewClusterWithIncrementalUpgrade` via Gateway API).
https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html · https://github.com/ray-project/ray/issues/65386

**vLLM production-stack** — Helm `vllmConfig{v0, enablePrefixCaching, enableChunkedPrefill, maxModelLen, dtype,
tensorParallelSize, maxNumSeqs, maxLoras, gpuMemoryUtilization, runner, convert, extraArgs[]}`; CRD `VLLMRuntime`
puts `maxModelLen/dtype/maxNumSeqs` under `model` and uses `v1: bool` — the two schemas of the same project disagree.
Emits underscore forms (`--gpu_memory_utilization`) and `--no-enable-prefix-caching` whenever false. `lmcacheConfig`
injects `--kv-transfer-config '{"kv_connector":"LMCacheConnector[V1]"}'`. Default `tag: latest`.
https://raw.githubusercontent.com/vllm-project/production-stack/main/helm/values.schema.json ·
https://raw.githubusercontent.com/vllm-project/production-stack/main/operator/api/v1alpha1/vllmruntime_types.go

**Gateway API Inference Extension** — v1.6.0 (2026-08-17) moved the full EPP, Body-Based Routing, `InferenceObjective`,
`InferenceModelRewrite` and `EndpointPickerConfig` into **llm-d-router**; GIE main keeps only `InferencePool` v1,
`InferencePoolImport`, conformance and a round-robin `lwepp`. Engine-agnostic by protocol: any backend speaking OpenAI
Completions/Chat and exposing `TotalQueuedRequests`, `TotalRunningRequests`, `KVCacheUtilization` gauges. The
per-engine metric table (vLLM / SGLang `sglang:num_queue_reqs` / Triton `nv_trt_llm_*` / trtllm-serve) in
`EndpointPickerConfig.core-metrics-extractor.engineConfigs[]` is the closest thing to a cross-engine equivalence spec
that exists, and mixed engines in one pool are supported via label `inference.networking.k8s.io/engine-type`.
Rollout guide = second InferencePool + HTTPRoute weights. https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v1.6.0 ·
https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/main/docs/proposals/003-model-server-protocol/README.md

### A.2 Managed / PaaS layers (detail)

**SageMaker LMI (DJL Serving)** — best prior art for a portable core. `engine=Python` + `option.rolling_batch=vllm|trtllm`;
shared keys `model_id, tensor_parallel_degree, pipeline_parallel_degree, max_rolling_batch_size, dtype, quantize,
max_rolling_batch_prefill_tokens` honoured by both backends (different images though). Translation table lives in code
(`vllm_rb_properties.py`): renames (`tensor_parallel_degree→tensor_parallel_size`, `max_rolling_batch_size→max_num_seqs`),
aliases (`quantize|quantization`), `DTYPE_MAPPER`, `task→runner/convert`. Passthrough extras are kept only if in
`EngineArgs.__annotations__`, then **built into a CLI list and parsed by the bundled vLLM's own `FlexibleArgumentParser`**
— validation by delegation. Weaknesses: LMI v15 removed `lmi-dist`/`scheduler` backends with a prose migration note;
`option.max_output_len` silently remapped to `max_seq_len`; no tool checks that lmi16/vLLM 0.10 properties still parse
on lmi28/vLLM 0.26. LMI number is now effectively the vLLM pin (v15=0.8.4, v16=0.10.2, v17=0.11.1, v18=0.12.0, v19=0.14.0,
v20=0.15.1 … v28=0.26.0, all on DJL 0.36.0). AWS also ships a second dialect, the vLLM DLC (`SM_VLLM_*` → `--flag`, no validation).
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/deployment_guide/configurations.html ·
https://github.com/deepjavalibrary/djl-serving/blob/master/engines/python/setup/djl_python/properties_manager/vllm_rb_properties.py ·
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/announcements/breaking_changes.html ·
https://aws.github.io/deep-learning-containers/vllm/configuration/

**dstack Presets (Aug 2026)** — `type: preset` runs an LLM-driven benchmark agent that switches engines between trials
("switched from vLLM to SGLang"), stores patches, and `dstack preset export` emits a service YAML. Closest live
competitor idea to Robotnik's verify loop, but agentic search rather than mapping, and output is still a shell command.
https://dstack.ai/docs/concepts/presets/ · https://dstack.ai/blog/presets/

**Baseten Truss** — `trt_llm_config.py` is the only fully typed, Pydantic-validated engine schema among PaaS targets
(enums for model family, quantisation ×11, scheduler policy, spec-decode mode, checkpoint source) — single engine only.
https://github.com/basetenlabs/truss/blob/main/truss/base/trt_llm_config.py

**Xinference** — engine chosen at launch; `check_format_with_engine` + per-engine `match_json` compute which
(engine, format, quant) combos are valid for a model spec. `vllm/core.py` has version-conditional shims
(`supports_guided = VLLM_VERSION < 1.12.0`, `GuidedDecodingParams` vs `StructuredOutputsParams`) — a ready-made
dataset of vLLM kwarg renames. https://inference.readthedocs.io/en/latest/user_guide/backends.html

**GPUStack** — `backend` + `backend_version` first-class; per-version `image_name`/`run_command` templates with
`{{model_path}} {{port}} {{gpu_count}}` placeholders; "for vLLM v0.11.1+ you must override entrypoint and command".
Catalog carries a separate `specs[]` entry per backend for the same model (humans do the translation).
https://docs.gpustack.ai/latest/user-guide/model-catalog/

**Others** (SkyServe, Modal, Anyscale, HF Endpoints, Vertex, Foundry, RHOAI, Harbor): infra typed, engine = image tag +
command string. Upgrade semantics (rolling/blue-green/rollback/history) live at the platform; none checks that the old
flags survive the new engine. HF Endpoints cannot change engine on an existing endpoint; TGI in maintenance mode since
2025-12-11 with "migrate to vLLM/SGLang". `[Together/Fireworks internals unverified — closed]`

### A.3 Cross-cutting findings (Part A)

1. **Only three layers have any engine-agnostic key set**: SageMaker LMI (`option.*` core), GPUStack (fields, native args),
   Xinference (computed compatibility). Everyone else is `enum backend` + raw passthrough, or vLLM-only.
2. **Engine version lives outside the deployment object** almost everywhere (KAITO embedded file, Ray pip pin, llm-d
   Kustomize Component, LMI container number, RHOAI release). Porting a config therefore means porting across an
   *implicit* version too.
3. **Version-coupled behaviour is encoded as shell/code, never data**: KServe `sort -V` gates, llm-d env patches,
   Xinference version shims, LMI remaps, GPUStack entrypoint override. A rename/behaviour ledger per engine would be
   new and would subsume all of these.
4. **Validation by delegation works**: LMI parses with the bundled engine's argparser; LocalAI checks `engine_args`
   against `AsyncEngineArgs`. Robotnik can load the pinned engine version's arg schema offline.
5. **Upgrade = image tag + HTTPRoute weights** everywhere; nobody gates promotion on a benchmark.
6. **Two dialects of the same engine coexist** even inside one vendor (LMI `OPTION_*` vs DLC `SM_VLLM_*`; production-stack
   Helm vs CRD; KAITO dash-flags vs Ray snake_case kwargs). Translation is trivial but real and currently manual.

---

## Part B — Prior art from other domains for detect → normalise → emit → verify

### B.1 Dependency / version managers

| System | Adapter unit | Count | Adapter size | Interface | Tests | LLM? |
|---|---|---|---|---|---|---|
| **Renovate** | manager (+ datasource + versioning) | 117 built-in + 2 custom / 82 / 54 | dockerfile ~500 LOC; npm ~6.5k (tests 1.5–3.5× source) | `defaultConfig`, `supportedDatasources`, `extractPackageFile` (most managers are ONLY this), optional `updateDependency`, `getRangeStrategy`, `updateArtifacts`, `bumpPackageVersion` | Vitest + fixtures; conformance spec iterates all managers | none |
| **Dependabot** | ecosystem (7 mandated classes) | 32 dirs, 36 CI suites | dotnet_sdk 695 LOC; npm 17.7k | `FileFetcher, FileParser, UpdateChecker, FileUpdater, MetadataFinder, Version, Requirement` + `PackageManager` w/ SUPPORTED/DEPRECATED versions | RSpec + VCR; `shared_examples_for_*` conformance; per-eco Docker image; 60 txtar smoke tests | none in core |
| **mise** | backend (Rust trait, 2–3 abstract methods) + registry TOML | 18 backends + 13 core; **985 registry TOMLs** | deno core 216 LOC; aqua backend 5.8k | `_list_remote_versions`, `install_version_`; registry entry = `backends[]` fallback list + `idiomatic_files` + `test = {cmd, expected}` | `mise test-tool` in Docker matrix on changed entries | none |
| **asdf** | 3 shell scripts | 845 shortnames | tiny | `bin/list-all`, `bin/download`, `bin/install` + optional hooks | `asdf plugin test` | none |
| **proto** | declarative TOML (no logic) or WASM | 16 built-in + ~100 third-party | ~30 lines | `[install] download-url` with `{version}{arch}{os}` tokens, `[resolve]`, `[detect] version-files` | `generate_download_install_tests!` macros | none |
| **aqua** | `registry.yaml` per package | **2,108 packages; 1,870 (89%) carry `version_overrides`** | 10–80 lines | `type`, `asset` Go-template, `overrides[]`, `version_constraint` (expr-lang) + first-match-wins `version_overrides[]` | CI installs every `pkg.yaml` boundary version on 6 OS/arch runners; `aqua gr` *infers* an entry from release evidence, humans check | none (`gr` is heuristic) |
| **Nix / devenv / Devbox / Flox** | flake / module (58 lang + 43 svc modules) / JSON plugin (18) | — | devenv module ~100 LOC; Devbox `postgresql.json` 17 lines | NixOS-module options; Devbox `match` regex + `env` + `create_files` + `init_hook` | devenv 128 tests; Devbox txtar testscripts | none |
| **devcontainer Features** | `devcontainer-feature.json` + `install.sh` | 28 official; 320 collections | hugo: 1 KB JSON + 132-line sh; python: 1,125-line sh | typed `options` (`proposals` vs `enum`), `dependsOn`/`installsAfter`, lifecycle hooks; OCI distribution | `scenarios.json` × 6 base images | none |

Sources: https://github.com/renovatebot/renovate/tree/main/lib/modules/manager · https://raw.githubusercontent.com/renovatebot/renovate/main/lib/modules/manager/types.ts ·
https://github.com/dependabot/dependabot-core · https://mise.jdx.dev/dev-tools/backends/ · https://github.com/jdx/mise/tree/main/registry ·
https://aquaproj.github.io/docs/reference/registry-config/version-overrides · https://raw.githubusercontent.com/aquaproj/aqua-registry/main/pkgs/cli/cli/registry.yaml ·
https://containers.dev/implementors/features/ · https://github.com/jetify-com/devbox/tree/main/plugins · https://github.com/cachix/devenv/tree/main/src/modules

**aqua's `version_overrides` is the cleanest prior art for "behaviour changed at version X"** — `gh` has 8 eras
(`semver("<= 0.4.0")` … `"true"`), the base entry is `version_constraint: "false"` so every version is chosen by an
override, and `pkg.yaml` lists one version per era so CI exercises each boundary.

### B.2 Codemod / migration tooling

| System | Unit | Size | Tests | Deterministic/LLM boundary | Status 2026 |
|---|---|---|---|---|---|
| **GritQL** | pattern (`.md` with frontmatter, first ```grit block = body) | 4–60 lines | sections = must-match / before→after / identical = negative; `grit patterns test` | engine explicitly deterministic; hosted LLM workflows undocumented since Honeycomb acquisition (2025-04) | maintenance-only, 4.6k★ |
| **OpenRewrite** | Recipe: YAML (`recipeList` DAG), Refaster, or Java visitor | YAML 50–225 lines; visitor ~150 LOC | `rewriteRun(java(before, after))`, 2-cycle idempotence | Moderne 2026 line: "LLM plans, recipes execute"; MCP exposes deterministic tools; "30K tokens with the right tool vs 61M without" | very active (v8.91.4, 2026-09-01); JS/Python archived to Moderne-only |
| **ast-grep** | YAML rule | 5–30 lines | `valid`/`invalid` + snapshots; Noisy/Missing classification | "AI authors, engine executes": official Claude skill + MCP (`dump_syntax_tree`, `test_match_code_rule`) | 15.7k★, 0.45.3 |
| **jscodeshift** | transform fn | 7 LOC–hundreds | `__testfixtures__/*.input\|output.js` | none | slow (last release 2025-03) |
| **codemod.com** | package (`codemod.yaml` + `workflow.yaml` + JSSG + `rules/*.yml` + `tests/<case>/{input,expected}`) | — | `codemod jssg test` strictness levels | workflow step types include `ai` (Rig harness, max 5 retries) beside `ast-grep`/`js-ast-grep`/`run`; Studio: AI drafts, fixtures verify | very active (1.17.3, 2026-09-02); 1,215 registry packages |
| **Angular `ng update`** | `migrations.json` `{version, factory}` selected by semver | — | schematics tests | deterministic | shipping |
| **Next.js `@next/codemod`** | codemod per breaking change; `upgrade` bumps + runs | — | fixtures | deterministic; **residue markers**: inserts `@next/codemod` comments / `UnsafeUnwrapped*` casts so the build fails until resolved | shipping |
| **Google LLM migrations** (arXiv 2501.06972, 2504.09691) | pipeline | — | build + tests + repair loop, pass@k prompt variants, Gemini-as-judge ranking | deterministic targeting (Kythe, AST) → LLM edits → validate → human review; "80% of modifications AI-authored", "~87% committed unchanged", "LLM planning often not needed" | — |
| **Amazon Q / AWS Transform** | transformation plan → steps → summary | — | local build+tests round-trip per step | server-side LLM edits; plan-vs-actual diff; partial-success handoff to chat | Java live; .NET/mainframe under AWS Transform |
| **Slack Enzyme→RTL** | hybrid | — | human verify | codemod alone 45%, LLM alone 40–60%, **hybrid 80%** (codemod partial output + DOM annotations → LLM) | 2024 |
| **Meta Kotlinator** | 6 deterministic phases (~50 pre, J2K, ~150 post, lint, compiler loop) | — | compiler | no LLM; 40k+ conversions | 2024 |
| **Deno / Bun Node compat** | per-module status table + Node's own test suite pass-rate (Deno 2.8: 76.4%; Bun per-module) | — | Node test suite | deterministic (`npm:`/`node:` specifiers, lockfile import); no `deno migrate`; `dnt` reverse adapter runs tests in Node | Deno 2.9.6, Bun 1.4.0 |

Sources: https://docs.grit.io/language/overview · https://docs.openrewrite.org/authoring-recipes/recipe-testing · https://www.moderne.ai/blog ·
https://ast-grep.github.io/advanced/prompting.html · https://docs.codemod.com/cli/workflows · https://arxiv.org/abs/2501.06972 ·
https://arxiv.org/abs/2504.09691 · https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/how-CT-works.html ·
https://slack.engineering/balancing-old-tricks-with-new-feats-ai-powered-conversion-from-enzyme-to-react-testing-library-at-slack/ ·
https://nextjs.org/docs/app/guides/upgrading/codemods · https://docs.deno.com/runtime/reference/node_apis/

### B.3 Cross-cutting patterns (Part B)

1. **Layered split, thin adapters.** Renovate's manager/datasource/versioning and Dependabot's 7-class contract keep the
   per-ecosystem adapter small by pushing version comparison and registry lookup into shared layers; ~half of Renovate
   managers are only an extractor. mise's trait needs 2–3 methods; asdf 3 scripts.
2. **Declarative adapter first, code as escape hatch.** aqua/proto/mise/Devbox/devcontainer adapters are *data* with a
   typed options schema; code appears only for the residue (WASM, `install.sh`, regex managers, Java visitors).
3. **Version-era switching is first-class** in aqua (89% of packages), Dependabot (`SUPPORTED/DEPRECATED_VERSIONS`),
   Angular (`migrations.json` by semver), OpenRewrite (version-chained recipe DAGs `3_5 → 3_4 → … → 2_7`).
4. **Verify = run the real thing in a matrix**: aqua 6-way OS/arch; mise `test = {cmd, expected}`; devcontainer
   scenarios × base images; Dependabot per-eco Docker. Renovate stays at fixtures because it never executes tools.
5. **Conformance suites over per-adapter judgement**: every adapter must satisfy the contract and ship docs.
6. **Detection is a separate, cheap, deterministic step** (`managerFilePatterns`, `required_files_in?`, `idiomatic_files`
   with `version_regex`, Devbox `match`).
7. **The deterministic/LLM boundary has converged** across Google, Moderne, codemod.com, ast-grep, Slack: deterministic
   discovery/targeting → deterministic transforms first → LLM for the residue → verification = build+tests in a bounded
   retry loop → human review with explicit residue markers. Heuristic generation (aqua `gr`, codemod Studio, Google
   pass@k) is quarantined and must be proven by CI before trust.
8. **Plans and summaries are artefacts** (Q "transformation plan" + plan-vs-actual; Copilot `plan.md`/`progress.md`/
   `summary.md`; AWS Transform Objective → Plan → Tasks with approvals).
9. **Format preservation is a requirement** (OpenRewrite LST, recast, tree-sitter CST) so diffs are reviewable — for
   config porting this means comment- and order-preserving YAML/TOML round-tripping.
10. **Compat tracking = per-module status table + published pass-rate** against the *source's own* test suite
    (Deno/Bun vs Node) — the model for an inference engine feature-parity matrix.

---

## Implications for Robotnik

- **Adapter = declarative data + version ledger, with code as the escape hatch.** Copy aqua's shape: per-(engine)
  registry file, `version_constraint` eras, first-match-wins overrides, and one pinned version per era in the test list.
  No inference layer has this today; every one of them encodes version drift as shell or Python.
- **Three layers, not one.** Split Robotnik like Renovate: *detectors* (find config; cheap, deterministic), *engine
  adapters* (arg schema per version, loaded from the engine's own argparser/Pydantic where possible), *targets*
  (KServe/llm-d/Dynamo/Ray/LMI/GPUStack emitters). Most adapters should be extract-only at first.
- **Seed the canonical core from SageMaker LMI's `option.*` set** — it is the only shipped, multi-backend, production-tested
  portable key set — and extend with the vLLM-recipes strategy vocabulary for parallelism.
- **Mine existing implicit mappings as test fixtures**: llm-d's same-guide-three-engines overlays, GIE's per-engine metric
  table, GPUStack's dual-backend catalog entries, Xinference's version shims, KServe's `sort -V` gates, LMI's remap table.
  These become before/after pairs in the OpenRewrite/GritQL style (`.md` per recipe, one block per engine/version).
- **Verify like aqua and devcontainers, not like Renovate**: install the real engine at each era boundary in a matrix and
  run the emitted config; the "did it parse and start" check is cheap and catches most drift before any GPU benchmark.
- **Keep the LLM out of the adapter path and in the residue path**: deterministic mapping for everything with a
  known equivalence, LLM only for fields with no analogue, gated by the parse/start check and then the benchmark. Emit
  residue markers (Next.js style) so an un-ported field fails loudly rather than silently defaulting.
- **Emit a plan artefact and a plan-vs-applied diff** (Q/Copilot pattern). For inference the plan is: canonical config →
  per-engine emitted file → expected-unsupported fields → verification recipe → promotion route patch.
- **Target HTTPRoute weights + a second pool for promotion** and plug into Argo Rollouts/Flagger/Kargo; do not build a
  rollout controller. Dynamo (restart-only) and GPUStack (delete/recreate) need native adapters later.
- **Treat KAITO's `autoUpgrade` + drift status as the UX benchmark for version management** and LMI's breaking-changes
  page as the anti-pattern (prose, no tooling).
- **Publish a per-engine feature-parity matrix with pass-rates**, Deno/Bun style, generated from the adapter test matrix.
  It is a credibility asset on its own and the honest answer to "can I port this?".
