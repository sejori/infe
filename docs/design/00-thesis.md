# Robotnik — kickoff thesis (2026-09-03)

Working name for a tool that detects a project's engine / versions / deps / commands / deploy config,
normalises them, and ports them to another engine, version, or target — for both software projects
(Node → Deno, Dockerfile → Wrangler) and inference deployments (vLLM x.y → SGLang a.b for a given model).

## 1. These are two different problems sharing one core

| | App porting (Node→Deno) | Inference config (vLLM↔SGLang, version bumps) |
|---|---|---|
| Schema | open-ended (source code semantics) | closed (finite CLI/YAML arg space) |
| Frequency | rare, one-off | continuous (every engine release) |
| Verifier | project's own tests + build | benchmark + correctness probe |
| Deterministic feasible? | config files yes; source no | almost entirely yes |
| Existing tooling | crowded (coding agents, codemods, Grit) | thin — this is the gap |

The shared core is a **manifest** (intermediate representation): "what is this thing" — engine,
version, deps, commands, runtime config, deploy target, verification commands. Detection
(codebase → manifest) is deterministic and cheap everywhere. Emission (manifest → target files)
is where the deterministic-vs-LLM choice actually lives.

## 2. Deterministic adapters vs LLM: it's a boundary, not a choice

- **Deterministic** for closed-schema work: engine/version detection, config-file parse+emit
  (package.json ↔ deno.json, vllm serve args ↔ sglang args), version bumps, lockfile handling.
  Prior art shows this is cheap when each adapter is narrow: Renovate has ~90 "managers" at
  ~100–300 lines each. Nobody writes one giant abstraction; they write many small, testable ones.
- **LLM** for open-ended work: source transforms (require→import, `node:` builtins, Express→Hono),
  semantic mapping where no 1:1 exists, explaining a failing verification. Packaged as prompts/skills
  that receive the *manifest* as structured context, never the raw repo.
- **The deterministic layer validates the LLM layer.** LLM output is acceptable only because the
  manifest declares how to verify it (test cmd, build cmd, benchmark, health probe).
- Cost/latency objection is real for daily version bumps and irrelevant for one-off ports.
  So: bumps and same-engine config changes are deterministic; cross-engine ports are LLM-assisted.

## 3. What makes it a tool and not "just use Claude Code"

The verification loop: `detect → plan → emit → verify → diff → promote | rollback`.
A migration without a verifier is a codemod. With one it's a CI-grade product.
For inference this is literally canary-by-config: deploy model M on engine E@v2 beside E@v1,
run the same probe/benchmark, compare, promote. That loop doesn't exist as a tool today.

## 4. Recommended wedge

Start with **inference engine configs**, not app porting:
- closed schema → deterministic adapters are tractable and provably correct
- daily pain, no incumbent tool (people keep YAML in git and remember which flags changed)
- verification is well-defined (throughput/TTFT/ITL + a correctness probe)
- we have the domain knowledge and live workloads to validate against
- the same manifest + verify loop then extends to app porting with LLM emitters bolted on

App porting is where LLM coding agents already do a passable ad-hoc job; the marginal value
of Robotnik there is the manifest + verifier, which we get for free from the wedge.

## 5. Prior art to survey (next step)

Engines/versions: mise, asdf, proto, volta, uv, Nix/devenv/Devbox/flox, devcontainers spec
Dependency managers: Renovate managers, Dependabot ecosystems
Codemods: jscodeshift, ast-grep, OpenRewrite, Grit (GritQL = deterministic patterns + LLM), Meta fastmod
LLM migration: Google's internal LLM-assisted migration paper (2024), Amazon Q Transform, Mantle
Cross-engine inference config: Ollama Modelfile, vLLM `--config`, SGLang server args, TGI, llama.cpp,
  NVIDIA Dynamo DGD/DCD, KServe, Ray Serve, llm-d, vllm production-stack, HF inference endpoints
Verification: vllm bench, GuideLLM, genai-perf, inference-benchmarker, lm-eval-harness

## 6. Open questions

- Manifest format: own schema vs extend devcontainer/mise/Nix conventions?
- How much of the inference config space is genuinely portable between engines (quantisation,
  parallelism, scheduler flags) vs engine-specific with no analogue?
- Where do runtime facts live (GPU count, memory) — manifest, or resolved at plan time?
- Is "port" really the verb, or is it "diff + verify" with port as one producer of diffs?

## 7. Research index (2026-09-03)

- `research/01-engine-config-surfaces.md` — 11 engines, flag-rename evidence, 21 platforms checked for cross-engine config
- `research/02-packaging-and-registries.md` — ~30 packaging/registry attempts + diagnosis of why none became "npm for inference"
- `research/03-verification-and-canary.md` — benchmark clients, accuracy gates, canary mechanics, artefact schemas
- `research/04-orchestration-and-prior-art.md` — K8s/PaaS layers above the engine; Renovate/aqua/OpenRewrite/Google-migration prior art
- `research/05-synthesis.md` — answers: what to wrap vs build, and why no package manager exists

## 8. Scope correction (2026-09-03)

Robotnik is the translator/package manager (adapters, rules, emitters, `check` = translation correctness).
Benchmarking across hardware and ranking "optimal" configs is a separate tuning product and is out of scope.
`bench` + `diff` exist as local tools only. See `01-package-model.md` "Scope correction".

## 9. Reframe: engines assembled from a registry (2026-09-03)

New stated goal: vLLM and SGLang should be *remakeable* from Robotnik packages, so a new paradigm = one package +
plug into your engine. Evidence in `research/06-engine-internals-composability.md`. Short version:

- Both engines are already assembled from registry packages **below the kernel boundary** (identical pins on torch,
  flashinfer, tilelang, quack-kernels, cutlass-dsl; shared xgrammar/llguidance, compressed-tensors, transformers v5,
  Mooncake/NIXL/LMCache). SGLang pulls FA3 from the HF Kernel Hub and has an in-tree per-op kernel registry;
  vLLM vendors forks in-wheel and does not use HF `kernels`.
- **Above the kernel boundary nothing is shared**: model definitions (300 vs 259 files), scheduler, KV manager
  (PagedAttention vs RadixAttention), model runner, CUDA-graph capture, sampler, API server, tool parsers (52 vs 43).
- The one working cross-engine seam is the **KV connector** (`KVConnectorBase_V1`): LMCache, Mooncake, NIXL, FlexKV
  and NVIDIA's Rust `kvbm` all plug into it. That is the CNI/CSI pattern, already proven once.
- Plugin interfaces are unstable: v0.11.0 deleted all V0 attention backends; vllm-ascend versions 1:1 with vLLM;
  vLLM's own guidance is `VLLMPatch` + `@min_vllm_version`; RFC #42770 abandons one-model-definition-for-all-hardware.

Ruling: "remake vLLM from packages" is not a realistic project. "Standardise one seam at a time and get both engines
to adopt it" is, and has precedent. Order: KV → kernels (reuse HF `kernels` packaging) → quant → platform →
scheduler → attention metadata → runner. Config translation (sections 1–8) remains the cheap on-ramp.
- `research/07-churn-and-attention-seam.md` — measured interface churn in both repos; three-layer attention model; seam goes at kernel-op layer
- `research/08-governance-and-convergence.md` — PyTorch Foundation/Inferact vs LMSYS/RadixArk; feature convergence timeline; merge unlikely

Ruling (2026-09-03): start the kernel-op manifest from SGLang's `sglang.kernels.spec` but validate against vLLM's
`AttentionBackend.supports_*` from day one; the engine-seam layer (entry points, KV connector) is stronger in vLLM.
