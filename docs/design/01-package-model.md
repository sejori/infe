# 01 — Package model: moving engine- and GPU-specific knowledge into packages (2026-09-03)

Question: can the engine-specific and GPU-specific code live in packages, leaving a generic core?
Answer: yes. It is the only shape the evidence in research/02 and research/04 supports, with one caveat:
the packages hold *rules and evidence*, never hardware-specific *values*. Values are derived at resolve time.

## Why this fixes the "no package manager" problem

research/05 concluded the package for inference is a tuple (model × engine × version × hardware × config) whose
contents change per target, so an npm-style registry of configs can't work. But the *knowledge* that produces a
config is small, textual, hardware-neutral and testable on a CPU:

- how engine E@v spells each concept, and what changed at each version
- how hardware H turns an intent (fit this model, leave this KV headroom) into engine values
- which (model, E@v, H, config) tuples have actually been verified, and what they scored

Those three things have npm economics. Putting them in packages restores what the registry approach lost.
The non-portable part (the emitted config) is an *output* of resolution, pinned in a lockfile, not a package.

## Four package kinds

| Kind | Keyed by | Contains | Mostly | Tested by | Prior art |
|---|---|---|---|---|---|
| **Engine adapter** | engine name; version eras inside | canonical→native field map per era, rename/removal ledger, validation hook (load the engine's own arg schema), emitter for native config file | data | parse-and-start against the real engine at each era boundary, CPU-only | aqua `version_overrides`, Renovate manager, LMI `vllm_rb_properties.py`, Xinference version shims |
| **Hardware profile** | GPU SKU (+ count, interconnect, driver era) | derivation rules: parallelism strategy from model size vs VRAM, memory budget → each engine's denominator, attention/kernel backend preferences, known-bad combos | data + small functions | evidence records that cite it | NIM profiles, KAITO `configureParallelism`, vLLM recipes `hardware_overrides`, AIConfigurator |
| **Target emitter** | deployment platform | canonical config + emitted engine file → KServe LLMISVC / llm-d Kustomize / Dynamo DGD / Ray `LLMConfig` / LMI `serving.properties` / GPUStack backend | data (templates) | golden-file diff | Renovate managers (extract + update), devcontainer features |
| **Evidence record** | full tuple (model+revision+precision, E@v, H, canonical config hash) | emitted config, benchmark report (llm-d BR 0.2.1 + hardware block), accuracy result, verdict, who ran it | data | is the test | vLLM recipes `verified` map, InferenceX rows, NIM "certified" |

The generic core is: manifest parser, resolver, lockfile, verify runner, diff engine, promotion emitter. It knows
no engine and no GPU.

## Resolution

```
manifest (intent)              robotnik.yaml: model, precision, features, SLA, target platform
  + engine adapter @ era       chosen by engine + version in manifest or lockfile
  + hardware profile @ SKU     chosen by target cluster/node facts (resolved at plan time, not authored)
  → canonical config           typed, engine-neutral, hardware-neutral
  → emitted native config      vllm --config yaml / sglang yaml / trtllm LlmArgs
  → target manifest            LLMISVC / DGD / Kustomize overlay
  → lockfile                   pins adapter version, profile version, emitted config hash, evidence ref
```

A port is: change the engine or version in the manifest, re-resolve with the new adapter era, diff the emitted
files, run verify, write an evidence record, promote. Fields with no mapping in the new era surface as explicit
residue (Next.js codemod style) rather than silently defaulting.

## Dependencies and constraints between packages

- Engine adapter eras are semver ranges on the engine (`>=0.12 <0.13`), aqua-style first-match-wins.
- Hardware profiles declare which engine adapters they have rules for; a missing pair resolves to "estimated", not an error.
- Evidence records pin exact adapter + profile versions. A verified record is invalidated when either bumps
  (that is the cache-key, and it is what makes "vLLM 0.26 → 0.27 on H200" a query, not a guess).
- Target emitters depend on a canonical-schema version only.

## Where code is allowed

Declarative first, code as a quarantined escape hatch (proto WASM / devcontainer `install.sh` / Renovate `updateArtifacts`):

- adapter `validate` hook: import the pinned engine's arg schema and check the emitted file (LMI pattern)
- profile `probe` hook: runtime measurement when a rule needs it (KAITO's `max-model-len` binary search) —
  runs inside verify, never at resolve time, so resolve stays deterministic and offline
- emitter `post` hook for platform quirks (llm-d's `USER=llm-d` env for vLLM ≥0.20)

No LLM in the resolve path. The LLM's job is drafting a new adapter era when an engine releases (aqua `gr` pattern):
diff the new arg schema against the previous era, propose the ledger entry, open a PR; CI proves it by
parse-and-start at the boundary version.

## What stays outside packages

- Engine GPU code itself (kernels, attention backends) lives in the engine image. Packages only carry
  *knowledge about* it (which backend on which SKU, which flag selects it).
- Model weights: OCI/ModelPack/HF. Robotnik references by HF path + revision; never re-packages.
- Rollout controllers: emit HTTPRoute weights and a second pool; let Argo/Flagger/Kargo drive.

## Risks

- **Adapter rot**: engine releases outpace maintainers. Mitigation: adapter CI runs nightly against engine `main`,
  the LLM-drafted era PR lands the same day a release does, and the parity matrix shows red honestly.
- **Profile explosion**: SKU × count × interconnect. Mitigation: profiles inherit (H100 → H100-SXM-8x-NVLink), and
  most rules are functions of `(vram_gb, gpus, nvlink: bool)` not SKU names; SKU-specific rules are the exception.
- **Evidence sparsity**: GPU hours are the bottleneck. Mitigation: parse-and-start is CPU-only and catches most
  drift; benchmark evidence is sparse by design and labelled as such.
- **Semantic mismatch hidden as a mapping**: `gpu-memory-utilization` vs `mem-fraction-static` vs
  `free_gpu_memory_fraction` have different denominators. Mitigation: canonical field is the outcome
  (KV bytes / headroom); each adapter derives; verify checks the derivation.

## First three packages to write

1. `engine/vllm` with eras 0.10 (V0 removal), 0.12 (structured-outputs rename), 0.13 (attention-backend flag), 0.2x current
2. `engine/sglang` with the 0.5.x eras, including the dp-attention semantics of `--tp-size`
3. `hardware/h100-8x-nvlink` and `hardware/h200-8x-nvlink`, seeded from vLLM recipes + InferenceX rows for DeepSeek-V4-Flash

## Scope correction (2026-09-03, later)

Pushback: "benchmark configs and versions on different hardware, then present optimal configs" is not a package
manager. Correct. The four-kind model above mixes two products. Split them:

### Product A — the package manager (Robotnik proper)

Standardised, deterministic, CPU-testable, no opinion about what is optimal.

- Engine adapters, hardware profiles (derivation *rules* only), target emitters.
- Verb set: `detect`, `resolve`, `emit`, `diff`, `check`, `port`.
- `check` = translation correctness, not performance: the emitted config parses against the pinned engine's
  arg schema, the engine starts, one request round-trips, unmapped fields are surfaced as residue.
- Lockfile records what *you* checked, locally. No central "verified" claims.
- Analogy: Renovate bumps the version and your CI decides; cargo swaps the crate and `cargo bench` is yours to run.
  Neither tells you which is fastest. Robotnik ports the config and `check` says it still boots.
- Hardware stays in packages only as rules (tp from GPU count, memory denominators per engine, known-bad combos).
  Rules are data, deterministic, and testable without a GPU. Benchmark-derived values are not rules.

### Product B — evidence / tuning (NOT Robotnik, at least not now)

- Running benchmarks across SKUs and ranking configs. That is InferenceX, AIConfigurator, dstack Presets, NIM
  certification. Needs a GPU fleet and continuous re-runs. Different economics, different product.
- Robotnik's only touchpoint: `bench` runs a benchmark recipe against two deployments and `diff` compares the
  reports. Tool, not service. Results stay with the user unless they choose to publish.
- If a community evidence layer ever exists, the resolver *reads* it as a hint ("verified on H200 by X");
  it never produces it centrally. Ingesting InferenceX rows or vLLM recipes' `verified` map is enough to start.

### What changes in the table above

- "Evidence record" is demoted from a package kind to a lockfile entry (`checked: {engine, version, host, date}`)
  plus an optional exported benchmark report. It is not published, not resolved against, not ranked.
- "Hardware profile" is narrowed to rules. Any rule that needs a measurement runs as a `probe` inside `check`,
  never at resolve time.
- The rename ledger and the parity matrix remain the two artefacts that do not exist elsewhere; both are
  Product A and both are CPU-only.
