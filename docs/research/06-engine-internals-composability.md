# 06 — Are vLLM and SGLang already "assembled from a registry"? Which seams are pluggable?

Research date: 2026-09-03. Sources are the `main` branches of both engines cloned today
(vLLM `31e9c13`, 2026-09-03; SGLang `bf71035`, 2026-09-03), PyPI metadata, and primary docs.
Anything not verified against a primary source is marked **[unverified]**.

---

## 0. One-page summary

**Answer: partially, and unevenly.** Both engines already pull a large and *growing* fraction of
their kernel layer from PyPI/registries (FlashInfer, DeepGEMM/DeepEP builds, FA4, Humming,
TokenSpeed-MLA, TileLang, CUTLASS-DSL/QuACK, xgrammar/llguidance/outlines, compressed-tensors,
Mooncake/NIXL/LMCache, transformers), and they pin the *same versions* of several of them
(torch 2.13.0, flashinfer 0.6.18, tilelang 0.1.12, humming-kernels 0.1.12, tokenspeed-mla 0.1.8,
quack-kernels 0.6.4, nvidia-cutlass-dsl 4.6.2). SGLang has gone furthest: it depends on Hugging
Face `kernels` and loads FA3 from the Hub (`kernels-community/sgl-flash-attn3`), and since
July 2026 it has an in-tree **kernel registry** (`sglang.kernels`, `KernelSpec`/`KernelBackend`,
RFC #29630) that inventories multiple backends per op. vLLM's equivalent is the newer **vLLM IR**
(`@register_op`, "ops and implementations can be registered anywhere, in-tree or out-of-tree")
— currently 2 ops.

**But the engine *core* — scheduler, KV-cache manager/prefix cache, model runner, CUDA-graph
capture, sampler, model definitions, API server, tool/reasoning parsers, LoRA, multimodal
processors — is duplicated in both trees** (~1.4M Python LOC in SGLang, ~0.9M in vLLM), and is
exposed only through *class-path strings* or *monkeypatch hooks*, not through versioned interfaces.
The one seam that has become genuinely registry-shaped across engines is the **KV connector**:
LMCache, Mooncake, NIXL, and NVIDIA's `kvbm` (Rust, `pip install kvbm`) all plug into vLLM via
`KVConnectorBase_V1` + `kv_connector_module_path`, and into SGLang via HiCache storage backends /
disaggregation transfer backends. That is the CNI/CSI pattern actually working.

**Stability evidence:** vLLM v0.11.0 (2025-10-02) deleted the entire V0 engine including *all* V0
attention backends; hardware plugins (vllm-ascend) version 1:1 with vLLM releases and track a
specific upstream commit on `main`; vLLM's own Nov-2025 plugin blog recommends `VLLMPatch` class
patching guarded by `@min_vllm_version`. RFC #42770 (May 2026, open) reverses the "one model
definition for all hardware" goal in favour of per-backend model code and manual fusion — which
cuts *against* "model definition as a portable package".

**For Robotnik:** "remake vLLM from packages" would mean re-implementing the ~10 core subsystems
that neither engine exposes; the realistic path is the CNI/CSI one, and it is already underway
for exactly one seam (KV transfer/offload) and half-underway for kernels (FlashInfer + HF
`kernels` + `sglang.kernels`/vLLM IR). Scheduler, KV manager, model runner and model definitions
have *no* cross-engine interface and no visible effort to create one.

---

## 1. Dependency overlap (current `main`, 2026-09-03)

Sources: vLLM `requirements/common.txt`, `requirements/cuda.txt`, `requirements/kv_connectors.txt`,
`requirements/tpu.txt`, `requirements/xpu.txt`, `setup.py`, `cmake/external_projects/*.cmake`
(https://github.com/vllm-project/vllm/tree/main/requirements); SGLang `python/pyproject.toml`
(https://github.com/sgl-project/sglang/blob/main/python/pyproject.toml) and
`python/sglang/kernels/aot/CMakeLists.txt`. PyPI versions via https://pypi.org/pypi/<pkg>/json.

### 1a. Shared third-party components from PyPI

| Component | vLLM pin | SGLang pin | Note |
|---|---|---|---|
| torch | `==2.13.0` (cuda.txt) | `==2.13.0` | identical; both also pin torchaudio 2.11.0 |
| flashinfer-python | `==0.6.18` (+ `flashinfer-cubin==0.6.18` from https://flashinfer.ai/whl/, not PyPI) | `flashinfer_python[cu13]==0.6.18` | identical; SGLang also vendors a flashinfer commit inside its AOT kernel build |
| transformers | `>=5.10.4` | `==5.12.1` | both on transformers v5 |
| tokenizers | `>=0.21.1` | `==0.22.2` | |
| xgrammar | `>=0.2.1,<1.0.0` | `==0.2.1` | PyPI latest 0.2.5.post1 (2026-09-03) |
| llguidance | `>=1.7.0,<1.8.0` | `>=1.7.6,<2.0.0` | |
| outlines | `outlines_core==0.2.14` | `outlines==0.1.11` | different packages/generations |
| lm-format-enforcer | `==0.11.3` | — | vLLM only |
| compressed-tensors | `==0.17.0` | `==0.18.0` | **mismatched pins** — same lib, different versions |
| tilelang / apache-tvm-ffi | `0.1.12` / `0.1.11` | `0.1.12` / `0.1.11` | identical |
| humming-kernels[cu13] | `==0.1.12` | `==0.1.12` | identical (quant GEMM) |
| tokenspeed-mla | `==0.1.8` | `==0.1.8` | identical (MLA + spec decode) |
| quack-kernels / nvidia-cutlass-dsl[cu13] | `0.6.4` / `4.6.2` | `0.6.4` / `4.6.2` | identical (CuTe-DSL FA4/MSA) |
| numba | `==0.65.0` | `==0.65.1` | n-gram spec decode |
| msgspec, pyzmq, fastapi, pydantic, prometheus-client, openai, openai-harmony, anthropic, mistral_common, partial-json-parser, pybase64, setproctitle, watchfiles, einops, pillow, sentencepiece, tiktoken, ninja | both | both | serving/plumbing layer is largely the same PyPI set |
| mooncake-transfer-engine | `>=0.3.12` (kv_connectors.txt, optional) | not declared; lazy-imported in `disaggregation/mooncake`, `layers/moe/token_dispatcher/mooncake.py`, `mem_cache/storage/mooncake_store` | PyPI 0.3.13.post1 |
| nixl | `==1.3.2` (kv_connectors.txt); **`==0.3.0` in tpu.txt** | lazy import (`disaggregation/nixl`, `mem_cache/storage/nixl`, `token_dispatcher/nixl.py`) | PyPI 1.4.1 |
| lmcache | `>=0.3.9` (kv_connectors.txt) | `mem_cache/storage/lmcache/` (lazy) | PyPI 0.5.4 |
| Hugging Face `kernels` | **not used** (no `from kernels import` in `vllm/`) | `kernels>=0.14.1,<0.15` — used to load FA3 from Hub (`kernels-community/sgl-flash-attn3`, rev `v1`) | PyPI 0.16.1 |
| flash-attn (FA4) | vendored: `vllm-project/tml-fa4` + `vllm-project/MSA` via CMake | `flash-attn-4>=4.0.0b18` from PyPI | |
| ray | only in tpu.txt / xpu.txt | optional extra `ray[default]>=2.55.1` | |
| triton | via torch (xpu: `triton==3.7.2+xpu`) | via torch; vendors triton v3.7.1 *source* in AOT build | |

### 1b. Engine-owned kernel packages and vendored forks

| | vLLM | SGLang |
|---|---|---|
| In-wheel compiled ops | `_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_qutlass_C`, `_flashmla_C`, `_flashmla_extension_C`, `_sparse_flashmla_C`, `_flashkda_C`, `vllm_flash_attn/_vllm_fa2_C`, `_vllm_fa3_C`, `cumem_allocator`, `fs_io_C`, `_rocm_C`, `triton_kernels` (setup.py L1010–1024) | none in the `sglang` wheel; all AOT ops in separate wheel |
| Separate kernel wheel | `vllm-xpu-kernels==0.1.14.1` (Intel only) | `sglang-kernel==0.4.6.post1` (renamed from `sgl-kernel`, last `sgl-kernel` PyPI 0.3.21 on 2026-01-15); source at `python/sglang/kernels/aot/`; `sgl-kernel-xpu` separate repo |
| Vendored forks (CMake) | `vllm-project/flash-attention` (FA2/FA3 fork), `vllm-project/FlashMLA`, `vllm-project/FlashKDA`, `vllm-project/tml-fa4`, `vllm-project/MSA`, DeepGEMM upstream (`deepgemm.cmake`), QuTLASS, `nvidia/cutlass`, triton_kernels | AOT build vendors `nvidia/cutlass` (pinned commit), `fmtlib/fmt`, `triton-lang/triton v3.7.1`, `flashinfer-ai/flashinfer` (pinned commit), `sgl-project/sgl-attn` (FA fork) |
| DeepGEMM | built from upstream source into wheel | `sgl-deep-gemm==0.1.7` PyPI wheel (SGLang-maintained build) |
| DeepEP | **not a dependency**; docs tell users to install nvshmem/pplx/DeepEP by hand (PR #17964) | `sgl-deep-ep==0.1.2` PyPI wheel |
| JIT kernels | vLLM IR (`vllm/ir`, 2 ops) + `CustomOp` dispatch; torch.compile/Inductor | `sglang.kernels.jit` (CUDA JIT infra) + ~280 Triton/CuTe/TileLang kernels swept into `sglang.kernels.ops.*` (RFC #29630) |

**Where one vendors what the other imports:** FA3/FA4 (vLLM vendors forks; SGLang imports
`flash-attn-4` from PyPI and FA3 from the HF Hub), DeepGEMM/DeepEP (vLLM builds/omits; SGLang
ships its own PyPI builds), FlashMLA (vLLM vendors a fork in-wheel; SGLang exposes `flashmla` as an
attention backend served through sglang-kernel/flashinfer — exact provenance **[unverified]**).

---

## 2. vLLM pluggable seams

Legend: **Public** = documented entry-point/config interface; **Class-path** = accepts a
`"module.Class"` string resolved via `resolve_obj_by_qualname` but not a documented contract;
**In-tree** = enum/Literal, no OOT registration path.

| Seam | Mechanism (file) | Public / class-path / in-tree | Stable across V0→V1? | Out-of-tree users |
|---|---|---|---|---|
| Plugin entry points | 6 groups in `vllm/plugins/__init__.py` + `v1/sample/logits_processor/__init__.py`: `vllm.general_plugins`, `vllm.platform_plugins`, `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`, `vllm.endpoint_plugins` (opt-in via `VLLM_PLUGINS` only), `vllm.logits_processors`. Docs: `docs/design/plugin_system.md`, `endpoint_plugins.md`, `io_processor_plugins.md`, `lora_resolver_plugins.md` (https://docs.vllm.ai/en/latest/design/plugin_system/) | **Public** (setuptools entry points) | Discovery mechanism stable; *what a plugin can touch* is not. vLLM's own blog (2025-11-20, https://vllm.ai/blog/2025-11-20-vllm-plugin-system) recommends `VLLMPatch[TargetClass]` surgical patching guarded by `@min_vllm_version` because releases are "two weeks apart" with "hundreds of PRs every week" | vllm-ascend, vllm-spyre, vllm-gaudi, tpu-inference, vllm-metal, vllm-neuron, bart-plugin; vLLM's own LoRA resolvers are registered through `vllm.general_plugins` in `pyproject.toml` |
| Hardware platform | `vllm.platform_plugins` → `Platform` subclass returning `worker_cls`, `get_attn_backend_cls`, `get_device_communicator_cls` (`vllm/platforms/__init__.py` L243–290; only one OOT platform may activate) | **Public** (documented) but the surface is the whole `Platform`/`WorkerBase`/attention-metadata contract | **No.** RFC #12992 (2025-02) had "migrating hardware pluggable capability to V1" as a *non-goal* (https://github.com/vllm-project/vllm/issues/12992); v0.11.0 (2025-10-02) removed V0 engine + all V0 attention backends (https://github.com/vllm-project/vllm/releases/tag/v0.11.0); vllm-ascend releases are versioned **1:1 with vLLM versions** and `main` tracks a specific vLLM commit (https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) | vllm-ascend (PyPI 0.23.0 vs vLLM 0.28.0 — 5 minors behind), vllm-spyre (1.10.0, independent versioning), vllm-gaudi (not on PyPI), tpu-inference (**co-versioned 0.28.0**, pinned in `requirements/tpu.txt`), vllm-openvino (last push 2026-07-16), vllm-metal (MLX), vllm-neuron; Intel XPU is *in-tree* platform + separate `vllm-xpu-kernels` wheel |
| OOT models | `ModelRegistry.register_model(arch, "pkg.mod:Class")` from a general plugin; Transformers fallback via `ModelImpl = Literal["auto","vllm","transformers","terratorch"]` (`vllm/config/model.py` L109); 14 registry entries already point at `TransformersForCausalLM`/`TransformersMoEForCausalLM` (`models/registry.py` L687–722) | **Public** (documented, example repo) | Registration API survived; model *base classes/interfaces* (attention metadata, `CustomOp`, quant layers) churn. **RFC #42770 (2026-05-15, open) explicitly abandons "single model definition for every hardware"** and moves to per-backend model code with manual fusion (https://github.com/vllm-project/vllm/issues/42770) | bart-plugin, tpu-inference (JAX model defs), vllm-ascend |
| KV transfer | `KVConnectorFactory.register_connector(name, module_path, class_name)` + `kv_transfer_config.kv_connector_module_path` (`vllm/distributed/kv_transfer/kv_connector/factory.py` L28–125); base `KVConnectorBase_V1`; 19 in-tree connectors incl. `NixlConnector`, `LMCacheConnectorV1`, `LMCacheMPConnector`, `MooncakeConnector`, `MooncakeStoreConnector`, `OffloadingConnector`, `MultiConnector`, `MoRIIOConnector`, `FlexKVConnector`, `HF3FSKVConnector`, `ExampleConnector` | **Public de-facto** — external packages ship connectors and are loaded by module path | V1 connector API (`KVConnectorBase_V1`) is the one interface external vendors build to; the V0 connector API was removed | LMCache (PR #12953, 2025-02-25), Mooncake (PR #10884, 2024-12-15), NIXL (PR #17751, 2025-05-12), **NVIDIA `kvbm`** (`pip install kvbm`, `kv_connector=DynamoConnector kv_connector_module_path=kvbm.vllm_integration.connector`, https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/kvbm/README.md), FlexKV, llm-d |
| Scheduler | `SchedulerConfig.scheduler_cls: str \| type` → `resolve_obj_by_qualname` (`vllm/config/scheduler.py` L148–222); `--scheduler-cls`; base `SchedulerInterface` | **Class-path** (documented as a config field; interface is "implement everything in `SchedulerInterface`") | Introduced for V1 by PR #14466 ("pluggable scheduler", joerunde/IBM); the V0 scheduler is gone | vllm-spyre overrides the scheduler for hardware constraints (search: https://github.com/vllm-project/vllm/pull/14466); vLLM plugin blog uses a priority scheduler patch as *the* example |
| Attention backend | `AttentionBackendEnum` with ~40 members mapping to class paths + `CUSTOM = None` + `register_backend(AttentionBackendEnum.CUSTOM, class_path)` (`vllm/v1/attention/backends/registry.py` L131–293); platform `get_attn_backend_cls` may also return any qualname | **Class-path** (a single `CUSTOM` slot; not documented outside the code) | **No** — v0.11.0 deleted every V0 attention backend; the V1 `AttentionMetadata`/builder contract is what plugins must re-implement | vllm-ascend, vllm-spyre, tpu-inference provide their own backends via the platform hook |
| Quantization | `QuantizationMethods` Literal (27 names) + `register_quantization_config(name)` decorator adding to `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` (`layers/quantization/__init__.py` L12–178) | **Public** (documented registration decorator) | Registration API stable; the `LinearMethodBase`/`FusedMoEMethodBase` contracts churn with kernel changes | amd-quark (`quark`), torchao, auto-round (XPU), Gaudi INC moved to vllm-gaudi |
| Structured output | `StructuredOutputsBackend = Literal["auto","xgrammar","guidance","outlines","lm-format-enforcer"]` (`vllm/config/structured_outputs.py`) | **In-tree** enum; backends are PyPI libs but the *selection* is hard-coded | V1 rewrote the structured-output layer; `lm-format-enforcer` retained | — |
| Speculative decoding | `SpeculativeMethod = Literal["ngram","medusa","mlp_speculator","draft_model","suffix","custom_class", Eagle*, NgramGPU*, DSpark*]` (`vllm/config/speculative.py` L71) | **In-tree** enum with a `custom_class` escape hatch (class path) | V1 rewrote spec decode (EAGLE-first); V0 methods dropped | — |
| LoRA | `--lora-modules`; `LoRAResolver.register_resolver` + `vllm.general_plugins` resolvers (`vllm/lora/resolver.py`; `pyproject.toml` entry points) | **Public** for *resolvers*; LoRA kernels (punica/sgmv) in-tree | resolver plugin doc added post-V1 | — |
| Multimodal processors | `MultiModalRegistry.register_processor` (`vllm/multimodal/registry.py` L149); IO processor plugins group | **Public** (decorator; documented) | V0 registry methods removed at v0.11.0 | prithvi/terratorch IO processors |
| Model loader | `LoadFormats` Literal + `register_model_loader(load_format)` decorator (`model_loader/__init__.py` L32–124) | **Public** (decorator) | — | runai-model-streamer, fastsafetensors, tensorizer are PyPI libs behind in-tree loaders |
| torch.compile passes | `CompilationConfig.inductor_passes: dict[str,str]` (qualname strings), `custom_ops` `+/-` toggles, `PassConfig`; `CustomOp.register_oot` for OOT op dispatch; **vLLM IR** `@register_op` "ops and implementations can be registered anywhere, in-tree or out-of-tree" (`docs/design/vllm_ir.md`) | **Class-path** (passes) / **Public** (`CustomOp.register_oot`, documented in `docs/design/custom_op.md`) | RFC #42770 is *removing* full-graph torch.compile reliance | vllm-ascend uses `register_oot` |
| Distributed | `DistributedExecutorBackend = Literal["ray","mp","uni","external_launcher"]` **or** a class/qualname (`v1/executor/abstract.py` L54–83); device communicators: `custom_all_reduce`, `quick_all_reduce`, `flashinfer_all_reduce`, `symm_mem`, `pynccl`, `xpu_communicator`, platform `get_device_communicator_cls` | **Class-path** for executor; communicator via platform plugin | — | tpu-inference, vllm-ascend supply communicators |
| Endpoints / stats / logits procs | `vllm.endpoint_plugins` (opt-in), `vllm.stat_logger_plugins`, `vllm.logits_processors` | **Public** | added 2025–26 | — |

**Net:** vLLM has a *real* entry-point plugin system, but only the platform, model, KV-connector,
model-loader, quant-config, multimodal-processor and endpoint/logger seams are documented; the
scheduler, attention backend, executor and compiler passes are "pass a class path and implement
the whole internal interface". The V0→V1 transition removed the entire V0 attention/scheduler/
worker stack, and every hardware plugin still pins itself to a vLLM release or commit.

---

## 3. SGLang pluggable seams

Sources: `python/sglang/srt/server_args.py`, `srt/plugins/__init__.py`, `srt/platforms/__init__.py`,
`srt/layers/attention/attention_registry.py`, `srt/models/registry.py`, `srt/mem_cache/storage/backend_factory.py`,
`python/sglang/kernels/README.md`, docs https://docs.sglang.io/docs/hardware-platforms/plugin.

| Seam | Choices (main, 2026-09-03) | Mechanism | Public / class-path / in-tree |
|---|---|---|---|
| Plugin system | `sglang.srt.platforms` (hardware platform, selected by `SGLANG_PLATFORM`), `sglang.srt.plugins` (general: `HookRegistry` with before/after/around/replace hooks and class replacement, filtered by `SGLANG_PLUGINS`) | setuptools entry points (`srt/plugins/__init__.py`); designed in issue #20372 (2026-03-11, closed) | **Public, documented**, but new (2026) — docs say the scope "currently targets OOT hardware platforms"; in-tree CUDA/ROCm/NPU/XPU "continue to use the existing `is_cuda()`/`is_npu()`" branches; platform methods are annotated `[Active]` vs `[Planned]` |
| Attention backend | `--attention-backend`: triton, torch_native, flex_attention, dsa, dsv4, cutlass_mla, fa3, fa4, flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, dual_chunk_flash_attn, hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu (+ separate prefill/decode/draft backend flags) | `ATTENTION_BACKENDS` dict + `@register_attention_backend(name)` decorator; `add_attention_backend_choices = ATTENTION_BACKEND_CHOICES.extend` exported so platform plugins can add names | **In-tree registry with a plugin extension hook** (no class-path string) |
| Sampling backend | `flashinfer`, `pytorch`, `ascend` | set literal | In-tree |
| Grammar backend | `xgrammar`, `outlines`, `llguidance`, `none` (+ `add_grammar_backend_choices` hook) | if/elif in `constrained/base_grammar_backend.py::create_grammar_backend` | In-tree; plugin can add a name |
| MoE runner | `auto, deep_gemm, triton, triton_kernel, flashinfer_trtllm, experimental_sgl_trtllm, flashinfer_trtllm_routed, flashinfer_cutlass, flashinfer_mxfp4, flashinfer_cutedsl, cutlass, aiter, marlin, humming, experimental_sgl_marlin, hpc_ops, megamoe, intel_xpu` | list | In-tree |
| MoE all-to-all | `none, deepep, mooncake, nixl, mori, ascend_fuseep, flashinfer, megamoe, deepep_v2, pplx, ascend_tp` (+ enum `CUSTOMIZED`) | enum `MoeA2ABackend` | In-tree |
| Speculative | `EAGLE, EAGLE3, NEXTN(implicit), STANDALONE, NGRAM, DFLASH, UNO, DSPARK, FROZEN_KV_MTP, NONE` | enum `SpeculativeAlgorithm` | In-tree |
| Disaggregation transfer | `mooncake, nixl, ascend, fake, mori, mooncake_tcp` | per-backend subpackages under `srt/disaggregation/` | In-tree list; the *libraries* are PyPI |
| HiCache storage | `file, sim, mooncake, hf3fs, nixl, aibrix, dynamic, eic, simm, mori, shm` (+ lmcache, flexkv, umbp dirs) | `mem_cache/storage/backend_factory.py` — has a `loader/module_path/class_name` registry entry shape (L60–62) and per-name elif | In-tree registry; `dynamic` suggests module-path loading **[unverified]** |
| Load format | `auto, pt, safetensors, npcache, dummy, sharded_state, presharded, gguf, expert_pack, bitsandbytes, mistral, layered, flash_rl, remote, remote_instance, fastsafetensors, private, runai_streamer` | list | In-tree |
| `--model-impl` | `auto`, `sglang`, `transformers`, `mindspore` | `models/transformers.py` (`TransformersForCausalLM`, `TransformersMoEForCausalLM`, `TransformersMultiModalForCausalLM`, embedding variants) | Public flag |
| Custom models | `SGLANG_EXTERNAL_MODEL_PACKAGE` env → `ModelRegistry.register(pkg, overwrite=True)` (`models/registry.py` L131–134) | package scan for `EntryClass` | Public but env-var based; no per-arch decorator |
| KV-cache dtype | `auto, fp8_e5m2, fp8_e4m3, bf16, mxfp8, nvfp4, fp4_mx_block16…` | list | In-tree |
| Kernels | `sglang.kernels` registry: `register_kernel(KernelSpec(op_id, backend, "module:attr"))`, `KernelBackend` ∈ {AOT (`sgl_kernel`), JIT, TRITON, FLASHINFER, DEEPGEMM, CUTE_DSL, AITER, TORCH_NPU, KDA…}, `BaseFusedOp` with `forward_<backend>` + `register_oot_forward()` for platform plugins; 20 op groups; FA3 loaded from HF Hub via `kernels.get_kernel("kernels-community/sgl-flash-attn3")` | RFC #29630 (2026-06-29 → complete 2026-07, https://github.com/sgl-project/sglang/issues/29630) | **Public in-tree namespace**; README: "The public wrappers currently default to the AOT `sgl_kernel` implementation (the stable wheel boundary)"; "no priority ranking or heuristic auto-selection" — alternatives are "inventory only" |
| Hardware backends | in-tree dirs `hardware_backend/{cpu,gpu,mlx,musa,npu,xpu}`, platforms `{cuda,rocm,cpu,xpu}`; TPU via separate `sglang-jax` repo (https://github.com/sgl-project/sglang-jax, pushed 2026-09-03) | mixture of `is_cuda()` guards and `SRTPlatform` | Partly plugin-ised; NPU/MUSA/MLX in-tree |
| Router | `sgl-model-gateway` 0.3.2 (Rust; crates `smg`, `amg`; formerly `sgl-router`) + Rust crates `sglang-radix-tree`, `sglang-server`, `sglang-grpc`, `sglang-mm` under `rust/` | separate Cargo workspace, gRPC proto | Separable component (own version) |

**Is there a documented plugin system?** Yes, since 2026 — two entry-point groups + a hook
registry (https://docs.sglang.io/docs/hardware-platforms/plugin). It is explicitly scoped to
hardware vendors and is younger and narrower than vLLM's. No `--scheduler-cls`-style seam exists;
the scheduler (`managers/scheduler.py` + mixins) and RadixAttention cache (`mem_cache/`) are only
reachable through hook/replace patches.

---

## 4. Registry-shaped efforts for engine internals

### 4a. Hugging Face `kernels` / Kernel Hub
- What: `kernels` (PyPI 0.16.1, 2026-08-24) loads compiled kernels from Hub repos of the
  first-class `kernel` type; `kernel-builder` (Nix) builds all variants. Docs:
  https://huggingface.co/docs/kernels/index, https://huggingface.co/docs/kernels/kernel-requirements.
- Variant scheme (verified): `build/<framework><version>-cxx<abiver>-<cu><cudaver>-<arch>-<os>`
  e.g. `torch26-cxx98-cu118-x86_64-linux`; backend types `cann, cpu, cuda, metal, neuron, rocm, tpu, xpu`;
  `metadata.json` with `version` (int major), `kernels-minver`, `backend.archs`, sha256 digests,
  build provenance; ABI3 + manylinux_2_28; torch ops must be registered under a per-build-unique
  `torch.ops.<namespace>` so multiple versions can coexist; **trusted-publisher allowlist**
  (`trust_remote_code` otherwise); **major-version bump required even for additive changes** that
  don't reach all variants. Layers API: pure `nn.Module` `forward`-only replacements
  (`use_kernel_forward_from_hub`, `kernelize`).
- Size: 61 repos under `kernels-community` (HF API, 2026-09-03; includes dash/underscore
  duplicates), incl. `flash-attn2/3/4`, `vllm-flash-attn3`, `sgl-flash-attn3`, `paged-attention`,
  `deep-gemm`, `flash-mla`, `megablocks`, `punica-sgmv`, `aiter-flash-attn`, `triton-kernels`,
  `metal-flash-sdpa`, `mlx-rmsnorm`. Hub-wide `filter=kernel` returned 74 repos (mostly
  individuals) — **count method unverified**.
- Consumers (verified): **transformers** (`use_kernels=True`, `KernelConfig(kernel_mapping=
  {"RMSNorm": "kernels-community/rmsnorm"})`, attention impls `kernels-community/vllm-flash-attn3`
  — https://huggingface.co/docs/transformers/kernels); **SGLang** (dependency + FA3 from Hub, see §1);
  diffusers/TRL per HF docs (search-level, **[unverified in code]**); **vLLM: none** (no import in
  `vllm/`; vllm-omni has an open feature request #4638). TGI **[unverified]**.

### 4b. Transformers as modeling backend
- vLLM: `--model-impl transformers` (`ModelImpl` Literal); 14 registry archs already *default* to
  the Transformers backend (FlexOlmo, GPTBigCode, HunYuan dense/MoE, Olmo/2/3, SmolLM3, Starcoder2,
  VaultGemma…). Blog https://vllm.ai/blog/2025-04-11-transformers-backend (2025-04-11; VLMs 2025-07-21).
- SGLang: `--model-impl transformers` with CausalLM/MoE/Multimodal/Embedding wrappers
  (`srt/models/transformers.py`); registry normalises unknown archs to `TransformersForCausalLM`.
  Blog https://huggingface.co/blog/transformers-backend-sglang (2025-06-23).
- **Duplication is still the norm**: vLLM `model_executor/models` = 281 top-level files / 300 incl.
  subdirs / 193,656 LOC, 384 registered archs; SGLang `srt/models` = 222 / 259 / 167,662 LOC.
  Neither blog claims transformers is "source of truth"; both frame it as fallback for
  long-tail archs. RFC #42770 (vLLM, May 2026) moves the *other* way: per-hardware model code.

### 4c. FlashInfer as de-facto NVIDIA kernel registry
- Both engines pin `flashinfer-python==0.6.18`; FlashInfer README lists SGLang, vLLM, TensorRT-LLM,
  TGI, MLC-LLM, LightLLM, lorax, ScaleLLM as adopters; packaging is `flashinfer-python` (JIT) +
  `flashinfer-cubin` (precompiled, **no longer on PyPI since 0.6.14** — vLLM `setup.py` L1322 strips
  it from install_requires) + `flashinfer-jit-cache`. Both engines expose trtllm-gen kernels through
  it (`trtllm_mla/trtllm_mha`, `flashinfer_trtllm` MoE in SGLang; `FLASHINFER_MLA*` in vLLM).
  Adoption: SGLang PR #1 (2024-01-08); vLLM PR #4353 (2024-05-03).
  https://github.com/flashinfer-ai/flashinfer

### 4d. Components adopted by both (earliest merged PR with the name in title, via GitHub search)
| Component | vLLM | SGLang |
|---|---|---|
| xgrammar | #10785, 2024-12-03 | #1752, 2024-10-25 |
| llguidance | #17839, 2025-05-09 (earlier use likely) | #3298, 2025-02-26 |
| compressed-tensors | #5350, 2024-06-10 | #4743, 2025-03-26 |
| DeepGEMM | #13917, 2025-03-06 | #3893, 2025-03-02 |
| DeepEP | #17964, 2025-05-12 (docs only; not a dep) | #4232, 2025-03-19 |
| Mooncake | #10884, 2024-12-15 | #4880, 2025-04-10 |
| NIXL | #17751, 2025-05-12 | #5477, 2025-04-21 |
| LMCache | #12953, 2025-02-25 | #9741, 2025-09-07 |

### 4e. Explicit engine-component interface standardisation
- **NVIDIA Dynamo KVBM** — verified: Rust core with Python bindings, `pip install kvbm` (PyPI 1.4.2,
  2026-08-28); package ships `kvbm.vllm_integration` (`DynamoConnector`, wired with
  `--kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector"}'`)
  and `kvbm.trtllm_integration`; **no `sglang_integration` subpackage** — SGLang reaches Dynamo via
  HiCache's NIXL storage backend (docs). So KVBM *is* a shared KV-block-manager package, but it
  attaches through each engine's *own* connector API rather than replacing the in-engine allocator.
  https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/kvbm
- **LMCache** — tech report (arXiv 2510.09665) describes the vLLM `KVConnectorBase_V1` as a
  "standardized KV cache connector interface … the design initiated by LMCache, maintained jointly
  with vLLM"; SGLang integration is via HiCache storage backend (different interface).
- **llm-d** — `llm-d-kv-cache` indexer consumes KV *events* (ZMQ, `BlockStored/BlockRemoved`) with
  `VLLMAdapter` and `SGLangAdapter`; llm-d docs: "works with any KV-cache connector compatible with
  vLLM or SGLang … LMCache, Mooncake, NVIDIA KVBM plug into the model server through its KV-cache
  connector API". https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/architecture.md
- **vLLM IR** (`docs/design/vllm_ir.md`): functional op dialect over torch FX, "ops and
  implementations can be registered anywhere, in-tree or out-of-tree", "OOT compiler backends can
  lower from the higher-level representation (in-progress)". 2 ops today.
- **SGLang `sglang.kernels`** (RFC #29630): per-op multi-backend `KernelSpec` registry — the closest
  thing to a "kernel registry with a stable public op API" inside an engine.
- **Modular MAX**: graph-compiled engine with Mojo kernels targeting CUDA/ROCm/Metal from one
  codebase (https://www.modular.com/) — a *vertically integrated* alternative, not a registry
  **[marketing-level source only]**.
- **LMDeploy**: two engines (TurboMind C++/CUDA, PyTorch) sharing an API layer
  (https://lmdeploy.readthedocs.io/en/latest/inference/pytorch.html) — an *intra-project* split,
  models ported PyTorch→TurboMind by hand.
- **Aphrodite**: vLLM fork (repo pushed 2026-08-13, 1.8k stars) — fork, not package reuse.
- **Splits**: `vllm-omni` (6.6k stars, active), `sglang-omni`, sglang diffusion as an *extra*
  (`sglang[diffusion]`), `sglang-jax`, `tpu-inference` — these split by *modality/hardware*, all
  still vendoring their own scheduler/runner.
- **PyTorch ecosystem**: both engines register custom ops in `torch.ops` (HF `kernels` mandates
  it), `torchao` is a vLLM quant method; `torch.compile` backends are pluggable in vLLM
  (`CompilationConfig.backend`), but RFC #42770 is de-emphasising full-graph compile.
- **Prior "engine from registry packages" / "LLVM for serving" / "CNI-CSI for engines"
  proposals: none found.** Searches surfaced FlashInfer's paper (kernel-library framing), the
  LMCache report (connector-as-standard framing), llm-d (connector-agnostic framing), and
  Modular's compiler framing. No blog/RFC proposing that an engine be *assembled* from registry
  packages was located — **treat as absent, not disproven.**

---

## 5. Duplication evidence (files / Python LOC per subtree, `main` 2026-09-03)

| Subsystem | vLLM path | files | LOC | SGLang path | files | LOC | Shared via PyPI? |
|---|---|---|---|---|---|---|---|
| Model definitions | `model_executor/models` | 300 | 193,656 | `srt/models` | 259 | 167,662 | No (transformers fallback only for long tail) |
| Scheduler | `v1/core/sched` | 7 | 4,117 | `managers` (scheduler + mixins + policies) | 50 | 38,339 | No |
| KV manager / block alloc / prefix cache | `v1/core` (PagedAttention block pool, `kv_cache_manager.py`) | 15 | 11,963 | `mem_cache` (RadixAttention, hiradix, Rust radix tree, allocators) | 134 | 69,464 | No |
| Attention backends | `v1/attention/backends` | 55 | 30,914 | `layers/attention` | 110 | 64,753 | Kernels yes (FlashInfer/FA/FlashMLA), *wrappers/metadata* no |
| Model runner / worker | `v1/worker` | 113 | 39,449 | `model_executor` | 57 | 20,442 | No |
| CUDA-graph capture | `compilation/cuda_graph.py`, `v1/cudagraph_dispatcher.py` | — | — | `model_executor/cuda_graph_*`, `graph_*` | — | — | No |
| Sampler | `v1/sample` | 15 | 5,252 | `sampling` | 11 | 1,933 | Partly (FlashInfer sampling kernels) |
| OpenAI API server | `entrypoints/openai` | 27 | 11,098 | `entrypoints/openai` | 33 | 15,188 | No |
| Tool-call parsers | `tool_parsers` | 52 | 16,847 | `function_call` | 43 | 14,012 | No |
| Reasoning parsers | `reasoning` | 33 | 5,154 | `parser` | 9 | 6,094 | No |
| LoRA | `lora` | 42 | 14,271 | `lora` | 46 | 15,531 | No |
| Multimodal processors | `multimodal` | 32 | 11,768 | `multimodal` | 87 | 23,451 | No (both re-wrap HF processors) |
| Speculative decoding | `v1/spec_decode` | 19 | 6,061 | `speculative` | 56 | 27,265 | No |
| Quantization | `layers/quantization` | 121 | 37,463 | `layers/quantization` | 101 | 39,814 | Checkpoint *format* yes (compressed-tensors, different pins); kernels partly |
| MoE | `layers/fused_moe` | 100 | 45,151 | `layers/moe` | 62 | 28,202 | DeepGEMM/DeepEP/FlashInfer kernels yes; runners no |
| KV transfer / disagg | `distributed/kv_transfer` | 71 | 34,234 | `disaggregation` | 34 | 27,673 | Transport libs yes (Mooncake, NIXL, LMCache); connector code no |
| Structured output | `v1/structured_output` | 8 | 2,571 | `constrained` | 9 | 2,351 | Grammar engines yes (xgrammar/llguidance/outlines) |
| Compilation | `compilation` | 45 | 16,138 | `compilation` | 15 | 2,381 | torch.compile yes; passes no |
| Distributed comms | `distributed` | 140 | 61,224 | `distributed` | 29 | 10,318 | NCCL yes; custom allreduce no |
| **Whole tree** | `vllm/` | — | **897,549** | `python/sglang/` | — | **1,398,830** | |

Pattern: everything *below* the kernel boundary is increasingly shared through PyPI (and pinned
to identical versions); everything *above* it — the control plane, memory manager, runner,
API/parsers — is fully duplicated with no shared interface.

---

## 6. Implications for Robotnik

1. **Already registry-shaped (adopt, don't build):** kernels (FlashInfer + HF `kernels` variant
   scheme + `sglang.kernels`/vLLM IR registries), grammar engines (xgrammar/llguidance/outlines),
   checkpoint quant formats (compressed-tensors), transport (Mooncake/NIXL), model definitions
   for the long tail (transformers backend), hardware platforms (entry points in both engines).
2. **The one cross-engine "CSI" that exists is the KV connector.** `KVConnectorBase_V1` +
   `kv_connector_module_path` is what LMCache, Mooncake, NIXL, FlexKV and NVIDIA `kvbm` all
   target; SGLang has a parallel but *different* HiCache-storage/disagg-backend interface. A
   Robotnik KV package should ship one implementation behind both of those, exactly as kvbm does.
3. **KVBM is the proof that a KV-cache manager can be an external Rust package** — but note it
   sits *beside* each engine's block allocator (offload/onboard tiers), not *instead of* it.
   Replacing the in-engine allocator (PagedAttention block pool vs RadixAttention tree) has no
   interface in either engine.
4. **Scheduler is pluggable only in vLLM, and only as "implement `SchedulerInterface`"**
   (`--scheduler-cls`); SGLang's scheduler is 38k LOC of mixins reachable only by hook patches.
   A shared scheduler package would need a new interface in SGLang and a stabilised one in vLLM.
5. **Attention backends are the most churned seam**: vLLM deleted all V0 backends at 0.11.0 and
   still only has one `CUSTOM` slot; SGLang has a decorator registry plus a plugin extension hook.
   The kernel underneath is shareable; the *metadata builder / paged-KV layout contract* is not.
6. **Model definitions are moving away from portability, not towards it**: vLLM RFC #42770 (May
   2026) drops "one model definition for all hardware" and full-graph compile in favour of
   per-backend model code with hand-fused kernels. A "model package" that works across engines
   *and* hardware would swim against vLLM's stated direction; the transformers backend is the
   only cross-engine model format and both engines treat it as fallback.
7. **Interface stability is the real blocker**: hardware plugins version 1:1 with vLLM (vllm-ascend
   0.23 vs vLLM 0.28; tpu-inference co-versioned), vLLM's own guidance is `VLLMPatch` +
   `@min_vllm_version`, and SGLang's platform interface annotates methods `[Active]`/`[Planned]`.
   Any Robotnik interface must be *narrower* than what plugins touch today to survive a release.
8. **HF `kernels` already solved the packaging problem** (torch/CUDA/arch/ABI variant matrix,
   major-version semantics, per-version op namespaces, trusted publishers, lockfiles). Reuse it
   for any Robotnik kernel packages rather than inventing a registry; SGLang consumes it, vLLM
   does not yet (opportunity for a first vLLM adopter PR).
9. **"Remake vLLM from packages" is not realistic** on current evidence: ~10 core subsystems
   (scheduler, KV manager, runner, CUDA graphs, sampler, API server, parsers, LoRA, MM, spec
   decode) have zero shared interface across the two engines and total >300k LOC each; nobody has
   even *proposed* assembling an engine from registry packages.
10. **"Standardise one seam at a time and get engines to adopt it" is realistic and has
    precedent**: KV connector (done, 2025), kernel loading (HF `kernels`, SGLang 2026), platform
    plugins (vLLM 2024–25, SGLang 2026), grammar backends (2024). Order of attack by existing
    seam maturity: KV transfer/offload → kernels → quant config → hardware platform → scheduler
    (vLLM only) → attention metadata → model runner (none).
11. **Pick the seam with an existing two-engine consumer**: the only packages both engines import
    *today* with identical pins are torch, flashinfer, tilelang, humming, tokenspeed-mla,
    quack/cutlass-dsl. A new Robotnik package should look like those (torch-op-level, no engine
    metadata types in its signature) to get adopted without a new interface.
12. **Measure adoption by the pin, not the PR**: mismatched pins (compressed-tensors 0.17 vs 0.18,
    nixl 1.3.2 vs 0.3.0 on TPU, outlines vs outlines_core) show that "both use X" still leaves
    version drift that a registry-first design would have to police.
