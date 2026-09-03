# 02 — Rust components with Python bindings, dual-published (2026-09-03)

## Why not rewrite vLLM or SGLang in Rust

- **The kernel ecosystem is Python-first**: PyTorch, Triton (Python DSL), CuTe DSL (Python), FlashInfer's JIT,
  torch.compile, the HF Kernel Hub. A Rust engine has to either bind all of it or re-create it; candle/burn have
  neither the kernels nor the contributors.
- **Model definitions are the real asset**: 300 + 259 PyTorch model files (research/06). Rewriting them is where
  both projects already spend most of their effort; a Rust engine restarts that at zero.
- **The counter-example already ran**: TensorRT-LLM was the C++ engine and in 2025 added a PyTorch backend to
  regain feature velocity; SGLang and vLLM both integrate it as `trtllm-serve --backend pytorch`. The C++ core lost
  to Python on velocity, not on speed.
- **Where Rust already won** is the layer *around* model execution: HF tokenizers, safetensors, llguidance,
  sglang-router, NVIDIA Dynamo runtime, NVIDIA kvbm (KV block manager), Ruff/Polars/pydantic-core as general
  precedents. Every one is a small Rust core, PyO3/maturin bindings, published as a wheel and a crate.

So the shape is: Python owns model execution; Rust owns the CPU-side data plane; the boundary is coarse-grained
and defined by a contract. That is exactly "rewrite small pieces in Rust, bind to Python, publish both ways".

## The boundary rule

The PyO3 crossing is not free (GIL, object conversion). A component pays for itself only if a whole hot loop lives
on one side. So every Robotnik component contract must be **one call per engine step**, exchanging arrays
(DLPack / Arrow / numpy views), never per-request or per-token Python calls. Rust releases the GIL inside the call;
that alone buys the overlap SGLang's "overlap scheduler" exists to approximate.

## Candidate components, ranked by (contract stability × duplication × GPU-independence)

| # | Component | Today | Contract stability (research/07) | Why it's a good/bad candidate |
|---|---|---|---|---|
| 1 | **Tool-call + reasoning stream parsers** | 52 files in vLLM, 43 in SGLang, all Python string-state-machines | OpenAI-compat boundary, stable | Pure text, zero GPU, duplicated, and the place engine upgrades break silently (research/03). Cheapest win; proves the dual-publish pipeline |
| 2 | **KV block manager + prefix cache** (paged blocks / radix tree / hash-block) | vLLM `kv_cache_manager.py`, SGLang `radix_cache.py` + `memory_pool.py`; NVIDIA kvbm already does it in Rust for vLLM | KV connector seam proven; kv_connector ABC cooling (33→13→6) | Pure data structures over block ids; kvbm is the existence proof; SGLang lacks a kvbm path today |
| 3 | **Scheduler policy** (admission, batching, preemption, priority) | vLLM 3.1k-line scheduler behind existing `--scheduler-cls` seam; SGLang 5.7k-line scheduler with no seam | vLLM scheduler ABC flat at ~6 changes/half | Logic over request metadata only; but couples to #2 and to spec-decode, so contract must version with them |
| 4 | **Request lifecycle / OpenAI server / streaming** | both Python (FastAPI); SGLang router already Rust | stable API | Already partly done (sglang-router, Dynamo frontend); low novelty |
| 5 | **Attention metadata builders** | vLLM separate `MetadataBuilder` class; SGLang inside backend class | attention ABCs accelerating | Builds block tables / cu_seqlens; produces tensors, couples to CUDA-graph protocol. Do last |
| – | Model runner, CUDA-graph capture, torch.compile, model definitions, sampling, kernels | Python / GPU | – | Stay in Python. They *are* torch |
| – | Tokenizer, grammar, KV transfer, router | already Rust/C++ (tokenizers, llguidance/xgrammar, NIXL/Mooncake, sglang-router) | – | Done; register them, don't rewrite |

## Packaging: one component, three artefacts

```
robotnik-registry/components/kv-block-manager/
  manifest.yaml        contract version, capabilities, conformance suite ref, engine shims supported
  crate:  kvbm-core    crates.io, Rust consumers (Dynamo-style runtimes, future Rust engines)
  wheel:  kvbm-py      PyPI via maturin (abi3), PyO3 bindings, DLPack in/out
  shims:  vllm/, sglang/   thin Python adapters generated from the manifest, regenerated per engine release
  conformance/         fixture-driven tests both shims must pass (aqua/devcontainer pattern)
```

Precedents for dual publishing: tokenizers, safetensors, pydantic-core, polars, ruff. Standard maturin workflow.

## Adoption realities

- Both engines already ship Rust-built deps (tokenizers, safetensors, llguidance); SGLang has Rust in-tree
  (router), so its build chain is already used to it. vLLM's CI has CUDA/C++ but not Rust today `[verify]`.
- A Rust component with a stable contract will lag engine features that need scheduler/runner changes (new
  spec-decode modes, new disaggregation shapes). Start where contracts are stable (#1, #2); accept lag on #3.
- The KV connector shows adoption happens when the piece is useful to the engine *without* the engine changing
  its architecture. Design every component to be opt-in behind an existing flag (`--scheduler-cls`,
  `kv_connector_module_path`, parser registry) before asking for a new seam.
- Governance: neither Inferact nor RadixArk will build a neutral component set (research/08). A third party
  can, and both have incentive to consume it if it removes duplicated maintenance.

## First milestone

`robotnik-parsers`: Rust crate + wheel implementing the union of both engines' tool-call and reasoning parsers
behind one streaming interface, with a conformance fixture set mined from both repos' test files, and shims
registered via vLLM's parser registry and SGLang's equivalent. Measures: fixture pass-rate per engine, and a
"parity matrix" page. It is not glamorous, but it exercises manifest → crate → wheel → shim → conformance end to
end on a piece nobody will fight over, before #2 asks for real trust.
