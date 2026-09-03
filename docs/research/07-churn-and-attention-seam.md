# 07 — How fast the engines rework their internals, and where an attention adapter seam could sit

Measured 2026-09-03 from full git history (blobless clones of vllm-project/vllm @ 20,817 commits and
sgl-project/sglang @ 17,698 commits). Method: `git log --follow` on interface files, add/delete diff-filters on
attention backend directories, tags per year. Raw scripts were run ad hoc; numbers are reproducible from the repos.

## 1. Release cadence

| | 2023 | 2024 | 2025 | 2026 (to 3 Sep) |
|---|---|---|---|---|
| vLLM tagged releases | 15 | 22 | 21 | 25 |
| SGLang tagged releases | – | 60 | 63 | 17 |

## 2. Interface-file churn (commits touching the file, by half-year)

| File | Role | Total | 2025H1 | 2025H2 | 2026H1 | 2026H2 (2 mo) | Trend |
|---|---|---|---|---|---|---|---|
| vllm `v1/attention/backend.py` (moved Jan 2026) | attention backend ABC | 133 | 14 | 32 | 41 | 21 | **accelerating** |
| vllm `kv_connector/v1/base.py` | KV connector ABC | 60 | 8 | 33 | 13 | 6 | **cooling** |
| vllm `v1/core/sched/interface.py` | scheduler ABC | 24 | 6 | 7 | 7 | 4 | flat, low |
| vllm `platforms/interface.py` | hardware platform ABC | 185 | 57 | 56 | 42 | 9 | cooling |
| vllm `v1/core/kv_cache_manager.py` | KV manager impl | 116 | 49 | 26 | 14 | 16 | – |
| vllm `v1/core/sched/scheduler.py` | scheduler impl | 352 | 89 | 109 | 91 | 51 | ~15–25/mo |
| vllm `v1/worker/gpu_model_runner.py` | model runner impl | 806 | 168 | 327 | 200 | 72 | ~35–55/mo |
| sglang `layers/attention/base_attn_backend.py` | attention backend ABC | 52 | 15 | 6 | 7 | 15 | **re-accelerating** |
| sglang `disaggregation/base/conn.py` | KV transfer ABC | 43 | 6 | 6 | 20 | 11 | rising |
| sglang `mem_cache/memory_pool.py` | KV pool impl | 314 | 59 | 66 | 80 | 66 | ~30/mo |
| sglang `model_executor/forward_batch_info.py` | batch metadata struct | 288 | 46 | 67 | 71 | 48 | ~20/mo |
| sglang `managers/scheduler.py` | scheduler impl | 1,059 | 170 | 242 | 340 | 170 | **~60–85/mo** |
| sglang `model_executor/model_runner.py` | model runner impl | 910 | 147 | 211 | 209 | 121 | ~35–60/mo |

## 3. Subsystem churn, commits per quarter (2025Q3 → 2026Q3)

| Dir | 25Q3 | 25Q4 | 26Q1 | 26Q2 | 26Q3* |
|---|---|---|---|---|---|
| vllm `v1/attention/backends` | 171 | 151 | 165 | 163 | 150 |
| vllm `v1/worker` | 241 | 266 | 302 | 245 | 254 |
| vllm `distributed/kv_transfer` | 61 | 89 | 84 | 154 | 112 |
| vllm `layers/quantization` | 177 | 206 | 192 | 190 | 114 |
| sglang `layers/attention` | 103 | 158 | 173 | 291 | 318 |
| sglang `managers` | 243 | 339 | 276 | 499 | 346 |
| sglang `mem_cache` | 110 | 123 | 109 | 270 | 309 |
| sglang `disaggregation` | 51 | 78 | 107 | 181 | 174 |
*26Q3 is two months.

## 4. Attention backend file turnover

| | files added since 2024 | files deleted | live now | biggest single events |
|---|---|---|---|---|
| vLLM | 80 | 28 | 52 | `Remove V0 attention backends` 2025-09-21 (#25351, 10 files); `Restructure attention: move files` 2026-01-09 (#31916); `Remove tree attention` 2026-05-08 |
| SGLang | 208 | 105 | 37 top-level + subdirs (linear 20, nsa 10, dsa 10, dsv4 7, mamba 5…) | **RFC #29630 kernel migration, July 2026: 93 files deleted/relocated in one week** (#30044, #30789, #30792, #30793, #30795, #31582) |

SGLang `--attention-backend` choices today (27): triton, torch_native, flex_attention, dsa, nsa, dsv4, compressed,
cutlass_mla, fa3, fa4, flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, dual_chunk_flash_attn,
hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu (+ draft/deterministic lists).

## 5. The two ABCs side by side

**vLLM `AttentionBackend`** (v1/attention/backend.py) is a *capability matrix + factory*: `get_impl_cls`,
`get_builder_cls` (metadata builder is a separate class), `get_supported_kernel_block_sizes`, and ~25 `supports_*`
classmethods (head size, dtype, kv-cache dtype, block size, sink, alibi_sqrt, mm_prefix, sliding window, non-causal,
batch invariance, kv connector, pcp, dcp, attn type, compute capability, per-head quant scales), `is_mla`, `is_sparse`,
`is_ssm`, `supported_kv_cache_layouts`, `customize_spec`, `validate_configuration`. Three-way split:
Backend (declares) / MetadataBuilder (engine batch → kernel metadata) / Impl (kernel call).

**SGLang `AttentionBackend`** (base_attn_backend.py) is a *runtime object bound to ModelRunner*: `init_forward_metadata`
(+ `_out_graph`, `_in_graph`, breakable-capture/replay variants), `init_cuda_graph_state`, `forward_decode` /
`forward_extend` / `forward_mixed`, `verify_mask` (spec decode), `shared_read_ends`, `get_indexer_metadata`,
`support_triton`. Metadata construction and kernel launch live in one class, coupled to `ForwardBatch`.

**Both ship a FlexAttention backend** (vLLM 2025-06-07 #16078; SGLang 2025-09-19 #9947): attention *algorithm* written
as Python `score_mod` / `mask_mod` and compiled by `torch.compile` to Triton. This is the "algorithm separated from
hardware kernel" abstraction already in production in both engines, used for sparse/experimental patterns and as a
portable fallback; it is not the fast path.

**SGLang `sglang.kernels` (July 2026, RFC #29630)**: `spec.py` (`KernelSpec`, `KernelBackend`, `FormatSignature`,
`CapabilityRequirement`, `PlatformInfo`), `registry.py` (process-wide `KernelRegistry` + `register_kernel()`),
`selector.py` (heuristic `select_kernel()`), `fused_op.py` (`BaseFusedOp`: "per-operator multi-backend contract"),
18 operator groups including `attention`, `moe`, `sampling`, `kvcache`, `quantization`. This is an in-tree,
per-operator, multi-backend adapter registry — exactly the shape the question asks about — two months old and
SGLang-only. Ref: python/sglang/kernels/README.md.

## 6. Reading

1. **Three layers already exist in both engines, and only the middle one is unshared.**
   - Algorithm (model-defined: MHA/GQA/MLA/DSA/NSA/linear/Mamba/sliding window/sinks). Abstracted by FlexAttention
     in both engines when speed doesn't matter.
   - Engine backend = metadata builder + KV layout + CUDA-graph protocol + capability matrix. Engine-specific,
     coupled to the batch struct and KV pool. This is where the churn is.
   - Kernel = FlashInfer / FA3 / FA4 / Triton / trtllm-gen / AITER / CuTe DSL. Already registry-shaped
     (PyPI pins identical across engines per research/06; HF Kernel Hub; `sglang.kernels` registry; FlashInfer JIT).
2. **Churn is layered the same way.** Kernel-facing contracts are cooling (vLLM kv_connector base 33 → 13 → 6
   commits per half; vLLM scheduler ABC flat at ~6/half). Engine-backend ABCs are accelerating (vLLM 14 → 32 → 41
   → 21-in-2-months; SGLang 6 → 7 → 15-in-2-months). Runner/scheduler *implementations* run at 35–85 commits/month.
   An adapter interface has to live where the numbers are small.
3. **So: yes, but target the kernel-op contract, not the backend class.** A cross-engine spec that unifies SGLang's
   `KernelSpec/FormatSignature/CapabilityRequirement/PlatformInfo` with vLLM's `supports_*` matrix +
   `supported_kv_cache_layouts` + `get_supported_kernel_block_sizes` is mostly a rename exercise — both sides already
   declare the same facts. A package = kernel(s) + that declaration. Each engine's backend class becomes a thin
   generated shim over its own batch struct. The residue that stays engine-specific: batch metadata construction
   (ForwardBatch vs CommonAttentionMetadata), CUDA-graph capture protocol, spec-decode verify masks.
4. **The V0→V1 lesson applies.** vLLM deleted every V0 backend in one commit; SGLang relocated 93 files in one week.
   Out-of-tree packages survive only if the contract is declarative data + conformance test (aqua/devcontainer
   pattern), not a Python base class to subclass.
5. **HF Kernel Hub already has the packaging half** (build variants keyed by torch/CUDA/arch); SGLang already
   consumes it for FA3. What it lacks is the capability/format declaration and the conformance test. That gap is
   the size of a Robotnik package spec.

## Implications for Robotnik

- The attention seam is viable, at the kernel-op layer, as a declarative manifest + conformance suite: formats
  (KV layout, dtypes, block sizes), capabilities (sliding window, sinks, MLA, sparse, spec-decode, batch-invariance),
  platform (compute capability, vendor), and a reference test that both engines' shims must pass.
- Start by mapping `sglang.kernels.spec` ↔ vLLM `AttentionBackend.supports_*`; propose the union upstream to both.
  The SGLang RFC is fresh enough that its authors are the natural co-designers.
- Do not define a cross-engine backend *class*. Generate each engine's shim from the manifest; regenerate on every
  engine release (the Renovate loop from research/04 handles this).
- Extend the same manifest shape to the other `sglang.kernels` groups that both engines already pin identically
  (moe, sampling, quantization, kvcache) — attention is the hardest one, not the first one.
- Expect the batch-metadata and CUDA-graph residue to stay per-engine for years; make the manifest explicit about
  which engine features a kernel package does *not* support rather than pretending parity.
