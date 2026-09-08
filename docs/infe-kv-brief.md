# `infe-kv` — component brief (handover, 2026-09-07)

Second component after `infe-parsers`. Read `docs/review-2026-09-04.md` §"Round 5" first: the parser component
reached full parity on both engines and produced **no** performance win. This brief is written to avoid
repeating that outcome.

Seam facts below were verified on 2026-09-07 inside the pinned images (`vllm/vllm-openai:latest` = 0.28.0,
`lmsysorg/sglang:latest` = 0.5.18), not from documentation.

---

## 0. The lesson from `infe-parsers`, stated once

The parser rewrite was correct, drop-in, dual-published — and flat. It failed criterion (3) because **parsing
was never on the critical path**: it sits on the SSE serialisation path, is a small fraction of a 100–190 %
whole-container CPU total, and `stream_span` (the delta-count-invariant streaming metric) did not move.

Nobody measured that *before* writing 1,300 lines of Rust.

**Therefore M0 of this component is a measurement, not an implementation, and it has a kill criterion.**
If the KV/prefix-cache path is not a meaningful share of per-step CPU under a prefix-heavy workload, say so and
stop. That is a successful outcome, delivered in days rather than weeks.

---

## 1. What changed since BRIEF §6.2 was written

The original brief said: vLLM KV-connector is the proven path; SGLang "has no seam, expect a documented gap".
**Both halves of that are now wrong.**

### SGLang 0.5.18 has a first-class public registry — this is the primary target

```python
# sglang/srt/mem_cache/registry.py
RadixCacheFactory = Callable[[TreeCacheBuildContext], BasePrefixCache]

def register_radix_cache_backend(name: str, factory: RadixCacheFactory) -> None: ...
def registered_radix_cache_backends() -> list[str]: ...
```

- Selected with `--radix-cache-backend <name>`; the flag **accepts only registered names**, and the help text
  says "Name of a radix-cache backend previously registered via `register_radix_cache_backend`".
- Registration happens at import time, so the same launcher trick as the SGLang parser shim applies
  (`python -m infe_kv_sglang.launch -- <sglang args>`), because the flag is validated during arg parsing.
- **A working third-party reference implementation already exists**: FlexKV, at
  `sglang/srt/mem_cache/storage/flexkv/__init__.py`, calls `register_radix_cache_backend("flexkv", _flexkv_factory)`.
  Read it before writing anything.
- `TreeCacheBuildContext` gives the factory everything: `server_args`, `params` (`CacheInitParams`),
  `is_hybrid_swa`, `is_hybrid_ssm`, `enable_hierarchical_cache`, `disable_radix_cache`,
  `effective_chunked_prefill_size`, `tp_worker`, `model_config`, `tp_size`, `tp_rank`, `tp_group`, `is_dsa`.

`BasePrefixCache` (in `mem_cache/base_prefix_cache.py`) is the contract to implement. Abstract methods:
`reset`, `match_prefix(MatchPrefixParams) -> MatchResult`, `cache_finished_req(req, is_insert=True)`,
`cache_unfinished_req(req)`, `evict(EvictParams) -> EvictResult`, `inc_lock_ref(node) -> IncLockRefResult`,
`dec_lock_ref(node, ...) -> DecLockRefResult`, `evictable_size()`. Plus optional hooks with defaults:
`supports_fast_match_prefix`, `resolve_node_handle`, `root_node_handle`, `is_backuped`, `is_root`,
`get_last_hash_value`, `get_prefix_hash_values`, `init_metrics_collector`, `update_eviction_metrics`,
`release_host_resources`. Params/results are typed dataclasses (`MatchPrefixParams`, `InsertParams`,
`EvictParams`, `IncLockRefResult`, `MatchResult` NamedTuple…), which makes the PyO3 boundary tractable.

### There is already a native (C++) radix tree in SGLang — know what you are competing with

`mem_cache/cpp_radix_tree/` (`tree_v2.cpp`, `tree_v2_impl.h`, `tree_v2_binding.cpp`) with a Python wrapper
`mem_cache/radix_cache_cpp.py::RadixCacheCpp`. It is **not** the default — it is gated behind the env var
`SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` in `default_radix_cache_factory`. Consequences:

- The **stock baseline in our benchmark is the Python `RadixCache`**, so a Rust backend is compared against
  Python by default. Good.
- But SGLang maintainers already built the native version and left it experimental and off. **Find out why
  before building a third one.** If it is off because it wasn't faster, that is the answer to this whole
  component and it is available for the cost of a git-log read.
- It also gives a free third arm: `stock` vs `cpp` (`SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=1`) vs `infe`. That
  triangulates "is native code worth it here" independently of our implementation quality.

### vLLM 0.28: the allocator is still not pluggable

- `KVConnectorFactory.register_connector(name, module_path, class_name)` exists and `--kv-transfer-config`
  selects it — but `KVConnectorBase_V1` is about **moving KV** (`start_load_kv`, `wait_for_layer_load`,
  `save_kv_layer`, `wait_for_save`, `get_finished`, `request_finished_all_groups`, `aggregate`), i.e.
  offload/disaggregation/cross-request reuse. It is not the block allocator or the prefix tree.
- The in-engine allocator is hard-constructed: `vllm/v1/core/sched/scheduler.py:277` →
  `self.kv_cache_manager = KVCacheManager(...)`. **No flag, no registry.** Replacing it needs an upstream RFC,
  as recorded in BRIEF §6.2.

**So: SGLang is the primary target for `infe-kv`, and vLLM is out of scope for the allocator** until either an
RFC lands or the scope is narrowed to a KV connector (a different component — call it `infe-kv-connector` —
where NVIDIA's `kvbm` is the incumbent to benchmark against, not to duplicate).

---

## 2. M0 — measure first, with a kill criterion (do this before writing Rust)

**Question: what fraction of per-step CPU does SGLang's radix cache actually consume under prefix-heavy load?**

1. **Build a prefix-heavy workload.** The current `bench/harness/e2e_tool_stream.py` sends unrelated tool-call
   prompts with no shared prefix, so the radix cache is nearly idle — it would measure nothing. Add a workload
   with a long shared system prompt (2–8 k tokens) and many short continuations, plus a multi-turn mode that
   replays a growing conversation. Prefix hit-rate should be high and reported.
2. **Profile the stock Python path.** `py-spy record`/`py-spy top` on the SGLang scheduler process under that
   load, or `cProfile` around the cache calls. Attribute time to `match_prefix`, `cache_finished_req`,
   `cache_unfinished_req`, `evict`, `inc/dec_lock_ref`.
3. **Get the three-arm baseline for free**: stock (Python) vs `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=1` (C++) on
   the same workload. If C++ ≈ Python end-to-end, a Rust backend will also be ≈ Python, and the component
   should stop there.
4. **Read the git history** of `cpp_radix_tree/` and `radix_cache_cpp.py` for why it is still experimental.

**Kill criterion — write it down before running:** if the radix-cache functions account for **< 5 % of scheduler
CPU** under the prefix-heavy workload, *and* the C++ arm shows no end-to-end improvement over Python, stop and
write it up. Do not build `infe-kv`. Go to `infe-sched` (BRIEF §6.3) or reconsider the component ranking.

M0 deliverable: a short `docs/infe-kv-m0-findings.md` with the profile, the three-arm numbers, and a go/no-go.

---

## 3. M1 — implement, only if M0 says go

Shape follows `infe-parsers` exactly; that pipeline is proven and should not be redesigned.

```
crates/infe-kv/              Rust core: radix tree over token-block ids, lock refs, eviction policy
python/infe-kv/              PyO3 bindings (maturin, abi3)
shims/sglang/infe_kv_sglang/ BasePrefixCache subclass + register_radix_cache_backend("infe", factory)
                             + launch.py (registration must precede arg parsing)
registry/infe-kv/manifest.yaml
conformance/fixtures/kv/     replayed operation traces (see §4)
```

**Boundary rule (BRIEF §5.1) matters far more here than it did for parsers.** `match_prefix` is called per
request per step. Crossing PyO3 with Python objects per call would lose immediately. Design for:
- token ids and block ids crossing as buffers (numpy/DLPack views), never Python lists;
- `MatchResult` returned as primitives + a buffer, not a nested object graph;
- node handles as opaque integer ids, not Python objects (`resolve_node_handle` exists precisely for this).

Scope the first cut narrowly: **plain `RadixCache` equivalent only.** Do not attempt hybrid SWA, SSM/Mamba,
hierarchical/HiCache, LMCache, DSA or disaggregation-backup variants — the default factory has a branch for each
and they are separate implementations. The manifest must declare them unsupported so the factory falls back.

## 4. Correctness bar — higher than parsers, and it is not optional

A parser bug produces a malformed delta. **A KV bug produces silently wrong tokens**, because a bad prefix match
serves another request's cached state. So:

- **Operation-trace conformance.** Instrument the stock cache to record `(op, args, result)` for a full run,
  then replay the trace against the Rust implementation and assert identical results — same match lengths, same
  evictions, same lock refs. This is the KV analogue of the parser fixtures and is the primary safety net.
- **Output equality under prefix reuse.** Same prompts, greedy, fixed seed, `stock` vs `infe`: token-identical
  outputs. Use `--enable-deterministic-inference` if available on the pinned build.
- **A cache-poisoning test**: two requests with a long shared prefix that then diverge must not contaminate each
  other. Assert on generated text, not on internal state.
- **Report prefix hit-rate parity** alongside latency; a "faster" cache that hits less is just a smaller cache.

## 5. Measurement — reuse the harness, and use the invariant metric

`bench/harness/` already has what is needed: `run_ab_docker.sh` (arms, health wait, teardown),
`e2e_tool_stream.py` (add the prefix-heavy mode), `cpu_sampler.py` (cgroup-wide, verified at 101 % on a
single-threaded busy loop), `summarize_ab.py`.

Two things carried over from round 5, both already fixed but easy to regress:
- **Use `stream_span`, not ITL.** Mean ITL ≈ streaming_time / delta_count, so any change in delta granularity
  moves ITL without anything getting faster. `stream_span` (sum of inter-token gaps) is invariant.
- **CPU must be cgroup-wide.** Sampling the entrypoint PID only produced a fake "28 % → 16 %" win in round 4
  that reversed to 184 % → 186 % once all container processes were counted.

For this component the headline metrics are: **TTFT under prefix reuse** (where a faster match should show),
**scheduler CPU %**, **prefix hit-rate**, and **e2e/stream_span as the no-regression guard**. Sample counts were
only 5–13 per run in round 5 — drop the sampler interval to 0.25 s or lengthen runs before quoting CPU.

## 6. Definition of done

Same four criteria as BRIEF §2, scored honestly:

1. **Drop-in** — `--radix-cache-backend infe` via the launcher, no SGLang fork.
2. **Parity** — operation-trace replay matches stock; token-identical outputs; prefix hit-rate ≥ stock.
3. **Measured improvement** — on `stream_span`/TTFT/scheduler-CPU with non-overlapping IQRs, against **both**
   the Python and the C++ arm.
4. **Published** — crate + wheel + manifest + conformance in CI.

(1), (2) and (4) without (3) still ships, and the report says so plainly — as it did for `infe-parsers`.

## 7. First three commands

```bash
# 1. Read the working reference implementation of the exact seam you are targeting
docker run --rm --entrypoint bash lmsysorg/sglang:latest -c \
  'cat /sgl-workspace/sglang/python/sglang/srt/mem_cache/storage/flexkv/__init__.py'

# 2. Read the contract you must implement
docker run --rm --entrypoint bash lmsysorg/sglang:latest -c \
  'cat /sgl-workspace/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py'

# 3. Find out why the C++ radix tree is still experimental — this may end the component early
git -C <sglang-checkout> log --oneline -- python/sglang/srt/mem_cache/cpp_radix_tree/
```
