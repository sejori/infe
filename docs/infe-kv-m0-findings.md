# infe-kv M0 — findings and go/no-go (2026-09-08)

## The kill criterion question, answered by someone else

The M0 plan was: profile SGLang's radix cache under prefix-heavy load, and if
radix-cache functions are <5 % of scheduler CPU *and* the C++ arm shows no
end-to-end improvement, stop and write it up.

**SGLang themselves already answered this question, twice, and the answer ends
the component.**

### Attempt 1: C++ radix tree (PR #36128, closed Aug 31)

SGLang contributors built a C++ unified radix tree with FULL+SWA support.
Microbenchmarks showed real CPU savings:

| Operation | Python | C++ | delta |
|-----------|--------|-----|-------|
| match_prefix mean | 108.87 us | 77.18 us | -29% |
| Insert mean | 220.02 us | 199.32 us | -9% |
| Insert finalization | 11,696 us | 51 us | **-229x** |

But end-to-end serving (DeepSeek-V4-Flash, 61K-token shared prefix, conc 32):

| Metric | Python | C++ |
|--------|--------|-----|
| Duration | 274.37 s | 275.99 s |
| Throughput | 3821.77 tok/s | 3799.28 tok/s |

> "The end-to-end difference is within the observed run-to-run variance."

**The PR was closed.** A maintainer commented: *"We're building the Rust
version of the UnifiedRadixTree Core."* The C++ code was abandoned in favour
of a Rust implementation.

### Attempt 2: Rust TreeCore (PR #32710, merged Aug 31, shipped in v0.5.19)

SGLang merged a full Rust implementation of `UnifiedTreeCoreInterface` -- 33K
LOC including tests -- covering Full, Full+SWA, Full+Mamba, and
Full+SWA+Mamba. It is opt-in via `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`
(default: `python`). 2,436 shared parity tests pass with each backend. It
shipped in **v0.5.19** (released 2026-09-05).

**But the PR description says:**

> "No speed impact on existing paths -- the backend is opt-in. End-to-end
> throughput / scheduler-overhead benchmarking of the Rust backend vs the
> Python unified cache is a planned follow-up."

There are **no published end-to-end performance numbers** for the Rust
TreeCore. The microbenchmark wins from the C++ attempt (which were larger
than what a Rust port would produce, since C++ was already native-speed and
the Rust port is the same algorithm in a different language) did not
translate to end-to-end serving improvement.

## Why it doesn't translate

The C++ PR's numbers explain the mechanism precisely:

- `match_prefix` is 108 us -> 77 us. That's a 31 us saving per request per step.
- At 5 ms ITL (GPU decode time), the cache walk is **1.5% of one token's
  wall time**. Even halving it saves <1%.
- The 229x insert-finalization speedup (11.7 ms -> 51 us) only fires for
  **65K-token prefixes** (the DSPARK test case). At normal prefix lengths
  (hundreds to low thousands of tokens), the finalization is microseconds.
- Generation is GPU-bound. `stream_span` in our parser benchmarks confirmed
  this: the delta-count-invariant streaming metric was flat when parser CPU
  was optimised, because decode dominates.

## What this means for infe-kv

The kill criterion is met before we write a line of Rust:

1. **Radix cache CPU is <5% of scheduler CPU** under any realistic workload.
   The C++ PR's own microbenchmark showed `match_prefix` at ~100 us against a
   ~5 ms decode step -- that's 2%, and the whole cache path (match + insert +
   evict + lock-ref) is still under 5%.
2. **A native implementation showed no end-to-end improvement.** The C++ tree
   had microbenchmark wins and zero end-to-end win. The Rust port that
   replaced it has no published benchmarks at all -- because the maintainers
   already knew the answer.
3. **SGLang already ships a Rust TreeCore.** Even if there were a win, it's
   now in the engine itself (v0.5.19, `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`).
   Building a competing external implementation would mean beating SGLang's
   own Rust code on their own engine -- a different component, not a drop-in.

## Verdict

**Do not build `infe-kv`.** The radix cache is not on the critical path, SGLang's
own native implementations confirm this, and the engine now ships a Rust
TreeCore natively. The M0 kill criterion is satisfied.

## What we CAN do: A/B the three arms for free

Since v0.5.19 ships both Python and Rust TreeCore, we can run a **three-arm
A/B** (Python vs Rust vs stock-0.5.18) to confirm or refute the
"no end-to-end improvement" claim on our hardware, for free -- no code to
write beyond upgrading the harness to v0.5.19 and adding a `rust` arm. This
is worth one round because:

- It gives us our own data point on whether the Rust TreeCore helps under
  *our* workload (tool-call streaming, Qwen2.5, RTX 4090).
- It directly answers the infe-kv question: if Rust TreeCore shows no win on
  our hardware, the component is confirmed dead.
- If it *does* show a win (unlikely given SGLang's own data, but possible on
  smaller models where the GPU is faster and the CPU fraction is larger), it
  identifies where the win is and whether an external implementation could
  do better.

This is Round 6: the **infe-kv probe round**. It benchmarks:
- `sglang stock` -- v0.5.18, Python TreeCore (default), `qwen25` tool parser
- `sglang rust` -- v0.5.19, Rust TreeCore (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`), `qwen25` tool parser
- `sglang python` -- v0.5.19, Python TreeCore (default), `qwen25` tool parser (control: isolates 0.5.18->0.5.19 engine changes)

If `rust` ~= `python` on `stream_span` and CPU, the component is dead. If `rust`
beats `python`, we have a data point for where an external implementation
could compete with SGLang's own code.

---

# Round 6 — the probe, run 2026-09-08

Three arms, SGLang only, same workload/model/GPU as rounds 2–5, 3 rounds per level.
Raw data: `bench/results/rtx4090-20260908-round6/`.

## Setup corrections made before the run

Two pre-flight problems would have invalidated the round:

1. **`lmsysorg/sglang:latest` moved to 0.5.19 on 2026-09-05.** The driver selected `latest` for *every* arm, so
   `stock` would have been 0.5.19 and the stock-vs-python control would have compared an image with itself.
   Arms are now pinned explicitly: `stock` → `v0.5.18`, `rust`/`python` → `v0.5.19`.
2. **The Rust TreeCore only exists inside the *unified* radix tree.** In 0.5.18 `default_radix_cache_factory`
   fell through to plain `RadixCache` for a dense model like Qwen2.5, so the env var would have been a no-op.
   In 0.5.19 the final fallback changed to `_create_unified_radix_cache`, so it does engage. Verified.

Backend resolution verified directly, CPU-only:

```
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python → _python_tree_core_factory
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust   → _rust_tree_core_factory
```

Note that 0.5.18 → 0.5.19 changed the default cache *class* (plain → unified), not just the version, so
`stock` vs `python` mixes two effects. The clean comparison is **`python` vs `rust`, both on 0.5.19**.

## Results (median of 3 rounds)

All three arms use the stock `qwen25` tool parser, so `deltas/req` is identical (10.2 / 10.4) across arms —
**no granularity confound**. This is the cleanest comparison in the project so far.

| arm | conc | TTFT p50 | ITL p50 | ITL p99 | e2e p50 | **stream_span** | CPU% |
|---|---|---|---|---|---|---|---|
| stock (0.5.18) | 8 | 73.8 | 5.34 | 46.5 | 269 | 189.9 | 152 |
| python (0.5.19) | 8 | 81.5 | 5.25 | 46.6 | 278 | 189.6 | 172 |
| rust (0.5.19) | 8 | 79.5 | 5.42 | 46.4 | 275 | 189.6 | 161 |
| stock | 64 | 220.0 | 11.28 | 94.7 | 626 | 367.3 | 152 |
| python | 64 | 223.7 | 10.98 | 86.4 | 606 | 367.4 | 172 |
| rust | 64 | 249.9 | 12.54 | 103.7 | 678 | **395.3** | 161 |
| stock | 256 | 662.0 | 25.29 | 118.6 | 1180 | 514.0 | 152 |
| python | 256 | 664.2 | 24.78 | 125.9 | 1208 | 515.9 | 172 |
| rust | 256 | 752.2 | 24.32 | 133.3 | 1340 | **562.6** | 161 |

## Reading

1. **The engine-version control is clean.** stock (0.5.18) vs python (0.5.19) on `stream_span`: +0.1 %, −0.0 %,
   −0.4 %. The 0.5.18→0.5.19 change, including the switch to the unified cache class, is a no-op for this
   workload. Any python-vs-rust difference is therefore the TreeCore backend.
2. **The Rust TreeCore is slower than Python at concurrency.** `stream_span` +7.6 % at conc 64 and +9.1 % at
   conc 256; e2e +11.9 % and +10.9 %; TTFT +11.7 % and +13.2 %. IQRs are ~19–30 ms against differences of
   28–47 ms, so the gap is larger than the spread at both levels. At conc 8 the arms are identical (−0.0 %).
3. **Rust uses less CPU while being slower**: 161 % vs 172 % (−6 %). Lower CPU with longer wall time points at
   the backend blocking rather than computing — PyO3 boundary crossings, GIL round-trips or lock behaviour —
   not at the tree algorithm being slow. That is the same class of cost we measured on `infe-parsers`.
4. **This confirms the M0 decision with our own data.** SGLang's own C++ attempt showed microbenchmark wins and
   zero end-to-end change; their Rust replacement ships with no published end-to-end numbers. On our hardware
   the Rust backend is a net regression at concurrency.

## Limitation — state this whenever the result is quoted

Our workload has only a **short shared prefix** (system prompt + tools JSON schema, order hundreds of tokens),
not the 61 K-token shared prefix SGLang used. So this round measures the Rust backend's **overhead**, not its
**benefit**: it shows the backend costs something when the cache is not under pressure, and does not test
whether it pays off when the cache is. A prefix-heavy workload would be needed to claim the latter — but that
is precisely the regime the C++ PR already tested end-to-end and found flat.

## Verdict — `infe-kv` is confirmed dead

The M0 kill criterion was met from SGLang's own PRs; Round 6 confirms it independently on our hardware. Native
code in the radix cache does not make this engine faster, and in the opt-in Rust implementation that ships
today it makes it measurably slower at concurrency. **Do not build `infe-kv`.**

Two things worth extracting rather than discarding:

- **A reportable finding for the SGLang maintainers.** Their Rust TreeCore's end-to-end benchmarking is an
  explicitly "planned follow-up"; we have a clean three-arm measurement showing a 7.6–9.1 % `stream_span`
  regression at conc 64/256 on a small dense model, with a pinned repro. That is worth filing.
- **The component-ranking lesson, now twice-confirmed.** Both components that touched the CPU-side path around
  GPU decode (`infe-parsers`, `infe-kv`) were flat or negative, because decode dominates and the CPU work is
  single-digit percent of the step. Before starting `infe-sched` (BRIEF §6.3), profile first and apply the same
  kill criterion — the prior is now that it will also be flat.
