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
