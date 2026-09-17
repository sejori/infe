# Batch-result ownership experiment (2026-09-17)

First establish whether a worthwhile CPU stage remains with decode CUDA graphs
through batch size 256. Use the same pinned image/model, TP=1 and RTX 4090 as
the scheduling investigation, at short c8 and c256. Run two balanced fresh
stock/probe pairs and two separate traces. This is diagnostic calibration,
not a Rust performance claim. Fine-grained per-request hooks can perturb the
workload: report that overhead, inclusive nesting and GPU/API overlap.

Attribute result normalization, completion checks, finish/cache actions,
output accumulation/packing and sender handoff separately. Inspect existing
native engine paths before proposing a duplicate implementation. No generated
content, external request data, machine IDs or raw profiler reports are public.

The first implementation deliverable is a versioned batch interface, persistent
Python state owner and deterministic replay tests. Support ordinary generation
with token-based stopping, length limits, cancellation and overlap overshoots.
Reject unsupported features explicitly. Admission establishes a request lifetime;
generation-stamped handles and ordered steps prevent stale outputs reaching a
reused slot. Completion and release notifications are exactly once. Do not keep
two authoritative histories or perform per-token FFI calls.

Check semantics against the pinned engine's actual completion logic, not only
a second version of our implementation. Include reordering, invalid atomic
batches, EOS/length ties, cancellation and release acknowledgement. Separate
the stage's batch output records from engine cache-release ownership.

Proceed to live Python-control/Rust comparison only if a bounded stage plausibly
offers approximately 5% serving improvement after packing and handoff costs.
If the target is smaller, retain the interface/replay evidence and stop before
a native port. If the target lies elsewhere (e.g. an existing native serving
path), report that and test reuse before writing another implementation.
Any later live performance claim requires independent counterbalanced sessions,
output parity, CPU per token, throughput and median/tail latency measurements.

Completed: [findings and gate decision](batch-result-findings.md). The Python
control/replay and tuned diagnostic are implemented; a live native port is gated
off for this boundary and workload.
