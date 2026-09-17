# Experimental batch-result interface, version 1

This is the persistent **Python control**, not a live SGLang plugin or a Rust
implementation. It makes a candidate ownership boundary executable before paying
for integration. The tuned profile did not pass the native-port gate; see
[the findings](../../docs/batch-result-findings.md).

## Ownership and lifecycle

One `PythonBatchStage(capacity)` owns request histories, completion state and output
cursors. The caller serializes admissions, observed cancellations, batch steps and
release acknowledgements. Admission takes a validated `RequestSpec` and returns a
64-bit handle: a 32-bit generation above a 32-bit slot. Never reuse the same handle
for a new request. Generation overflow fails; late outputs and acknowledgements
from older generations cannot affect new occupants.

`advance(step, handles, token_ids, offsets, out)` accepts one batch. Step numbers
are unsigned 64-bit and strictly increase. Handles are uint64; token IDs and offsets
are uint32. Offsets have N+1 entries, begin at zero, end at the token count and
specify zero or one token per request. Duplicate and never-admitted handles fail.
A future native binding would validate contiguous buffers of these widths once
per call; this Python implementation also accepts integer sequences. The arrays
use host byte order and are an in-process interface, not a serialized wire format.

Returned columns contain handles, token offsets/IDs, completion reasons, matched
stop tokens, release handles and ignored handles. Reasons are running=0, length=1,
token=2, cancelled=3. The uint64 maximum means no matched token. Cancelled requests
emit a terminal record with zero new tokens at the next step, including an empty
step; these records precede ordinary batch rows. Other records follow input order.
A cancellation means the engine has already observed a terminal cancellation; it
does not define HTTP abort timing or precedence over an unobserved engine event.

Each accepted token is emitted once. EOS/stop IDs remain in the owned history, as
in SGLang's internal request state; downstream detokenization must handle text
trimming. Token stopping wins a same-step length tie. `ignore_eos` suppresses
explicit stop-token checks too, matching the pinned engine. A terminal request
emits exactly one release notification. The engine still performs cache/allocator
side effects, then calls `acknowledge_release(handle)` to make the slot reusable.
Repeated old acknowledgements are harmless. Early release fails.

`BatchOutput` is mutable and borrowed until the next call that reuses it. Consumers
must finish reading before reuse, or explicitly copy it. `snapshot()` copies state
for diagnostics; it must not become a second authoritative history in a live
adapter. Reusing Python arrays does not promise zero allocation. Input validation
completes before state/output mutation. Allocation failure is not recoverable as an
atomic transaction; a native implementation must reserve capacity first.

## Deliberate scope

Only valid-token, ordinary generation is represented. Admission rejects declared
optional features. An engine adapter must detect and reject unsupported features;
passing an empty `features` list is not automatic engine capability detection.
String/regex stops, grammar, speculative multi-token acceptance, beam search,
logprobs, disaggregated execution, Mamba state, text reconstruction and cache
ownership are outside this interface. Invalid token IDs are rejected; SGLang's
invalid-token repair behavior is not replicated.

The current SGLang result loop and output streamer read and mutate `Req.output_ids`
and many other request fields. Simply calling this stage and copying its history
back would preserve duplicate state and per-request Python work. A useful live
adapter would move all consumers of the selected state to batch records and retain
the engine's required finish/cache side effects. That integration has not been
claimed or installed.

## Verification

From the repository root:

```sh
python3 -m unittest discover -s bench/harness -p 'test_*.py'
python3 bench/harness/batch_result_replay.py
```

The replay generator is deterministic (seed 20260917), with 640 request lifetimes,
1,280 reordered steps, cancellation, terminal overshoots and slot reuse. These are
synthetic events, not a captured GPU execution trace. `--write-events` exports the
fixture and `--events` replays it. The event hash is recorded with results.

Use the same image digest as the benchmark manifest for the engine oracle:

```sh
docker run --rm -v "$PWD:/repo:ro" -w /repo --entrypoint python3 \
  lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9 \
  bench/harness/batch_result_replay.py --engine-oracle
```

No GPU is required. This calls SGLang's actual `Req.update_finish_state` on minimal
request objects and compares histories and completion reasons. It is token-state
conformance, not end-to-end engine, transport or cancellation-race parity. The
oracle intentionally creates a separate reference history for testing only.
