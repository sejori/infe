"""ABI-v1 Python control. Experimental; not an installed SGLang replacement.

All request histories live here. Numeric handles/columns cross the step boundary;
snapshots copy state for diagnostics only. Native execution is intentionally gated
on the tuned serving profile. Token IDs include a matched EOS/stop token: text
trimming remains the downstream detokenizer's job, as in SGLang's Req state.
"""
from array import array
from dataclasses import dataclass, field
from enum import IntEnum

ABI_VERSION = 1
UINT32_MAX = (1 << 32) - 1
NO_TOKEN = (1 << 64) - 1


class Reason(IntEnum):
    RUNNING = 0
    LENGTH = 1
    TOKEN = 2
    CANCELLED = 3


def uint(value, limit=UINT32_MAX):
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError("Expected an unsigned integer within the ABI range")
    return value


@dataclass(frozen=True)
class RequestSpec:
    max_new_tokens: int
    vocab_size: int
    stop_tokens: frozenset = frozenset()
    ignore_eos: bool = False
    features: tuple = ()

    def checked(self):
        if not uint(self.max_new_tokens) or not uint(self.vocab_size):
            raise ValueError("Positive generation limit and vocabulary required")
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be boolean")
        if self.features:
            raise NotImplementedError("Optional generation features are outside ABI v1")
        stops = frozenset(uint(t, self.vocab_size - 1) for t in self.stop_tokens)
        return RequestSpec(self.max_new_tokens, self.vocab_size, stops, self.ignore_eos)


@dataclass
class BatchOutput:
    handles: array = field(default_factory=lambda: array("Q"))
    offsets: array = field(default_factory=lambda: array("I", [0]))
    token_ids: array = field(default_factory=lambda: array("I"))
    reasons: array = field(default_factory=lambda: array("B"))
    matched_tokens: array = field(default_factory=lambda: array("Q"))
    release_handles: array = field(default_factory=lambda: array("Q"))
    ignored_handles: array = field(default_factory=lambda: array("Q"))

    def reset(self):
        for value in vars(self).values():
            del value[:]
        self.offsets.append(0)

    def record(self, handle, tokens, reason, matched=None):
        self.handles.append(handle)
        self.token_ids.extend(tokens)
        self.offsets.append(len(self.token_ids))
        self.reasons.append(reason)
        self.matched_tokens.append(NO_TOKEN if matched is None else matched)
        if reason:
            self.release_handles.append(handle)

    def snapshot(self):
        return {key: list(value) for key, value in vars(self).items()}


@dataclass
class _State:
    spec: RequestSpec
    history: list = field(default_factory=list)
    reason: Reason = Reason.RUNNING
    matched: int | None = None
    output_cursor: int = 0
    notified: bool = False
    cancel_pending: bool = False


class PythonBatchStage:
    """Single-owner, ordered event stream; callers serialize control and steps.

    Ordinary generation accepts zero or one token per request per step. A released
    handle may occur in a delayed batch: it is ignored even after slot reuse.
    Future/never-admitted handles, duplicates and malformed batches are errors.
    Validation errors are atomic. Allocation failure is not a recoverable ABI
    transaction; a production adapter must reserve capacity before integration.
    """

    def __init__(self, capacity):
        if not uint(capacity):
            raise ValueError("Positive capacity required")
        self._states = [None] * capacity
        self._generations = [0] * capacity
        self._free = list(reversed(range(capacity)))
        self._cancelled = []
        self._last_step = -1

    def admit(self, spec):
        spec = spec.checked()
        if not self._free:
            raise BufferError("No released request slot")
        slot = self._free[-1]
        generation = self._generations[slot] + 1
        if generation > UINT32_MAX:
            raise OverflowError("Handle generation exhausted")
        self._free.pop()
        self._generations[slot] = generation
        self._states[slot] = _State(spec)
        return (generation << 32) | slot

    def _lookup(self, handle):
        uint(handle, NO_TOKEN)
        slot, generation = handle & UINT32_MAX, handle >> 32
        if slot >= len(self._states) or not generation or generation > self._generations[slot]:
            raise ValueError("Never-admitted handle")
        if generation < self._generations[slot]:
            return None
        return self._states[slot]

    def cancel(self, handle):
        state = self._lookup(handle)
        if state is not None and not state.reason and not state.cancel_pending:
            state.cancel_pending = True
            self._cancelled.append(handle)

    def acknowledge_release(self, handle):
        state = self._lookup(handle)
        if state is None:
            return  # repeated acknowledgement, including an older generation
        if not state.reason or not state.notified:
            raise ValueError("Cannot release a live/unreported request")
        slot = handle & UINT32_MAX
        self._states[slot] = None
        self._free.append(slot)

    def snapshot(self, handle):
        state = self._lookup(handle)
        if state is None:
            return None
        return {"tokens": list(state.history), "reason": int(state.reason),
                "matched": state.matched, "output_cursor": state.output_cursor,
                "notified": state.notified, "cancel_pending": state.cancel_pending}

    def advance(self, step, handles, token_ids, offsets, out=None):
        uint(step, NO_TOKEN)
        if step <= self._last_step:
            raise ValueError("Steps must increase; retries must not recommit tokens")
        if len(offsets) != len(handles) + 1 or not offsets or offsets[0] != 0 or offsets[-1] != len(token_ids):
            raise ValueError("Malformed packed offsets")
        for offset in offsets:
            uint(offset)
        if any(b < a or b - a > 1 for a, b in zip(offsets, offsets[1:])):
            raise ValueError("ABI v1 permits at most one token per request per step")
        if len(set(handles)) != len(handles):
            raise ValueError("Duplicate handle in batch")
        states = [self._lookup(handle) for handle in handles]
        for token in token_ids:
            uint(token)
        for i, state in enumerate(states):
            if state is not None and offsets[i + 1] > offsets[i]:
                uint(token_ids[offsets[i]], state.spec.vocab_size - 1)

        # No state/output mutation before the entire batch has been validated.
        out = BatchOutput() if out is None else out
        out.reset()
        self._last_step = step
        for handle in self._cancelled:
            state = self._lookup(handle)
            if state is not None and not state.reason:
                state.reason, state.notified = Reason.CANCELLED, True
                state.cancel_pending = False
                out.record(handle, (), Reason.CANCELLED)
        self._cancelled.clear()
        for i, (handle, state) in enumerate(zip(handles, states)):
            if state is None or state.reason:
                out.ignored_handles.append(handle)
                continue
            if offsets[i] == offsets[i + 1]:
                continue
            token = token_ids[offsets[i]]
            state.history.append(token)
            # Matches pinned SGLang: token stop takes precedence at a length tie;
            # ignore_eos also suppresses explicit stop-token checks.
            if not state.spec.ignore_eos and token in state.spec.stop_tokens:
                state.reason, state.matched = Reason.TOKEN, token
            elif len(state.history) >= state.spec.max_new_tokens:
                state.reason = Reason.LENGTH
            out.record(handle, (token,), state.reason, state.matched)
            state.output_cursor += 1
            state.notified = bool(state.reason)
        return out
