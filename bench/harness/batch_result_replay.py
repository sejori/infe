#!/usr/bin/env python3
"""Replay ABI-v1 events; optionally check token completion against pinned SGLang.

The engine oracle uses the real Req.update_finish_state method. Cancellation
is compared at the already-observed terminal-event boundary, not through an
HTTP abort race. This does not establish whole-engine integration parity.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/infe-batch-results"))
from batch_results import BatchOutput, NO_TOKEN, PythonBatchStage, RequestSpec, Reason


def events(seed=20260917, cycles=80):
    rng = random.Random(seed)
    result, previous = [], []
    step = 0
    for cycle in range(cycles):
        names = [f"r{cycle * 8 + i}" for i in range(8)]
        for i, name in enumerate(names):
            result.append({"op": "admit", "name": name, "spec": {
                "max_new_tokens": 12 if i == 0 else rng.randint(1, 12),
                "vocab_size": 32, "stop_tokens": [] if i == 0 else [rng.randrange(32)],
                "ignore_eos": bool(i % 2)}})
        for tick in range(16):
            if tick == 3:
                result.append({"op": "cancel", "name": names[0]})
            order = names + previous[:2]
            rng.shuffle(order)
            result.append({"op": "step", "step": step, "names": order,
                           "tokens": [rng.randrange(32) for _ in order],
                           "offsets": list(range(len(order) + 1))})
            step += 1
        for name in names:
            result.append({"op": "release", "name": name})
        previous = names
    return result


class EngineOracle:
    def __init__(self):
        from sglang.srt.managers.schedule_batch import Req, FINISH_ABORT
        self.Req, self.abort = Req, FINISH_ABORT
        self.requests = {}

    def admit(self, handle, spec):
        req = object.__new__(self.Req)
        req.finished_reason = req.to_finish = req.finished_len = None
        req.output_ids = []
        req.vocab_size = spec.vocab_size
        req.eos_token_ids = set()
        req.tokenizer = req.grammar = None
        req.sampling_params = SimpleNamespace(max_new_tokens=spec.max_new_tokens,
            ignore_eos=spec.ignore_eos, stop_token_ids=set(spec.stop_tokens),
            stop_strs=[], stop_regex_strs=[])
        self.requests[handle] = req

    def cancel(self, handle):
        req = self.requests.get(handle)
        if req is not None and not req.finished():
            req.finished_reason = self.abort("synthetic cancellation")

    def commit(self, handle, tokens):
        req = self.requests.get(handle)
        if req is not None and not req.finished():
            for token in tokens:
                req.output_ids.append(token)
                req.update_finish_state(1)

    def check(self, handle, snapshot):
        req = self.requests.get(handle)
        if req is None:
            assert snapshot is None
            return
        finish = req.finished_reason.to_json() if req.finished() else {}
        kind = finish.get("type")
        reason = {None: Reason.RUNNING, "length": Reason.LENGTH,
                  "stop": Reason.TOKEN, "abort": Reason.CANCELLED}[kind]
        assert snapshot["tokens"] == req.output_ids, (handle, snapshot, req.output_ids)
        assert snapshot["reason"] == reason, (handle, snapshot, finish)
        if reason == Reason.TOKEN:
            assert snapshot["matched"] == finish["matched"]


def replay(trace, oracle=None):
    stage, out = PythonBatchStage(8), BatchOutput()
    handles, released, notifications, streams = {}, set(), set(), {}
    counters = {"admissions": 0, "steps": 0, "completion_records": 0, "ignored_rows": 0,
                "tokens_emitted": 0, "engine_state_checks": 0}
    for event in trace:
        op = event["op"]
        if op == "admit":
            spec = RequestSpec(**event["spec"])
            handle = stage.admit(spec)
            handles[event["name"]] = handle
            streams[handle] = []
            counters["admissions"] += 1
            if oracle:
                oracle.admit(handle, spec)
        elif op == "cancel":
            handle = handles[event["name"]]
            stage.cancel(handle)
            if oracle:
                oracle.cancel(handle)
        elif op == "release":
            handle = handles[event["name"]]
            assert handle in notifications
            stage.acknowledge_release(handle)
            released.add(handle)
            if oracle:
                oracle.requests.pop(handle, None)
        elif op == "step":
            batch = [handles[name] for name in event["names"]]
            tokens, offsets = event["tokens"], event["offsets"]
            stage.advance(event["step"], array("Q", batch), array("I", tokens), array("I", offsets), out)
            if oracle:
                for i, handle in enumerate(batch):
                    oracle.commit(handle, tokens[offsets[i]:offsets[i + 1]])
            assert len(out.offsets) == len(out.handles) + 1
            assert out.offsets[-1] == len(out.token_ids)
            assert list(out.release_handles) == [h for h, reason in zip(out.handles, out.reasons) if reason]
            for i, handle in enumerate(out.handles):
                assert handle not in released
                streams[handle].extend(out.token_ids[out.offsets[i]:out.offsets[i + 1]])
                if out.reasons[i]:
                    assert handle not in notifications
                    notifications.add(handle)
                    counters["completion_records"] += 1
            for handle in set(batch) - released:
                snapshot = stage.snapshot(handle)
                assert snapshot["tokens"] == streams[handle]
                assert snapshot["output_cursor"] == len(streams[handle])
                if oracle:
                    oracle.check(handle, snapshot)
                    counters["engine_state_checks"] += 1
            counters["steps"] += 1
            counters["ignored_rows"] += len(out.ignored_handles)
            counters["tokens_emitted"] += len(out.token_ids)
        else:
            raise ValueError("Unknown event")
    assert len(notifications) == counters["admissions"] == len(released)
    return counters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path)
    ap.add_argument("--write-events", type=Path)
    ap.add_argument("--engine-oracle", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    trace = json.loads(a.events.read_text()) if a.events else events()
    encoded = json.dumps(trace, separators=(",", ":")).encode()
    if a.write_events:
        a.write_events.write_bytes(encoded + b"\n")
    result = {"abi_version": 1, "synthetic": True, "engine_oracle": a.engine_oracle,
              "events_sha256": hashlib.sha256(encoded).hexdigest(),
              "counts": replay(trace, EngineOracle() if a.engine_oracle else None)}
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
