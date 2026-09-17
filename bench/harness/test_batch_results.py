from array import array
import unittest

from batch_result_replay import events, replay
from batch_results import BatchOutput, NO_TOKEN, PythonBatchStage, Reason, RequestSpec


class BatchStageTests(unittest.TestCase):
    def test_stop_wins_length_tie_and_ignore_eos_suppresses_all_token_stops(self):
        stage = PythonBatchStage(2)
        a = stage.admit(RequestSpec(1, 32, {7}))
        b = stage.admit(RequestSpec(1, 32, {7}, ignore_eos=True))
        out = stage.advance(0, [b, a], [7, 7], [0, 1, 2])
        self.assertEqual(list(out.reasons), [Reason.LENGTH, Reason.TOKEN])
        self.assertEqual(list(out.matched_tokens), [NO_TOKEN, 7])
        self.assertEqual(list(out.release_handles), [b, a])
        self.assertEqual(stage.snapshot(a)["tokens"], [7])

    def test_overlapping_result_cannot_reach_reused_slot(self):
        stage = PythonBatchStage(1)
        old = stage.admit(RequestSpec(1, 32))
        stage.advance(0, [old], [1], [0, 1])
        stage.acknowledge_release(old)
        new = stage.admit(RequestSpec(2, 32))
        self.assertNotEqual(new, old)
        stage.acknowledge_release(old)
        out = stage.advance(1, [old, new], [9, 2], [0, 1, 2])
        self.assertEqual(list(out.handles), [new])
        self.assertEqual(list(out.ignored_handles), [old])
        self.assertEqual(stage.snapshot(new)["tokens"], [2])

    def test_cancellation_is_reported_once_even_without_a_gpu_row(self):
        stage = PythonBatchStage(1)
        handle = stage.admit(RequestSpec(8, 32))
        stage.advance(0, [handle], [3], [0, 1])
        stage.cancel(handle)
        stage.cancel(handle)
        out = stage.advance(1, [], [], [0])
        self.assertEqual(list(out.reasons), [Reason.CANCELLED])
        self.assertEqual(list(out.offsets), [0, 0])
        self.assertEqual(list(out.release_handles), [handle])
        out = stage.advance(2, [handle], [8], [0, 1])
        self.assertEqual(list(out.release_handles), [])
        self.assertEqual(stage.snapshot(handle)["tokens"], [3])

    def test_bad_batch_is_atomic_for_state_output_and_pending_cancellation(self):
        stage = PythonBatchStage(2)
        a, b = [stage.admit(RequestSpec(8, 32)) for _ in range(2)]
        out = stage.advance(0, [a], [3], [0, 1])
        saved = out.snapshot()
        stage.cancel(a)
        for handles, tokens, offsets in [([a, b], [4, 99], [0, 1, 2]),
                                         ([b, b], [4, 5], [0, 1, 2]),
                                         ([b], [4, 5], [0, 2]),
                                         ([b], [4], [1, 1]),
                                         ([0], [4], [0, 1])]:
            with self.assertRaises(ValueError):
                stage.advance(1, handles, tokens, offsets, out)
            self.assertEqual(out.snapshot(), saved)
            self.assertEqual(stage.snapshot(b)["tokens"], [])
            self.assertTrue(stage.snapshot(a)["cancel_pending"])
        stage.advance(1, [b], [4], [0, 1], out)
        self.assertEqual(list(out.release_handles), [a])
        self.assertEqual(stage.snapshot(b)["tokens"], [4])

    def test_retries_early_release_and_unsupported_features_fail_explicitly(self):
        stage = PythonBatchStage(1)
        with self.assertRaises(NotImplementedError):
            stage.admit(RequestSpec(8, 32, features=("grammar",)))
        handle = stage.admit(RequestSpec(8, 32))
        with self.assertRaises(ValueError):
            stage.acknowledge_release(handle)
        stage.advance(0, [handle], [4], [0, 1])
        with self.assertRaises(ValueError):
            stage.advance(0, [handle], [4], [0, 1])
        self.assertEqual(stage.snapshot(handle)["tokens"], [4])

    def test_snapshot_does_not_alias_owned_history_and_buffers_can_be_reused(self):
        stage = PythonBatchStage(1)
        handle = stage.admit(RequestSpec(2, 32))
        out = BatchOutput()
        stage.advance(0, array("Q", [handle]), array("I", [4]), array("I", [0, 1]), out)
        snapshot = stage.snapshot(handle)
        snapshot["tokens"].append(9)
        stage.advance(1, [handle], [5], [0, 1], out)
        self.assertEqual(list(out.token_ids), [5])
        self.assertEqual(stage.snapshot(handle)["tokens"], [4, 5])

    def test_synthetic_lifetime_replay(self):
        result = replay(events(cycles=10))
        self.assertEqual(result["admissions"], 80)
        self.assertEqual(result["completion_records"], 80)
        self.assertGreater(result["ignored_rows"], 500)


class AttributionTests(unittest.TestCase):
    def test_residual_subtracts_nested_ranges_and_gpu_api_union_once(self):
        from summarize_batch_results import residual
        window = {"duration_ns": 100, "gpu_busy_ns": [[0, 20]],
                  "cuda_api_ns": {"synchronise": [[10, 30]]},
                  "phases_ns": {"result": [[0, 100]],
                                "finish_actions": [[40, 80]],
                                "cache_release": [[50, 70]]}}
        self.assertEqual(residual(window, ["cache_release"]), 50)
        self.assertEqual(residual(window, ["finish_actions", "cache_release"]), 30)


if __name__ == "__main__":
    unittest.main()
