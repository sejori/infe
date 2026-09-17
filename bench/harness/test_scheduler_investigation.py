import importlib.util
from pathlib import Path
import unittest

from scheduler_investigation import plan, summary

source = Path(__file__).resolve().parents[2] / "shims/sglang/infe_sched_probe/infe_sched_probe.py"
spec = importlib.util.spec_from_file_location("scheduler_probe_under_test", source)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class RecorderTests(unittest.TestCase):
    def test_disabled_recorder_does_not_collect(self):
        recorder = plugin.Recorder()
        self.assertEqual(recorder.wrap("schedule")(lambda x: x + 1, 2), 3)
        self.assertEqual(recorder.stats, {})

    def test_explicit_windows_reset_and_exclude_straddling_calls(self):
        recorder = plugin.Recorder()
        recorder.start("short")
        recorder.wrap("schedule")(lambda: None)
        self.assertEqual(recorder.stop()["phases"]["schedule"]["calls"], 1)
        recorder.start("mixed")
        recorder.wrap("schedule")(recorder.stop)
        self.assertEqual(recorder.stats["schedule"]["calls"], 0)

    def test_nested_ranges_are_reported_separately(self):
        recorder = plugin.Recorder()
        recorder.start("mixed")
        child = recorder.wrap("prepare_decode")
        recorder.wrap("schedule")(lambda: child(lambda: None))
        stats = recorder.stop()["phases"]
        self.assertEqual(stats["schedule"]["calls"], 1)
        self.assertEqual(stats["prepare_decode"]["calls"], 1)
        self.assertGreaterEqual(stats["schedule"]["wall_ns"], stats["prepare_decode"]["wall_ns"])

    def test_exceptions_propagate_and_close_nvtx_range(self):
        class NVTX:
            def range_start(self, label): return 1
            def range_end(self, handle): pass
            def range_push(self, label): self.depth = getattr(self, "depth", 0) + 1
            def range_pop(self): self.depth -= 1
        nvtx = NVTX()
        recorder = plugin.Recorder()
        recorder.start("short", nvtx)
        def fail(): raise ValueError("test")
        with self.assertRaises(ValueError):
            recorder.wrap("schedule")(fail)
        self.assertEqual(nvtx.depth, 0)
        self.assertEqual(recorder.stop()["phases"]["schedule"]["calls"], 1)


class WorkloadTests(unittest.TestCase):
    def test_workloads_are_reproducible_with_fixed_lengths(self):
        for mode in ("short", "mixed", "cache_churn"):
            self.assertEqual(plan(mode, 140), plan(mode, 140))
        self.assertEqual((len(plan("short", 0)[0]), plan("short", 0)[1]), (32, 16))
        self.assertEqual({len(plan("mixed", i)[0]) for i in range(9)}, {128, 1024, 4096})
        self.assertEqual({plan("mixed", i)[1] for i in range(9)}, {16, 64, 128})

    def test_prefix_pressure_includes_reuse_and_diversity(self):
        prefixes = {tuple(plan("cache_churn", i)[0][:-32]) for i in range(128)}
        self.assertEqual(len(prefixes), 128)
        self.assertEqual(plan("cache_churn", 128)[0][:-32], plan("cache_churn", 0)[0][:-32])
        self.assertNotEqual(plan("cache_churn", 128)[0][-32:], plan("cache_churn", 0)[0][-32:])
        self.assertIn(tuple(plan("cache_churn", 131)[0][:-32]), prefixes)


if __name__ == "__main__":
    unittest.main()
