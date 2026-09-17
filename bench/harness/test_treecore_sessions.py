import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from treecore_sessions import comparisons, public_config, redact_log, schedule, validate
from summarize_treecore import load_sessions


class SessionTests(unittest.TestCase):
    def test_public_config_excludes_local_identity(self):
        self.assertEqual(public_config({"gpu": "GPU-private", "hf_cache": Path("/cache"),
                                        "output": Path("/results"), "rounds": 3}), {"rounds": 3})

    def test_log_redaction_preserves_image_digest_and_measurements(self):
        gpu = "GPU-" + "01234567-89ab-cdef-0123-456789abcdef"
        digest = "sha256:" + "a" * 64
        source = (f"operator@benchmark-node /home/operator/cache {gpu} "
                  f"docker-{'b' * 64}.scope infe-treecore-012345abcdef {digest} 2610 MHz")
        cleaned = redact_log(source, "benchmark-node")
        for identifier in ("operator", "benchmark-node", gpu, "b" * 64, "012345abcdef"):
            self.assertNotIn(identifier, cleaned)
        self.assertIn(digest, cleaned)
        self.assertIn("2610 MHz", cleaned)

    def test_balanced_reproducible_schedule(self):
        orders = schedule(6, 123)
        self.assertEqual(orders, schedule(6, 123))
        self.assertEqual(orders.count(["python", "rust"]), 3)
        with self.assertRaises(ValueError):
            schedule(3, 123)

    def test_effects_are_paired_not_pooled(self):
        sessions = []
        for pair, baseline in enumerate([10, 100]):
            for arm, factor in [("python", 1), ("rust", 1.1)]:
                sessions.append({"pair": pair, "arm": arm, "completed": 1,
                                 "levels": [{"concurrency": 64,
                                             "summary": {"e2e_p50": baseline * factor}}]})
        effect = comparisons(sessions)[64]["e2e_p50"]
        self.assertEqual(effect["pairs"], 2)
        for change in effect["paired_percent_changes"]:
            self.assertAlmostEqual(change, 10)
        del sessions[0]["completed"]
        with self.assertRaises(ValueError):
            comparisons(sessions)

    def test_http_success_without_tool_calls_is_failure(self):
        with self.assertRaises(RuntimeError):
            validate([{"error": None, "ttft_ms": 1, "tool_calls_parity": []}])

    def test_archived_sessions_require_complete_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({
                "orders": [["python", "rust"]], "config": {"concurrency": [64], "rounds": 1}}))
            for position, arm in enumerate(["python", "rust"]):
                with gzip.open(path / f"pair-00-{position}-{arm}.json.gz", "wt") as output:
                    json.dump({"pair": 0, "position": position, "arm": arm, "completed": 1,
                               "levels": [{"concurrency": 64, "rounds": [{}]}]}, output)
            self.assertEqual(len(load_sessions(path)[1]), 2)
            (path / "pair-00-1-rust.json.gz").unlink()
            with self.assertRaises(FileNotFoundError):
                load_sessions(path)

    def test_legacy_cpu_file_is_not_used_as_per_level_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            summary = dict(concurrency=64, ttft_p50=1, itl_p50=1, itl_p99=1,
                           e2e_p50=1, chunks_per_s=1, errors=0, parity_calls=2,
                           parity_args_ok=2, parity_has_id=2, ok=1)
            path.write_text(json.dumps({"engine": "sglang", "arm": "python",
                                        "levels": [{"summary": summary}]}))
            path.with_suffix(".cpu.txt").write_text("172.3\n")
            output = subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("summarize_ab.py")), str(path)],
                text=True)
            row = next(line for line in output.splitlines() if line.startswith("sglang"))
            self.assertEqual(row.split()[8], "nan")
            summary["cpu_percent"] = 123
            path.write_text(json.dumps({"engine": "sglang", "arm": "python",
                                        "levels": [{"summary": summary}]}))
            output = subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("summarize_ab.py")), str(path)],
                text=True)
            row = next(line for line in output.splitlines() if line.startswith("sglang"))
            self.assertEqual(row.split()[8], "123")


if __name__ == "__main__":
    unittest.main()
