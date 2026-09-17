import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scheduler_trace import analyse, extract, intersection, merge


class IntervalTests(unittest.TestCase):
    def test_overlapping_gpu_streams_are_not_double_counted(self):
        gpu = merge([[0, 20], [10, 30], [40, 50]])
        self.assertEqual(gpu, [[0, 30], [40, 50]])
        self.assertEqual(intersection([[15, 45]], gpu), 20)

    def test_nested_phases_are_not_added_to_parent(self):
        result = analyse({"windows": [{"case": "mixed_c64", "duration_ns": 100,
                                        "gpu_busy_ns": [[20, 80]],
                                        "phases_ns": {"schedule": [[0, 40]],
                                                      "prepare_decode": [[10, 30]],
                                                      "result": [[75, 90]], "ingest": []}}]})["mixed_c64"]
        self.assertEqual(result["gpu_busy_percent"], 60)
        self.assertEqual(result["phases"]["schedule"]["gpu_uncovered_ns"], 20)
        self.assertEqual(result["phases"]["non_launch_bookkeeping"]["gpu_uncovered_ns"], 30)

    def test_full_gpu_overlap_has_no_exposed_cpu(self):
        result = analyse({"windows": [{"case": "short_c8", "duration_ns": 100,
                                        "gpu_busy_ns": [[0, 100]],
                                        "phases_ns": {"schedule": [[10, 50]]}}]})["short_c8"]
        self.assertEqual(result["phases"]["schedule"]["gpu_uncovered_ns"], 0)

    def test_cuda_api_and_gpu_union_not_double_subtracted(self):
        result = analyse({"windows": [{"case": "short_c8", "duration_ns": 100,
                                        "gpu_busy_ns": [[0, 40]],
                                        "cuda_api_ns": {"synchronise": [[20, 70]]},
                                        "phases_ns": {"result": [[0, 80]]}}]})["short_c8"]["phases"]["result"]
        self.assertEqual(result["gpu_uncovered_percent_of_window"], 40)
        self.assertEqual(result["gpu_uncovered_outside_cuda_api_percent_of_window"], 10)

    def test_export_strips_identifiers_and_matches_api_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.sqlite"
            con = sqlite3.connect(path)
            con.executescript("""
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER, globalTid INTEGER);
                INSERT INTO StringIds VALUES (1, 'cudaEventSynchronize_v3020'), (2, 'private-host /home/private-user');
                INSERT INTO NVTX_EVENTS VALUES
                  (1000, 1100, 'infe.window.short_c8', NULL, 777),
                  (1010, 1080, 'infe.phase.result', NULL, 888),
                  (1000, 1100, NULL, 2, 999);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1020, 1040);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1030, 1070, 1, 888), (1000, 1100, 1, 999);
            """)
            con.close()
            result = extract(path)
        window = result["windows"][0]
        self.assertEqual(window["gpu_busy_ns"], [[20, 40]])
        self.assertEqual(window["cuda_api_ns"]["synchronise"], [[30, 70]])
        self.assertEqual(window["phases_ns"]["result"], [[10, 80]])
        for sensitive in ("private-host", "private-user", "cudaEventSynchronize", "globalTid", "777", "888", "999"):
            self.assertNotIn(sensitive, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
