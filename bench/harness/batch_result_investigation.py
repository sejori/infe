#!/usr/bin/env python3
"""Tuned baseline and detailed result attribution; no engine decisions change."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import scheduler_investigation as runner
from scheduler_graph_check import launch_command
from scheduler_trace import analyse, extract

RESULT = "sglang.srt.managers.scheduler_components.batch_result_processor."
OUTPUT = "sglang.srt.managers.scheduler_components.output_streamer."
EXTRA_PHASES = {
    "result_decode": RESULT + "SchedulerBatchResultProcessor.process_batch_result_decode",
    "result_prefill": RESULT + "SchedulerBatchResultProcessor.process_batch_result_prefill",
    "normalize_tokens": RESULT + "SchedulerBatchResultProcessor._normalize_decode_outputs",
    "finish_update": "sglang.srt.managers.schedule_batch.Req.update_finish_state",
    "finish_actions": RESULT + "SchedulerBatchResultProcessor._handle_finish_state_updated_req",
    "cache_release": RESULT + "release_kv_cache",
    "output_stream": OUTPUT + "SchedulerOutputStreamer.stream_output",
    "output_pack": OUTPUT + "_GenerationStreamAccumulator.to_payload",
    "output_send": "sglang.srt.managers.scheduler_components.output_sender.SenderWrapper.send_output",
}
FLAGS = ["--cuda-graph-max-bs-decode", "256"]


def make_plugin(base, destination):
    source = (base / "infe_sched_probe.py").read_text()
    marker = "\nRECORDER = Recorder()"
    if source.count(marker) != 1:
        raise ValueError("Unexpected base recorder layout")
    source = source.replace(marker, "\nPHASES.update(" + repr(EXTRA_PHASES) + ")\n" + marker)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "infe_sched_probe.py").write_text(source)
    (destination / "pyproject.toml").write_bytes((base / "pyproject.toml").read_bytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, help="Export a completed trace instead of running servers")
    ap.add_argument("--gpu")
    ap.add_argument("--hf-cache", type=Path)
    ap.add_argument("--base-plugin", type=Path)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--trace-sessions", type=int, default=2)
    ap.add_argument("--pilot-count", type=int, default=0)
    ap.add_argument("--port", type=int, default=18106)
    ap.add_argument("--control-port", type=int, default=19107)
    a = ap.parse_args()
    if a.sqlite:
        evidence = extract(a.sqlite, EXTRA_PHASES)
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "intervals.json.gz").write_bytes(gzip.compress(json.dumps(evidence).encode(), mtime=0))
        (a.output / "overlap.json").write_text(json.dumps(analyse(evidence), indent=2) + "\n")
        print("Exported relative batch-result intervals.")
        return
    if a.gpu is None or a.hf_cache is None or a.base_plugin is None:
        ap.error("Run mode requires --gpu, --hf-cache and --base-plugin")
    plugin = a.output.resolve().with_name(a.output.name + "-plugin")
    make_plugin(a.base_plugin, plugin)
    original_save = runner.save
    def save(path, value):
        if path.name == "manifest.json":
            value.update(extra_server_args=FLAGS, extra_phases=EXTRA_PHASES,
                         purpose="Tuned result-stage attribution; diagnostic calibration only")
            for p in [Path(__file__), Path(__file__).with_name("scheduler_graph_check.py")]:
                value["source_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        original_save(path, value)
    runner.save = save
    runner.command = launch_command(FLAGS)
    sys.argv = ["scheduler_investigation", "--gpu", a.gpu, "--hf-cache", str(a.hf_cache),
                "--plugin", str(plugin), "--output", str(a.output), "--workloads", "short",
                "--pairs", str(a.pairs), "--trace-sessions", str(a.trace_sessions),
                "--pilot-count", str(a.pilot_count), "--port", str(a.port),
                "--control-port", str(a.control_port)]
    runner.main()


if __name__ == "__main__":
    main()
