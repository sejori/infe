#!/usr/bin/env python3
"""Render a complete TreeCore session experiment as Markdown, including paired effects."""
import argparse
import gzip
import json
from pathlib import Path

from treecore_sessions import comparisons


def load_sessions(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    sessions = []
    for pair, order in enumerate(manifest["orders"]):
        for position, arm in enumerate(order):
            path = directory / f"pair-{pair:02d}-{position}-{arm}.json"
            if path.exists():
                session = json.loads(path.read_text())
            else:
                with gzip.open(str(path) + ".gz", "rt") as source:
                    session = json.load(source)
            if (session["pair"], session["position"], session["arm"]) != (pair, position, arm):
                raise ValueError(f"Session does not match schedule: {path}")
            if "completed" not in session or "error" in session:
                raise ValueError(f"Incomplete session: {path}")
            if [level["concurrency"] for level in session["levels"]] != manifest["config"]["concurrency"]:
                raise ValueError(f"Missing/different concurrency levels: {path}")
            for level in session["levels"]:
                if len(level["rounds"]) != manifest["config"]["rounds"]:
                    raise ValueError(f"Missing rounds: {path}")
            sessions.append(session)
    return manifest, sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    manifest, sessions = load_sessions(a.directory)
    results = comparisons(sessions)
    print(f"# TreeCore paired-session results\n\n{len(sessions)} fresh sessions / "
          f"{len(manifest['orders'])} counterbalanced pairs on one GPU.")
    print("\nValues are medians across sessions; effects are medians of paired percentage changes. "
          "These differ from the percentage change between the two displayed medians. "
          "Positive latency effects mean Rust is slower. Ranges are observed pair ranges, "
          "not confidence intervals.\n")
    print("| Concurrency | Metric | Python | Rust | Paired effect | Pair range |")
    print("|---|---|---:|---:|---:|---:|")
    for conc, metrics in sorted(results.items()):
        for metric, effect in metrics.items():
            print(f"| {conc} | {metric} | {effect['python_session_median']:.2f} | "
                  f"{effect['rust_session_median']:.2f} | {effect['median_percent_change']:+.2f}% | "
                  f"{effect['min_percent_change']:+.2f}% to {effect['max_percent_change']:+.2f}% |")
    print("\nLatency units: milliseconds. CPU: percent of one core. "
          "Throughput: requests/second. Delta count: meaningful tool-call deltas/request.\n")
    print("| Pair | Order | " + " | ".join(f"c={c} e2e effect" for c in sorted(results)) + " |")
    print("|---|---|" + "---:|" * len(results))
    for pair, order in enumerate(manifest["orders"]):
        effects = [results[c]["e2e_p50"]["paired_percent_changes"][pair] for c in sorted(results)]
        print(f"| {pair} | {' → '.join(order)} | " + " | ".join(f"{e:+.2f}%" for e in effects) + " |")


if __name__ == "__main__":
    main()
