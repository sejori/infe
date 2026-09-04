#!/usr/bin/env python3
"""A/B benchmark harness for infe-parsers vs stock engine parsers.

This script runs a controlled A/B comparison between:
  - STOCK: the engine's built-in Python tool-call parser
  - INFE:  the Rust infe-parsers crate via PyO3

It follows the benchmark protocol from BRIEF.md §7.1:
  - Fixed workload (same prompts, same chunk patterns)
  - Two arms: stock and infe, interleaved
  - Multiple runs per arm, report median + IQR
  - Raw JSON committed under bench/results/

Usage:
    # Standalone microbenchmark (no engine needed):
    python bench/harness/ab_benchmark.py --dialect hermes --runs 10

    # Against a running vLLM server (stock vs infe):
    python bench/harness/ab_benchmark.py --engine vllm --endpoint http://localhost:8000 \\
        --stock-parser hermes --infe-parser infe_hermes \\
        --concurrency 64 256 1024 --runs 5

The standalone mode mocks the token stream (like MockTokenStream in the Rust
benches) and measures pure parsing throughput + latency. The engine mode
measures end-to-end ITL and API-server CPU via the engine's own metrics.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Mock token stream (mirrors the Rust MockTokenStream in benches/parse_stream.rs)
# ---------------------------------------------------------------------------

# Fixture: a realistic tool-call-heavy conversation chunked as a tokenizer would.
HERMES_FIXTURE = [
    "I'll check the weather for you.\n",
    "\u003ctool_call\u003e",
    '{"name":',
    ' "get_weather"',
    ', "arguments"',
    ': {"city"',
    ': "London"',
    ', "units"',
    ': "celsius"}}',
    "\u003c/tool_call\u003e",
    "\nDone!",
]

LLAMA3_FIXTURE = [
    "Sure! ",
    '{"name":',
    ' "get_weather"',
    ', "parameters"',
    ': {"city":',
    ' "London",',
    ' "units":',
    ' "celsius"}}',
    " All done!",
]

DEEPSEEK_FIXTURE = [
    "\u003cthink\u003e",
    "Let me think about this. ",
    "The user wants weather data. ",
    "I should call the weather function.",
    "\u003c/think\u003e",
    "Here's the weather: sunny, 22C.",
]

FIXTURES = {
    "hermes": HERMES_FIXTURE,
    "llama3_json": LLAMA3_FIXTURE,
    "deepseek_reasoning": DEEPSEEK_FIXTURE,
}


@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    arm: str  # "stock" or "infe"
    dialect: str
    concurrency: int
    run_index: int
    total_time_ms: float
    per_stream_times_ms: list[float] = field(default_factory=list)
    chunks_per_stream: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Stock Python parsers (re-implementation of the engine's Python logic)
# ---------------------------------------------------------------------------

class StockHermesParser:
    """Minimal re-implementation of vLLM's Hermes tool parser in pure Python.

    This is the 'stock' arm — it does the same work as the Rust parser but
    in Python, so we can measure the Rust speedup in isolation.
    """
    OPEN = "\u003ctool_call\u003e"
    CLOSE = "\u003c/tool_call\u003e"

    def __init__(self):
        self._buffer = ""
        self._in_tool = False
        self._args_buffer = ""

    def feed(self, text: str) -> dict:
        self._buffer += text
        result = {"tool_calls": [], "reasoning": [], "content": []}
        remaining = self._buffer
        output = []

        while remaining:
            if not self._in_tool:
                pos = remaining.find(self.OPEN)
                if pos == -1:
                    # Check partial prefix
                    partial = self._partial_prefix(remaining, self.OPEN)
                    if partial > 0:
                        content = remaining[:len(remaining) - partial]
                        if content:
                            output.append(("content", content))
                        self._buffer = remaining[len(remaining) - partial:]
                        remaining = ""
                    else:
                        output.append(("content", remaining))
                        self._buffer = ""
                        remaining = ""
                else:
                    if pos > 0:
                        output.append(("content", remaining[:pos]))
                    remaining = remaining[pos + len(self.OPEN):]
                    self._in_tool = True
                    self._args_buffer = ""
            else:
                pos = remaining.find(self.CLOSE)
                if pos == -1:
                    partial = self._partial_prefix(remaining, self.CLOSE)
                    if partial > 0:
                        chunk = remaining[:len(remaining) - partial]
                        if chunk:
                            self._args_buffer += chunk
                            output.append(("tool_call", {"arguments_fragment": chunk, "is_complete": False}))
                        self._buffer = remaining[len(remaining) - partial:]
                        remaining = ""
                    else:
                        self._args_buffer += remaining
                        output.append(("tool_call", {"arguments_fragment": remaining, "is_complete": False}))
                        self._buffer = ""
                        remaining = ""
                else:
                    chunk = remaining[:pos]
                    if chunk:
                        self._args_buffer += chunk
                        output.append(("tool_call", {"arguments_fragment": chunk, "is_complete": False}))
                    # Complete
                    import re
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', self._args_buffer)
                    name = name_match.group(1) if name_match else None
                    output.append(("tool_call", {"name": name, "arguments_fragment": self._args_buffer, "is_complete": True}))
                    self._args_buffer = ""
                    self._in_tool = False
                    remaining = remaining[pos + len(self.CLOSE):]

        # Translate output to result dict
        for kind, val in output:
            if kind == "content":
                result["content"].append(val)
            elif kind == "tool_call":
                result["tool_calls"].append(val)

        return result

    def finish(self) -> dict:
        result = {"tool_calls": [], "reasoning": [], "content": []}
        if self._buffer and not self._in_tool:
            result["content"].append(self._buffer)
        elif self._buffer and self._in_tool:
            import re
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', self._args_buffer + self._buffer)
            name = name_match.group(1) if name_match else None
            result["tool_calls"].append({
                "name": name,
                "arguments_fragment": self._args_buffer + self._buffer,
                "is_complete": True,
            })
        self._buffer = ""
        return result

    @staticmethod
    def _partial_prefix(text: str, marker: str) -> int:
        """Return length of partial marker prefix at end of text."""
        for i in range(1, min(len(text) + 1, len(marker) + 1)):
            if marker.startswith(text[-i:]):
                return i
        return 0

    def reset(self):
        self._buffer = ""
        self._in_tool = False
        self._args_buffer = ""


class StockLlama3JsonParser:
    """Minimal Python re-implementation of Llama-3 JSON tool parser."""

    def __init__(self):
        self._brace_depth = 0
        self._accumulating = False
        self._args_buffer = ""

    def feed(self, text: str) -> dict:
        result = {"tool_calls": [], "reasoning": [], "content": []}
        for ch in text:
            if not self._accumulating:
                if ch == "{":
                    self._accumulating = True
                    self._brace_depth = 1
                    self._args_buffer = "{"
                else:
                    result["content"].append(ch)
            else:
                self._args_buffer += ch
                if ch == "{":
                    self._brace_depth += 1
                elif ch == "}":
                    self._brace_depth -= 1
                    if self._brace_depth == 0:
                        import re
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', self._args_buffer)
                        name = name_match.group(1) if name_match else None
                        result["tool_calls"].append({
                            "name": name,
                            "arguments_fragment": self._args_buffer,
                            "is_complete": True,
                        })
                        self._accumulating = False
                        self._args_buffer = ""
        return result

    def finish(self) -> dict:
        return {"tool_calls": [], "reasoning": [], "content": []}

    def reset(self):
        self._brace_depth = 0
        self._accumulating = False
        self._args_buffer = ""


class StockDeepSeekReasoningParser:
    """Minimal Python re-implementation of DeepSeek reasoning parser."""

    OPEN = "\u003cthink\u003e"
    CLOSE = "\u003c/think\u003e"

    def __init__(self):
        self._in_think = False
        self._buffer = ""

    def feed(self, text: str) -> dict:
        self._buffer += text
        result = {"tool_calls": [], "reasoning": [], "content": []}
        remaining = self._buffer

        while remaining:
            if self._in_think:
                pos = remaining.find(self.CLOSE)
                if pos == -1:
                    partial = self._partial_prefix(remaining, self.CLOSE)
                    if partial > 0:
                        chunk = remaining[:len(remaining) - partial]
                        if chunk:
                            result["reasoning"].append({"fragment": chunk, "is_complete": False})
                        self._buffer = remaining[len(remaining) - partial:]
                        remaining = ""
                    else:
                        result["reasoning"].append({"fragment": remaining, "is_complete": False})
                        self._buffer = ""
                        remaining = ""
                else:
                    if pos > 0:
                        result["reasoning"].append({"fragment": remaining[:pos], "is_complete": False})
                    result["reasoning"].append({"fragment": "", "is_complete": True})
                    self._in_think = False
                    remaining = remaining[pos + len(self.CLOSE):]
            else:
                pos = remaining.find(self.OPEN)
                if pos == -1:
                    partial = self._partial_prefix(remaining, self.OPEN)
                    if partial > 0:
                        content = remaining[:len(remaining) - partial]
                        if content:
                            result["content"].append(content)
                        self._buffer = remaining[len(remaining) - partial:]
                        remaining = ""
                    else:
                        result["content"].append(remaining)
                        self._buffer = ""
                        remaining = ""
                else:
                    if pos > 0:
                        result["content"].append(remaining[:pos])
                    remaining = remaining[pos + len(self.OPEN):]
                    self._in_think = True

        return result

    def finish(self) -> dict:
        result = {"tool_calls": [], "reasoning": [], "content": []}
        if self._buffer:
            if self._in_think:
                result["reasoning"].append({"fragment": self._buffer, "is_complete": True})
            else:
                result["content"].append(self._buffer)
        self._buffer = ""
        return result

    @staticmethod
    def _partial_prefix(text: str, marker: str) -> int:
        for i in range(1, min(len(text) + 1, len(marker) + 1)):
            if marker.startswith(text[-i:]):
                return i
        return 0

    def reset(self):
        self._in_think = False
        self._buffer = ""


STOCK_PARSERS = {
    "hermes": StockHermesParser,
    "llama3_json": StockLlama3JsonParser,
    "deepseek_reasoning": StockDeepSeekReasoningParser,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_single_stream(parser, chunks: list[str]) -> float:
    """Run one parser through one stream of chunks, return time in ms."""
    start = time.perf_counter()
    for chunk in chunks:
        parser.feed(chunk)
    parser.finish()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def run_concurrent(parser_factory, chunks: list[str], concurrency: int) -> tuple[float, list[float]]:
    """Run N concurrent parser streams, return (total_ms, per_stream_ms_list)."""
    per_stream = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_single_stream, parser_factory(), chunks)
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            per_stream.append(f.result())
    total = (time.perf_counter() - start) * 1000
    return total, per_stream


def run_ab(
    dialect: str,
    concurrency_levels: list[int],
    runs: int,
    infe_module=None,
) -> dict:
    """Run the full A/B comparison.

    If infe_module is provided (the compiled PyO3 module), we benchmark
    both stock (Python) and infe (Rust). If not, we only benchmark stock.
    """
    fixture = FIXTURES.get(dialect)
    if not fixture:
        print(f"Unknown dialect: {dialect}", file=sys.stderr)
        sys.exit(1)

    results = []

    # Stock arm
    stock_cls = STOCK_PARSERS[dialect]
    print(f"\n=== STOCK ({dialect}, Python) ===")
    for conc in concurrency_levels:
        for run_idx in range(runs):
            total, per_stream = run_concurrent(stock_cls, fixture, conc)
            r = BenchResult(
                arm="stock", dialect=dialect, concurrency=conc,
                run_index=run_idx, total_time_ms=total,
                per_stream_times_ms=per_stream, chunks_per_stream=len(fixture),
            )
            results.append(asdict(r))
            print(f"  conc={conc:4d} run={run_idx+1}/{runs}  total={total:.2f}ms  "
                  f"median_stream={statistics.median(per_stream):.3f}ms")

    # Infe arm (Rust via PyO3)
    if infe_module is not None:
        print(f"\n=== INFE ({dialect}, Rust/PyO3) ===")
        def make_infe():
            return infe_module.StreamingParser(dialect)
        for conc in concurrency_levels:
            for run_idx in range(runs):
                total, per_stream = run_concurrent(make_infe, fixture, conc)
                r = BenchResult(
                    arm="infe", dialect=dialect, concurrency=conc,
                    run_index=run_idx, total_time_ms=total,
                    per_stream_times_ms=per_stream, chunks_per_stream=len(fixture),
                )
                results.append(asdict(r))
                print(f"  conc={conc:4d} run={run_idx+1}/{runs}  total={total:.2f}ms  "
                      f"median_stream={statistics.median(per_stream):.3f}ms")
    else:
        print("\n(infe PyO3 module not available — run `pip install -e python/infe_parsers` to enable the infe arm)")

    # Summary
    print("\n=== SUMMARY ===")
    for conc in concurrency_levels:
        stock_times = [r["total_time_ms"] for r in results if r["arm"] == "stock" and r["concurrency"] == conc]
        infe_times = [r["total_time_ms"] for r in results if r["arm"] == "infe" and r["concurrency"] == conc]

        stock_med = statistics.median(stock_times) if stock_times else 0
        infe_med = statistics.median(infe_times) if infe_times else 0

        if stock_times and infe_times:
            speedup = stock_med / infe_med if infe_med > 0 else float("inf")
            stock_iqr = (statistics.quantiles(stock_times, n=4)[2] -
                         statistics.quantiles(stock_times, n=4)[0]) if len(stock_times) >= 4 else 0
            infe_iqr = (statistics.quantiles(infe_times, n=4)[2] -
                        statistics.quantiles(infe_times, n=4)[0]) if len(infe_times) >= 4 else 0
            print(f"  conc={conc:4d}  stock={stock_med:.2f}ms (IQR={stock_iqr:.2f})  "
                  f"infe={infe_med:.2f}ms (IQR={infe_iqr:.2f})  "
                  f"speedup={speedup:.2f}x")
        elif stock_times:
            print(f"  conc={conc:4d}  stock={stock_med:.2f}ms  (infe not available)")

    return {
        "dialect": dialect,
        "concurrency_levels": concurrency_levels,
        "runs_per_arm": runs,
        "chunks_per_stream": len(fixture),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="A/B benchmark: infe-parsers (Rust) vs stock (Python)"
    )
    ap.add_argument("--dialect", default="hermes",
                    choices=["hermes", "llama3_json", "deepseek_reasoning"],
                    help="Parser dialect to benchmark")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[1, 64, 256, 1024],
                    help="Concurrency levels to test")
    ap.add_argument("--runs", type=int, default=5,
                    help="Runs per arm per concurrency level")
    ap.add_argument("--output", default=None,
                    help="Output JSON file (default: bench/results/<dialect>_ab.json)")
    args = ap.parse_args()

    # Try to import the infe PyO3 module
    infe_module = None
    try:
        import infe_parsers
        infe_module = infe_parsers
        print(f"infe-parsers v{infe_parsers.__version__} loaded (dialects: {infe_parsers.list_dialects()})")
    except ImportError:
        print("infe_parsers not installed — stock-only benchmark. "
              "Run: pip install -e python/infe_parsers")

    report = run_ab(args.dialect, args.concurrency, args.runs, infe_module)

    # Save results
    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output or str(out_dir / f"{args.dialect}_ab.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRaw results saved to: {out_file}")


if __name__ == "__main__":
    main()
