# infe-parsers

Streaming tool-call and reasoning-content parsers for LLM inference engines.

## Contract

**Version:** v0.1.0  
**Step granularity:** One `feed()` call per engine step. The engine passes all
decoded text chunks for the step as a batch; the parser returns structured
deltas (tool-call fragments, reasoning fragments, plain content).

## What it replaces

| Engine | Subsystem | Files | LOC |
|--------|-----------|-------|-----|
| vLLM | `tool_parsers` | 52 | 16,847 |
| vLLM | `reasoning` | 33 | 5,154 |
| SGLang | `function_call` | 43 | 14,012 |
| SGLang | `reasoning/parser` | 9 | 6,094 |

Both engines call their parsers **per token** -- each decoded token triggers a
Python method call into the parser. `infe-parsers` batches tokens into a
single Rust call per step, eliminating per-token Python crossings.

## Dialects

| Dialect | Format | Models |
|---------|--------|--------|
| `hermes` | XML-like tags around JSON | NousResearch/Hermes, many fine-tunes |
| `llama3_json` | Bare JSON with `name`/`parameters` | Llama-3.1+ |
| `deepseek_reasoning` | `<think>...</think>` reasoning blocks | DeepSeek-R1, Qwen-QwQ |

## Usage

```rust
use infe_parsers::{DialectRegistry, StreamingParser};

let dialect = DialectRegistry::create("hermes")?;
let mut parser = StreamingParser::new(dialect);

// Called once per engine step with decoded text:
let result = parser.feed("<tool_call>{\"name\":\"fn\",\"arguments\":{}}</tool_call>");

for tc in &result.tool_calls {
    println!("tool call: {} (complete: {})", tc.name.as_deref().unwrap_or("?"), tc.is_complete);
}
```

## Engine integration

### vLLM

Registered via `--tool-call-parser` + `--tool-parser-plugin <path>`, or
`--reasoning-parser` + `--reasoning-parser-plugin <path>`. The shim is a
thin Python package that creates a `StreamingParser` and calls `feed` with
each batch of decoded text.

### SGLang

Registered via `--tool-call-parser` or `--reasoning-parser`, using the
`sglang.srt.plugins` hook registry (2026).

## Conformance

Fixtures live in `conformance/fixtures/parsers/`. Each fixture specifies
input chunks and expected deltas. The conformance runner feeds chunks
sequentially and asserts the accumulated output matches.

The acceptance criterion (BRIEF section 6.1) is 100% pass on fixtures mined
from both engines' parser test suites.

## Benchmarks

Criterion microbenchmarks in `benches/parse_stream.rs`:

- `hermes_single_chunk` -- parse a complete Hermes tool call in one call
- `llama3_json_single_chunk` -- parse a complete Llama-3 JSON tool call
- `plain_content_pass_through` -- parse plain text with no tool calls

### A/B benchmarking against vLLM/SGLang stock parsers

The A/B harness (BRIEF section 9 M1) measures `infe-parsers` vs the stock
Python parsers in both engines. The design:

1. **Microbenchmark (this crate):** Criterion measures pure parser throughput
   -- deltas/sec, ns/delta -- for each dialect across single-chunk and
   multi-chunk inputs. This isolates the parser's CPU cost from the engine.

2. **PyO3 crossing cost (M0 deliverable):** A separate microbenchmark measures
   the cost of one `feed` call through PyO3 vs one Python method call, to
   quantify the per-call overhead reduction (batch vs per-token).

3. **End-to-end A/B (M1):** Run the same model with tool-heavy traffic at
   64/256/1024 concurrent streams, comparing:
   - Stock vLLM with `--tool-call-parser hermes` (Python)
   - vLLM with `infe-parsers` shim (Rust via PyO3)
   - Stock SGLang with `--tool-call-parser hermes` (Python)
   - SGLang with `infe-parsers` shim (Rust via PyO3)

   Metrics: ITL p50/p99, API-server CPU%, throughput (tokens/sec).
   Raw JSON reports committed to `bench/results/`.

4. **Parity matrix:** The conformance suite runs against both engines'
   pinned versions, producing a per-dialect pass/fail matrix. Nightly CI
   runs against engine `main` and reports drift.

## Last A/B result

*Not yet measured -- M1 pending.*
