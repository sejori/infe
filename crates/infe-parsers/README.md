# infe-parsers

Streaming tool-call and reasoning-content parsers for LLM inference engines.

## Contract

**Version:** v0.1.0 (alpha)  
**Status:** Parity fixes applied, not yet re-validated against live engines.  
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

Both engines call their parsers **per token** — each decoded token triggers a
Python method call into the parser. `infe-parsers` batches tokens into a
single Rust call per step, eliminating per-token Python crossings.

## Dialects

| Dialect | Kind | Format | Models |
|---------|------|--------|--------|
| `hermes` | tool | XML-like tags around JSON | NousResearch/Hermes, many fine-tunes |
| `llama3_json` | tool | Bare JSON with `name`/`parameters` | Llama-3.1+ |
| `deepseek_reasoning` | reasoning | `<think>...` reasoning blocks | DeepSeek-R1, Qwen-QwQ |

Use `DialectRegistry::kind(name)` to check whether a dialect is a tool or
reasoning parser. Shims register tool and reasoning dialects through their
respective engine interfaces.

## Usage

Rust:

```rust
use infe_parsers::{DialectRegistry, StreamingParser};

let dialect = DialectRegistry::create("hermes")?;
let mut parser = StreamingParser::new(dialect);

// Called once per engine step with decoded text:
let result = parser.feed("");
for tc in &result.tool_calls {
    println!("tool call: {} (complete: {})", tc.name.as_deref().unwrap_or("?"), tc.is_complete);
}
```

Python (via PyO3):

```python
from infe_parsers import StreamingParser

parser = StreamingParser("hermes")
result = parser.feed("")
for tc in result["tool_calls"]:
    print(tc["name"], tc["arguments_fragment"], tc["id"])
```

## Engine integration

### vLLM

Registered via `--tool-call-parser infe_hermes` + `--tool-parser-plugin <path>`
for tool dialects, or `--reasoning-parser infe_deepseek_reasoning` +
`--reasoning-parser-plugin <path>` for reasoning dialects. The shim is a
thin Python package (`shims/vllm/infe_parsers_vllm/`) that creates a
`StreamingParser` and calls `feed` with each batch of decoded text.

No fork required — vLLM's plugin system loads the shim at startup.

Tested against vLLM 0.28.0. The shim handles import-path differences between
vLLM main and release <=0.28 via a try/except fallback.

### SGLang

Registered via `--tool-call-parser infe_hermes`. SGLang validates parser names
at arg-parse time, so the shim must be imported **before** `ServerArgs.add_cli_args`
runs. Use the launcher module:

```bash
python -m infe_parsers_sglang.launch -- <sglang args> --tool-call-parser infe_hermes
```

The launcher imports the shim (registering detectors) then delegates to
`sglang.launch_server`. No fork required.

Tested against SGLang 0.5.18.

## Conformance

Fixtures live in `conformance/fixtures/parsers/`. Each fixture specifies
input chunks, expected tool calls (with name, arguments sub-object, index,
completion flag), expected reasoning, and expected content. The conformance
runner feeds chunks sequentially and asserts:

- **Tool calls:** exact match on `name`, `arguments_fragment` (the extracted
  arguments sub-object, not the wrapper JSON), `index`, and `is_complete`.
- **IDs:** every tool call delta has a non-empty `id` matching `call_\w+`.
- **Reasoning:** exact match on `fragment` and `is_complete`.

Current fixtures are synthetic — **they should be mined from both engines'
parser test suites** (`vllm/tests/tool_use`, `sglang/test/.../function_call`)
for full coverage. This is tracked as a remaining task.

## Benchmarks

### Criterion microbenchmarks (`benches/parse_stream.rs`)

- `hermes_single_chunk` — parse a complete Hermes tool call in one call
- `llama3_json_single_chunk` — parse a complete Llama-3 JSON tool call
- `plain_content_pass_through` — parse plain text with no tool calls

### A/B benchmarking against vLLM/SGLang stock parsers

**Latest A/B result: RTX 4090, 2026-09-04** (`bench/results/rtx4090-20260904/`)

The first live A/B run identified parity failures (arguments not extracted,
missing tool-call IDs) in the pre-fix parser. These defects have been fixed in
code but **not yet re-validated**. A post-fix A/B run is the next step.

Raw JSON results are committed to `bench/results/rtx4090-20260904/` and should
never be summarised without referencing the raw files.

## Manifest

The component manifest lives at `registry/infe-parsers/manifest.yaml` and
describes dialects, engine seams, parity status, and conformance info.
