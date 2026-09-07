# infe

Rust crates for faster LLM inference. Drop-in tool-call and reasoning-content
streaming parsers for vLLM and SGLang, via PyO3 shims -- no engine fork
required.

## What it does

When an LLM generates a tool call, it emits structured text like:

- Hermes: tool_call markers wrapping JSON
- Llama-3 JSON: bare JSON objects with name and parameters
- DeepSeek reasoning: think tags wrapping chain-of-thought

The engine parser must detect these markers, extract the function name and
arguments, and emit structured deltas to the SSE client token by token.

Stock engines do this in Python, per token. infe-parsers does it in Rust,
called once per engine step with a batch of decoded text.

## Supported dialects

- Hermes -- Qwen2.5, NousResearch/Hermes
- Llama-3 JSON -- Llama-3.1+
- DeepSeek reasoning -- DeepSeek-R1, Qwen-QwQ

## Status

Alpha. All dialects pass conformance tests (37 unit tests + 6 fixtures).
PyO3 wheels build on Linux x86_64. Live A/B benchmarks run on RTX 4090
with Qwen2.5-1.5B-Instruct against vLLM 0.28.0 and SGLang 0.5.18.

## Parity

| Engine | Args | ID  | Index | Streaming diff | Markerless continuation |
|--------|------|-----|-------|----------------|-------------------------|
| vLLM   | OK   | OK  | OK    | OK             | N/A                     |
| SGLang | OK   | OK  | OK    | OK             | OK (B6 fix)             |

## Key design decisions

- Incremental argument streaming: arguments streamed as diff fragments
  (name+id on first delta, arg diffs as they arrive, completion at close)
- Marker-less continuation (B6): after first tool call, bare JSON detected
  via brace-depth tracking (SGLang template omits wrapper on second call)
- Random tool-call IDs (B7): xorshift PRNG replaces deterministic IDs
- vLLM delta buffering: shim buffers excess deltas and drips them one per
  token, matching stock per-token streaming granularity

## Usage

### Rust

    use infe_parsers::{DialectRegistry, StreamingParser};
    let dialect = DialectRegistry::create("hermes").unwrap();
    let mut parser = StreamingParser::new(dialect);
    let result = parser.feed("decoded text");

### Python (PyO3)

    from infe_parsers import StreamingParser
    parser = StreamingParser("hermes")
    result = parser.feed("decoded text")

### vLLM

    vllm serve Qwen/Qwen2.5-1.5B-Instruct --tool-call-parser infe_hermes --tool-parser-plugin shims/vllm/infe_parsers_vllm

### SGLang

    python -m infe_parsers_sglang.launch -- --model Qwen/Qwen2.5-1.5B-Instruct --tool-call-parser infe_hermes

## Testing

    cargo test -p infe-parsers
    cargo clippy -p infe-parsers --all-features -- -D warnings
    cargo fmt -p infe-parsers --check
    maturin develop -m python/infe-parsers/Cargo.toml

## Benchmarks

    python bench/harness/ab_benchmark.py --dialect hermes
    python bench/harness/e2e_tool_stream.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct --arm infe --engine vllm --concurrency 8 64 256 --output bench/results/out.json

## Project layout

    crates/infe-core/        Shared types, error model, component trait
    crates/infe-parsers/     Streaming parser crate
    python/infe-parsers/    PyO3 bindings (abi3 wheel via maturin)
    shims/vllm/             vLLM tool-parser plugin
    shims/sglang/           SGLang detector + launcher
    conformance/fixtures/   JSON test fixtures
    bench/harness/          A/B benchmark scripts
    bench/results/          Raw JSON benchmark results
    docs/                   Review documents and task tracking

## Engine versions

Tested against vLLM 0.28.0 and SGLang 0.5.18.

## License

MIT OR Apache-2.0
