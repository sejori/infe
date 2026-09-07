# Next-wave task list (handoff, 2026-09-04) — updated after round 2

**Round-3 status (post B6/B7/shim-fix commit):** A1–A5, B1–B4, B6, B7 done. The vLLM shim now
buffers excess deltas and drips them one per token (matching stock streaming granularity). Tool-call IDs
are random (xorshift PRNG). Marker-less continuation calls are parsed via brace-depth tracking.**

Source of truth for *why*: `docs/review-2026-09-04.md`. This file is the *what*, ordered, with acceptance checks.
Working fixes exist only as scratch copies under `bench/results/rtx4090-20260904/scratch/`; nothing in
`crates/`, `python/` or `shims/` has been changed yet.

## A. Make the infe arms run at all (engine-side, small)

| # | File | Change | Evidence |
|---|---|---|---|
| A1 | `shims/vllm/infe_parsers/__init__.py` | Import `DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall` with a try/except: `vllm.entrypoints.generate.base.protocol` (main) → fallback `vllm.entrypoints.openai.engine.protocol` (≤0.28.0). Drop the inline import inside `extract_tool_calls`. | `diff -u shims/vllm/infe_parsers/__init__.py bench/results/rtx4090-20260904/scratch/vllm_shim.py` |
| A2 | same | `DeltaMessage(tool_calls=delta_tool_calls)` — always a list; 0.28 rejects `None` with a pydantic error on every stream. | same diff |
| A3 | `shims/sglang/infe_parsers/__init__.py` | Implement abstract `structure_info(self)` on the detector (mirror `HermesDetector`: `StructureInfo(begin='<tool_call>{"name":"'+name+'", "arguments":', end='}</tool_call>', trigger='<tool_call>')`). Without it `FunctionCallParser` cannot instantiate the class → HTTP 500 on every request. | `diff -u shims/sglang/infe_parsers/__init__.py bench/results/rtx4090-20260904/scratch/sglang_shim.py` |
| A4 | new `python/infe-parsers/python/infe_parsers/shims/sglang/launch.py` | File-based launcher (not `-c`/stdin — multiprocessing spawn re-imports `__main__` from path): under `if __name__ == "__main__":` import the shim, then `runpy.run_module("sglang.launch_server", run_name="__main__")`. SGLang validates `--tool-call-parser` against `FunctionCallParser.ToolCallParserEnum` *at arg-parse time*, so the shim must be imported first. | `bench/results/rtx4090-20260904/scratch/launch_sglang.py` |
| A5 | `shims/*/infe_parsers/` | Rename: these packages are literally named `infe_parsers` and shadow the wheel if ever on `sys.path`. Move them inside the wheel as `infe_parsers.shims.vllm` / `infe_parsers.shims.sglang` (the docstrings already claim that path). vLLM's `--tool-parser-plugin` accepts a module name or a file path (`import_plugin`). | review §3 |

Acceptance: `vllm serve … --tool-call-parser infe_hermes --tool-parser-plugin infe_parsers.shims.vllm` and
`python -m infe_parsers.shims.sglang.launch … --tool-call-parser infe_hermes` both serve a streamed tool call with HTTP 200.

## B. Make the Rust parser output-compatible (the real work)

Measured against stock on identical text (review §2): arguments wrong, duplicated, no id, second call merged.

| # | File | Change |
|---|---|---|
| B1 | `crates/infe-parsers/src/dialects/hermes.rs::extract_name` | Return the `arguments` **sub-object** serialised, not `json_str.to_string()`. Same for `llama3_json.rs` (`parameters`). |
| B2 | `hermes.rs` streaming | Emit `arguments_fragment` as the **diff of the arguments object** (vLLM `hermes_tool_parser.py` semantics: partial-JSON parse, send only newly-stable suffix), not raw bytes including `{"name":`. Do not re-send the buffer on the close marker; the completion delta carries no arguments. |
| B3 | `types.rs` / all dialects | Assign `state.id` (`"call_" + 24 random alnum`, or accept an injected generator so parity tests are deterministic) on the first delta of each call; assign `state.index` and increment per completed call; reset `arguments_buffer`/`name_buffer` after `</tool_call>`. Today neither `id` nor `index` is ever set. |
| B4 | `parser.rs` | `finish()` must close an open tool call the way stock does (emit what is parseable, mark complete). |
| B5 | reasoning | `deepseek_reasoning` must be exposed through the engines' **reasoning** interfaces (vLLM `--reasoning-parser` + `--reasoning-parser-plugin`, class in `vllm.reasoning`; SGLang `ReasoningParser.DetectorMap`), not the tool registry. |

Acceptance: the CPU-only probe in review §2 prints identical `calls=[…]` for `hermes` and `infe_hermes`, non-streaming and
streaming, including two consecutive tool calls; e2e client shows `args_ok == calls == 2×requests` and `has_id == calls`.

| B6 | `hermes.rs` | **Marker-less continuation calls.** After a completed `</tool_call>`, Qwen2.5 (under SGLang's template) emits the next call as bare JSON with no `<tool_call>` opener, then a stray `}`. Stock `qwen25`/`hermes` parse it as call #2; infe streams it as content. Treat a `{` at Idle after ≥1 completed call as a new call (as stock does), and drop the trailing `}`. Raw SSE evidence in review "Round 2" §2. |
| B7 | `types.rs::make_tool_call_id` | Random ids (vLLM: `chatcmpl-tool-<16 hex>`, SGLang: `call_<uuid>`), not index-derived; keep a seedable generator for tests. |
| A6 | manifest + README + driver | vLLM 0.28 `--tool-parser-plugin` = **file path only** (`import_tool_parser` → `import_from_path`); module names work only on `main`. Document both; driver already passes the path. Set `parity.streaming_diff: false` until B2 lands. |

**B2, spelled out** (the remaining core task): inside `InToolCall`, after the `"arguments":` key is seen, run a partial-JSON
validator on the accumulating buffer each feed and emit `arguments_fragment` = newly-stable suffix (what vLLM's
`hermes_tool_parser.py` does with `partial_json_parser`, what SGLang's `BaseFormatDetector` does with `_find_common_prefix`);
emit `name` + `id` as soon as the name field is complete, not at the close marker. Acceptance: same delta *count and
content* as stock on the token-boundary replay in review "Round 2" (stock: 9–20 deltas/call; infe today: 1).

## C. Conformance that would have caught B

- Mine fixtures from `vllm/tests/tool_use/` and `tests/entrypoints/openai/tool_parsers/` and SGLang `test/srt/function_call/`
  (record `source:`); include multi-call, split-marker, nested-JSON-in-string, and reasoning+tool cases.
- Fixture `expected_tool_calls[].arguments` must be set and asserted (currently `null` → skipped). Assert content
  equality, not `contains`. Assert `index` and presence of `id`.
- Add a Python-level parity test that runs both engines' stock parsers and the shim over the fixtures inside the pinned
  engine containers (this is the "parity matrix"; CI job below).

## D. Harness and CI

- `bench/harness/{e2e_tool_stream.py, run_ab_docker.sh, summarize_ab.py}` are committed; wire `run_ab_docker.sh` to
  the wheel-internal shims once A5 lands (today it mounts `$INFE_BENCH_DIR/shims`).
- CI: add a maturin job (`ghcr.io/pyo3/maturin`, abi3 wheel artefact) and a conformance job that installs the wheel into
  `vllm/vllm-openai:<pin>` and `lmsysorg/sglang:<pin>` and runs the parity test; nightly variant against `:latest`
  reporting drift without failing.
- Replace docker-stats CPU sampling with a 1 s psutil sampler on the container's API-server pid; docker stats gave 2–5
  samples per run.
- Delete or relabel `bench/results/{hermes,llama3_json,deepseek_reasoning}_ab.json` (Rust vs a hand-written Python toy).
- `infe-core` is unused by `infe-parsers`; either use it or stop listing it as a dependency and don't describe it as used.
- `registry/infe-parsers/manifest.yaml` exists; correct `streaming_diff`, plugin-path note, and `fixture_count`/`source` as they change (D5).

## E. Reproduce the benchmark (any Linux box with an NVIDIA GPU, Docker, nvidia-container-toolkit)

```
export INFE_BENCH_DIR=~/infe-bench; mkdir -p $INFE_BENCH_DIR/{hf,wheels,shims,results}; chmod 777 $INFE_BENCH_DIR/hf
docker pull vllm/vllm-openai:latest; docker pull lmsysorg/sglang:latest          # 0.28.0 / 0.5.18 on 2026-09-04
docker run --rm -v $PWD:/io -v $INFE_BENCH_DIR/wheels:/out -w /io/python/infe-parsers ghcr.io/pyo3/maturin:latest build --release --out /out
docker run --rm --user $(id -u):$(id -g) -v $INFE_BENCH_DIR/hf:/hf -e HF_HOME=/hf --entrypoint python3 vllm/vllm-openai:latest \
  -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
cp bench/harness/{e2e_tool_stream.py,run_ab_docker.sh} $INFE_BENCH_DIR/; cp bench/results/rtx4090-20260904/scratch/* $INFE_BENCH_DIR/shims/
for spec in "vllm stock 18001" "vllm infe 18001" "sglang stock 18000" "sglang infe 18000"; do set -- $spec
  PORT=$3 ROUNDS=3 $INFE_BENCH_DIR/run_ab_docker.sh $1 $2 <gpu-index> 8 64 256; done
(cd $INFE_BENCH_DIR/results && python3 /path/to/bench/harness/summarize_ab.py "*.json")
```
Gotchas hit on the way: pick a free host port (8000 was taken); the HF cache dir must be writable by the container's
uid; `vllm serve` wants the model positional and no longer accepts `--disable-log-requests`; the maturin container
must write to a mounted `/out`, not a relative path.

## F. Facts verified this session (so nobody re-derives them)

- vLLM 0.28.0 (`vllm/vllm-openai:latest`, sha256:61fc8a89…): `--tool-parser-plugin` → `import_plugin()` accepts module
  name or file path; `ToolParserManager.register_module` is honoured by the new `vllm.parser.parser_manager`;
  `api_server` validates `--tool-call-parser` against `ToolParserManager.list_registered()` when
  `--enable-auto-tool-choice`. Streaming calls `extract_tool_calls_streaming` once per delta.
- SGLang 0.5.18 (`lmsysorg/sglang:latest`, sha256:9e148f5a…): `--tool-call-parser` choices = `["auto"] +
  ToolCallParserEnum.keys()` at parse time; `qwen25` is the stock parser for Qwen2.5 (Hermes-style tags);
  `BaseFormatDetector` requires `structure_info`; `parse_streaming_increment(new_text, tools)` per delta.
- Stock parsers' 46–154 ms ITL p99 spikes are deliberate buffering for partial-JSON validity + argument diffing, not
  CPU cost. A Rust parser must implement the same semantics before its latency can be compared.
