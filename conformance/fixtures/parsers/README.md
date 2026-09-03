# infe-parsers conformance fixtures

Each fixture is a JSON file with this shape:

```json
{
  "dialect": "hermes",
  "input_chunks": ["chunk1", "chunk2", ...],
  "expected_tool_calls": [
    {"name": "fn", "arguments": "{...}", "index": 0}
  ],
  "expected_reasoning": [
    {"fragment": "...", "is_complete": false}
  ],
  "expected_content": ["plain text..."]
}
```

Fixtures are mined from vLLM and SGLang parser test suites. The source of
each fixture is recorded in a `source` field (e.g. `"vllm:test_tool_call_hermes"`
or `"sglang:test_hermes_parser"`).

The conformance runner feeds `input_chunks` sequentially into a
`StreamingParser` and asserts that the accumulated deltas match the expected
output. 100% pass rate is the M1 acceptance criterion (BRIEF §6.1).
