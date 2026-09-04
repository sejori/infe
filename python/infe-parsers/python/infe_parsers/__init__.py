"""infe-parsers: Rust-backed streaming tool-call and reasoning parsers.

This package exposes the Rust `infe-parsers` crate to Python via PyO3.
The streaming parser is called once per engine step with a batch of
decoded token-text chunks, producing structured deltas (tool-call /
reasoning / content) without per-token Python calls.

Usage:
    from infe_parsers import StreamingParser, list_dialects

    parser = StreamingParser("hermes")
    result = parser.feed('...decoded text...')
    # result = {"tool_calls": [...], "reasoning": [...], "content": [...]}
    final = parser.finish()

To check whether a dialect is a tool or reasoning parser:
    from infe_parsers import dialect_kind
    assert dialect_kind("hermes") == "tool"
    assert dialect_kind("deepseek_reasoning") == "reasoning"
"""
from ._infe_parsers import (
    StreamingParser,
    list_dialects,
    list_tool_dialects,
    list_reasoning_dialects,
    dialect_kind,
)

__all__ = [
    "StreamingParser",
    "list_dialects",
    "list_tool_dialects",
    "list_reasoning_dialects",
    "dialect_kind",
]
__version__ = "0.1.0"
