"""SGLang tool-parser shim for infe-parsers.

This module registers infe-parsers dialects as SGLang detectors via the
FunctionCallParser.ToolCallParserEnum registry. No fork or patch to
SGLang source is required — this is a pure plugin loaded at import time.

Note: This package is named `infe_parsers_sglang` (not `infe_parsers`) to
avoid shadowing the real wheel on `sys.path`.

Usage:
    # Register the infe detectors (do this before starting the server,
    # or via the launcher module).
    import infe_parsers_sglang  # noqa: F401

    # Then start SGLang with:
    #   --tool-call-parser infe_hermes

The shim creates an infe_parsers.StreamingParser per request and translates
between SGLang's StreamingParseResult/ToolCallItem types and the Rust
parser's delta dicts. The heavy lifting happens in Rust.

This file is intentionally small and easy to regenerate when SGLang
restructures — that is the entire point of the shim layer (BRIEF.md §5.2).
"""

import json
import logging

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from infe_parsers import StreamingParser as RustStreamingParser

logger = logging.getLogger(__name__)

# Mapping from SGLang parser names to infe dialect names.
# Note: deepseek_reasoning is a reasoning parser, not a tool parser —
# it should be registered via the reasoning-parser interface.
_INFE_DIALECT_MAP = {
    "infe_hermes": "hermes",
    "infe_llama3_json": "llama3_json",
}


class InfeDetector(BaseFormatDetector):
    """SGLang detector backed by the Rust infe-parsers crate.

    Implements the BaseFormatDetector interface so SGLang's
    FunctionCallParser can use it transparently. The parser state lives
    in the Rust side — this class just forwards calls and translates types.
    """

    # The infe dialect name (set by subclass).
    _infe_dialect: str = "hermes"

    def __init__(self, tokenizer=None):
        super().__init__()
        self._rust_parser: RustStreamingParser | None = None
        # B8: Track the tool_index and name of the currently-open tool call
        # so nameless argument-fragment deltas can be forwarded with the
        # correct index.  SGLang's _process_tool_call_stream treats
        # name is None as "argument delta for the call at tool_index".
        self._open_tool_index: int = -1
        self._open_tool_name: str | None = None
        if tokenizer is not None:
            self.model_tokenizer = tokenizer
        # Set bot/eot tokens from the dialect for SGLang's has_tool_call().
        if self._infe_dialect == "hermes":
            self.bot_token = "\u003ctool_call\u003e"
            self.eot_token = "\u003c/tool_call\u003e"
        elif self._infe_dialect == "llama3_json":
            self.bot_token = ""
            self.eot_token = ""
        else:
            self.bot_token = ""
            self.eot_token = ""

    def _ensure_parser(self):
        """Lazily create the Rust parser on first use."""
        if self._rust_parser is None:
            self._rust_parser = RustStreamingParser(self._infe_dialect)

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call marker."""
        if self._infe_dialect == "hermes":
            return self.bot_token in text
        elif self._infe_dialect == "llama3_json":
            return '"name"' in text and text.strip().startswith("{")
        else:
            return False

    def detect_and_parse(
        self, text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """One-shot parse of the full text.

        With incremental streaming, arguments are emitted as multiple diff
        fragments across deltas. We accumulate by index.
        """
        self._ensure_parser()
        self._rust_parser.reset()

        result = self._rust_parser.feed(text)
        finish_result = self._rust_parser.finish()

        all_tool_calls = list(result.get("tool_calls", []))
        all_tool_calls.extend(finish_result.get("tool_calls", []))
        all_content = list(result.get("content", []))
        all_content.extend(finish_result.get("content", []))

        if not all_tool_calls:
            return StreamingParseResult(normal_text=text)

        tool_indices = self._get_tool_indices(tools)
        # Accumulate by index: name from first delta, args from all fragments.
        acc: dict[int, dict] = {}
        completed: set[int] = set()
        for tc in all_tool_calls:
            idx = tc.get("index", 0)
            entry = acc.setdefault(idx, {"name": "", "args": ""})
            if tc.get("name"):
                entry["name"] = tc["name"]
            entry["args"] += tc.get("arguments_fragment", "")
            if tc.get("is_complete"):
                completed.add(idx)

        calls = []
        for idx in sorted(acc):
            if idx not in completed:
                continue
            entry = acc[idx]
            name = entry["name"]
            args_str = entry["args"] or "{}"
            try:
                args_json = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_json = {}
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(args_json, ensure_ascii=False),
                )
            )

        normal_text = "".join(all_content) if all_content else ""
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """Streaming incremental parse.

        B8 fix: nameless argument-fragment deltas (no name field) are
        now forwarded as ToolCallItem(tool_index=<open call>, name=None,
        parameters=frag).  Previously they were silently dropped by the
        elif name: guard, causing SGLang to receive only 2 deltas per
        call instead of ~5 and inflating ITL p99 by 90-98%.

        SGLang's _process_tool_call_stream already handles
        call_item.name is None as an argument delta for the call at
        tool_index, so no SGLang-side change is needed.
        """
        self._ensure_parser()
        result = self._rust_parser.feed(new_text)

        tool_calls = result.get("tool_calls", [])
        content_parts = result.get("content", [])
        reasoning_parts = result.get("reasoning", [])

        tool_indices = self._get_tool_indices(tools)
        calls = []
        for tc in tool_calls:
            name = tc.get("name")
            args_frag = tc.get("arguments_fragment", "")
            idx = tc.get("index", 0)
            is_complete = tc.get("is_complete", False)

            if is_complete:
                # Complete call: send name + full arguments.
                # Resolve tool_index from the name if we have one,
                # otherwise from the open-call tracker.
                resolved_name = name or self._open_tool_name
                resolved_idx = idx
                if resolved_name and resolved_name in tool_indices:
                    resolved_idx = tool_indices[resolved_name]
                elif name and name in tool_indices:
                    resolved_idx = tool_indices[name]
                else:
                    resolved_idx = idx

                calls.append(
                    ToolCallItem(
                        tool_index=resolved_idx,
                        name=resolved_name,
                        parameters=args_frag if args_frag else "{}",
                    )
                )
                # Reset the open-call tracker.
                self._open_tool_index = -1
                self._open_tool_name = None
            elif name is not None:
                # First delta for this call: carries name (+ id on the Rust
                # side).  Record the tool_index so subsequent nameless
                # arg-fragment deltas can be forwarded correctly.
                resolved_idx = tool_indices.get(name, idx)
                self._open_tool_index = resolved_idx
                self._open_tool_name = name
                calls.append(
                    ToolCallItem(
                        tool_index=resolved_idx,
                        name=name,
                        parameters=args_frag,
                    )
                )
            else:
                # B8: nameless argument-fragment delta.  Forward it with
                # name=None so SGLang treats it as an argument delta for
                # the currently-open call at _open_tool_index.
                if self._open_tool_index >= 0:
                    calls.append(
                        ToolCallItem(
                            tool_index=self._open_tool_index,
                            name=None,
                            parameters=args_frag,
                        )
                    )
                # If no call is open, drop the fragment (shouldn't happen
                # in well-formed streams, but guard against it).

        normal_text = "".join(content_parts) if content_parts else ""
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self):
        """Required abstract method (constrained decoding for tool_choice).
        Mirrors HermesDetector."""
        from sglang.srt.function_call.core_types import StructureInfo
        if self._infe_dialect == "hermes":
            return lambda name: StructureInfo(
                begin='\u003ctool_call\u003e{"name":"' + name + '", "arguments":',
                end='}\u003c/tool_call\u003e',
                trigger='\u003ctool_call\u003e')
        return lambda name: StructureInfo(
            begin='{"name":"' + name + '", "parameters":',
            end='}',
            trigger='')

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Flush any buffered content at end of stream."""
        self._ensure_parser()
        result = self._rust_parser.finish()

        content_parts = result.get("content", [])
        normal_text = "".join(content_parts) if content_parts else ""
        # Reset the open-call tracker.
        self._open_tool_index = -1
        self._open_tool_name = None
        return StreamingParseResult(normal_text=normal_text)


def _make_detector_class(dialect: str, sglang_name: str) -> type[InfeDetector]:
    """Create a detector subclass with the dialect baked in."""
    return type(
        sglang_name,
        (InfeDetector,),
        {"_infe_dialect": dialect},
    )


# Register all tool-parser dialects with SGLang's FunctionCallParser registry.
for _sglang_name, _dialect in _INFE_DIALECT_MAP.items():
    _cls = _make_detector_class(_dialect, _sglang_name)
    FunctionCallParser.ToolCallParserEnum[_sglang_name] = _cls
    logger.info("Registered infe SGLang detector: %s (dialect=%s)", _sglang_name, _dialect)
