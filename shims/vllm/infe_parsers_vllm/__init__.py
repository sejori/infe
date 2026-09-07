"""vLLM tool-parser shim for infe-parsers.

This module is loaded by vLLM via `--tool-parser-plugin`, which makes the
`InfeHermesToolParser` (and other dialects) available as named tool parsers.
No fork or patch to vLLM source is required — this is a pure plugin.

Usage:
    vllm serve <model> \\
        --tool-call-parser infe_hermes \\
        --tool-parser-plugin infe_parsers_vllm

The shim creates one `infe_parsers.StreamingParser` per request and translates
between vLLM's `DeltaMessage`/`ExtractedToolCallInformation` types and the
Rust parser's delta dicts. The heavy lifting (marker detection, JSON
streaming, dialect state machines) all happens in Rust.

This file is intentionally small and easy to regenerate when vLLM
restructures — that is the entire point of the shim layer (see BRIEF.md §5.2).

Note: This package is named `infe_parsers_vllm` (not `infe_parsers`) to
avoid shadowing the real wheel on `sys.path`. vLLM loads it via
`--tool-parser-plugin <path-to-this-dir>`.
"""

import json
import logging
from collections import deque

try:  # vLLM main (2026-09) moved these; releases <=0.28 keep them under openai.engine
    from vllm.entrypoints.generate.base.protocol import (
        DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall,
    )
except ModuleNotFoundError:
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall,
    )
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.tool_parsers.utils import Tool
from vllm.tokenizers import TokenizerLike

from infe_parsers import StreamingParser as RustStreamingParser

logger = logging.getLogger(__name__)

# Mapping from vLLM parser names to infe dialect names.
# Note: deepseek_reasoning is a reasoning parser, not a tool parser —
# it should be registered via the reasoning-parser interface, not here.
_INFE_DIALECT_MAP = {
    "infe_hermes": "hermes",
    "infe_llama3_json": "llama3_json",
}


class _PendingDeltas:
    """FIFO queue of tool-call deltas and content fragments waiting to be sent.

    vLLM calls `extract_tool_calls_streaming` once per decoded token batch and
    expects at most one `DeltaMessage` in return.  The Rust parser, however,
    may produce multiple deltas in a single `feed()` call (e.g. name+id, an
    argument diff, and a completion delta all at once).  We buffer the excess
    here and drip-feed them to vLLM one at a time, so the SSE client sees the
    same fine-grained streaming as the stock parser.
    """

    def __init__(self):
        self.tool_calls: deque[dict] = deque()
        self.content_parts: deque[str] = deque()
        self.reasoning_parts: deque[str] = deque()

    def extend(self, result: dict) -> None:
        """Append all deltas from a Rust parser result."""
        for tc in result.get("tool_calls", []):
            self.tool_calls.append(tc)
        for content in result.get("content", []):
            self.content_parts.append(content)
        for r in result.get("reasoning", []):
            self.reasoning_parts.append(r["fragment"])

    def pop_delta_message(self) -> DeltaMessage | None:
        """Build the next single DeltaMessage, or None if the queue is empty.

        Each call returns at most one tool-call delta OR one content fragment
        OR one reasoning fragment, matching vLLM's per-token streaming
        behaviour.  Tool-call deltas get priority so name+id arrive before
        arguments.
        """
        if self.tool_calls:
            tc = self.tool_calls.popleft()
            function_kwargs: dict = {}
            if tc.get("name"):
                function_kwargs["name"] = tc["name"]
            if tc.get("arguments_fragment"):
                function_kwargs["arguments"] = tc["arguments_fragment"]
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=tc.get("index", 0),
                        id=tc.get("id") or None,
                        type="function",
                        function=DeltaFunctionCall(
                            **function_kwargs
                        ).model_dump(exclude_none=True),
                    )
                ],
            )

        if self.content_parts:
            content = self.content_parts.popleft()
            if content:
                return DeltaMessage(content=content, tool_calls=[])

        if self.reasoning_parts:
            fragment = self.reasoning_parts.popleft()
            if fragment:
                return DeltaMessage(
                    reasoning_content=fragment, tool_calls=[]
                )

        return None


class InfeToolParser(ToolParser):
    """vLLM ToolParser backed by the Rust infe-parsers crate.

    Each dialect (hermes, llama3_json) is registered as a separate vLLM
    parser name. Internally they all create an `infe_parsers.StreamingParser`
    with the appropriate dialect.
    """

    # The infe dialect name this parser instance uses.
    _infe_dialect: str = "hermes"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # One Rust parser per request — created fresh, fed deltas.
        self._rust_parser: RustStreamingParser | None = None
        # Buffered deltas waiting to be sent across successive
        # extract_tool_calls_streaming calls.
        self._pending = _PendingDeltas()

    def _ensure_parser(self):
        """Lazily create the Rust parser on first use."""
        if self._rust_parser is None:
            self._rust_parser = RustStreamingParser(self._infe_dialect)

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Non-streaming: feed the entire model output and collect results.

        With incremental streaming, arguments are emitted as multiple
        diff fragments across deltas. We accumulate by index and also
        collect the name (which appears on the first delta for each call).
        """
        self._ensure_parser()
        self._rust_parser.reset()

        result = self._rust_parser.feed(model_output)
        finish_result = self._rust_parser.finish()

        # Merge all deltas.
        all_tool_calls = list(result.get("tool_calls", []))
        all_tool_calls.extend(finish_result.get("tool_calls", []))
        all_content = list(result.get("content", []))
        all_content.extend(finish_result.get("content", []))

        if not all_tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        # Accumulate by index: name from first delta, args from all fragments.
        acc: dict[int, dict] = {}
        for tc in all_tool_calls:
            idx = tc.get("index", 0)
            entry = acc.setdefault(idx, {"name": "", "args": ""})
            if tc.get("name"):
                entry["name"] = tc["name"]
            entry["args"] += tc.get("arguments_fragment", "")

        tool_calls = []
        for idx in sorted(acc):
            entry = acc[idx]
            args_str = entry["args"] or "{}"
            try:
                args_json = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_json = {}
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=entry["name"],
                        arguments=json.dumps(args_json, ensure_ascii=False),
                    ),
                )
            )

        content_str = "".join(all_content) if all_content else None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content_str,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Streaming: feed only the new delta text, translate the result.

        The Rust parser may produce multiple deltas per feed() call. We
        buffer them and return one DeltaMessage per call, so the SSE
        stream mirrors the stock parser's per-token granularity.
        """
        self._ensure_parser()

        # If we have buffered deltas, drain them first before feeding new
        # text. This ensures ordering: previous deltas are sent before
        # new ones are produced.
        if not self._pending.tool_calls and not self._pending.content_parts and not self._pending.reasoning_parts:
            # Nothing buffered — feed new text and populate the queue.
            result = self._rust_parser.feed(delta_text)
            self._pending.extend(result)

        msg = self._pending.pop_delta_message()
        if msg is not None:
            return msg

        # Queue was empty and feed produced nothing — feed the delta text
        # now and try once more.
        result = self._rust_parser.feed(delta_text)
        self._pending.extend(result)
        return self._pending.pop_delta_message()


def _make_parser_class(dialect: str, vllm_name: str) -> type[InfeToolParser]:
    """Create a subclass with the dialect baked in."""
    return type(
        vllm_name,
        (InfeToolParser,),
        {"_infe_dialect": dialect},
    )


# Register all tool-parser dialects with vLLM's ToolParserManager.
# Note: reasoning dialects (deepseek_reasoning) are registered via the
# reasoning-parser interface, not the tool registry.
for _vllm_name, _dialect in _INFE_DIALECT_MAP.items():
    _cls = _make_parser_class(_dialect, _vllm_name)
    ToolParserManager.register_module(_vllm_name, module=_cls)
    logger.info("Registered infe tool parser: %s (dialect=%s)", _vllm_name, _dialect)
