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

    def _ensure_parser(self):
        """Lazily create the Rust parser on first use."""
        if self._rust_parser is None:
            self._rust_parser = RustStreamingParser(self._infe_dialect)

    def _result_to_delta_message(
        self, result: dict, request: ChatCompletionRequest
    ) -> DeltaMessage | None:
        """Translate the Rust parser's result dict into a vLLM DeltaMessage."""
        tool_calls = result.get("tool_calls", [])
        content_parts = result.get("content", [])
        reasoning_parts = result.get("reasoning", [])

        content = None
        if content_parts:
            content = "".join(content_parts)

        delta_tool_calls: list[DeltaToolCall] = []
        for tc in tool_calls:
            # vLLM expects the first delta to carry the tool name + id.
            function_kwargs: dict = {}
            if tc.get("name"):
                function_kwargs["name"] = tc["name"]
            if tc.get("arguments_fragment"):
                function_kwargs["arguments"] = tc["arguments_fragment"]
            if tc.get("id"):
                function_kwargs["id"] = tc["id"]

            delta_tool_calls.append(
                DeltaToolCall(
                    index=tc.get("index", 0),
                    type="function",
                    function=DeltaFunctionCall(**function_kwargs).model_dump(
                        exclude_none=True
                    ),
                )
            )

        # If we have reasoning, put it in the reasoning_content field.
        reasoning_content = None
        if reasoning_parts:
            reasoning_content = "".join(r["fragment"] for r in reasoning_parts)

        if content or delta_tool_calls or reasoning_content:
            return DeltaMessage(
                content=content,
                tool_calls=delta_tool_calls,  # must be a list, not None (vLLM 0.28)
                reasoning_content=reasoning_content,
            )
        return None

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Non-streaming: feed the entire model output and collect results."""
        self._ensure_parser()
        self._rust_parser.reset()

        result = self._rust_parser.feed(model_output)
        finish_result = self._rust_parser.finish()

        # Merge
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

        tool_calls = []
        for tc in all_tool_calls:
            if not tc.get("is_complete"):
                continue
            args_str = tc.get("arguments_fragment", "{}")
            try:
                args_json = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_json = {}
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=tc.get("name") or "",
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
        """Streaming: feed only the new delta text, translate the result."""
        self._ensure_parser()
        result = self._rust_parser.feed(delta_text)
        return self._result_to_delta_message(result, request)


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
