"""Demonstrate the buffered-shim text-loss bug: count feed() calls vs chunks fed."""
import infe_parsers_vllm  # noqa: F401
from vllm.tool_parsers import ToolParserManager
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
req = ChatCompletionRequest(
    model="m",
    messages=[{"role": "user", "content": "hi"}],
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
)
out = (
    'I will check.\n<tool_call>\n{"name": "get_weather", "arguments": '
    '{"city": "London", "units": "celsius"}}\n</tool_call>'
)
chunks = [tok.decode([i]) for i in tok.encode(out, add_special_tokens=False)]

p = ToolParserManager.get_tool_parser("infe_hermes")(tok)
p._ensure_parser()
fed = []


class TracingParser:
    """Proxy around the PyO3 parser (its attributes are read-only)."""

    def __init__(self, inner):
        self._inner = inner

    def feed(self, text):
        fed.append(text)
        return self._inner.feed(text)

    def __getattr__(self, name):
        return getattr(self._inner, name)


p._rust_parser = TracingParser(p._rust_parser)

for i, c in enumerate(chunks):
    prev = "".join(chunks[:i])
    cur = prev + c
    p.extract_tool_calls_streaming(prev, cur, c, [], [], [], req)

print("chunks passed to shim : %d" % len(chunks))
print("chunks reaching parser: %d" % len(fed))
lost = [c for c in chunks if chunks.count(c) > fed.count(c)]
print("text fed to parser    : %r" % "".join(fed))
print("text the model emitted: %r" % "".join(chunks))
print("MATCH" if "".join(fed) == "".join(chunks) else "*** TEXT LOST ***")
