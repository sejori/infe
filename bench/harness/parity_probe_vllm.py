import infe_parsers_vllm  # noqa: F401  registers infe_* parsers
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
    '\n<tool_call>\n{"name": "get_time", "arguments": {"city": "London"}}\n</tool_call>'
)
chunks = [tok.decode([i]) for i in tok.encode(out, add_special_tokens=False)]


def fget(f, k):
    if f is None:
        return None
    return f.get(k) if isinstance(f, dict) else getattr(f, k, None)


for name in ["hermes", "infe_hermes"]:
    p = ToolParserManager.get_tool_parser(name)(tok)
    prev = cur = ""
    n = 0
    acc = {}
    content = ""
    seq = []
    for c in chunks:
        prev, cur = cur, cur + c
        d = p.extract_tool_calls_streaming(prev, cur, c, [], [], [], req)
        if d is None:
            continue
        n += 1
        if d.content:
            content += d.content
        for t in (d.tool_calls or []):
            a = acc.setdefault(t.index, {"id": None, "name": None, "args": ""})
            if t.id:
                a["id"] = t.id
            nm = fget(t.function, "name")
            if nm:
                a["name"] = nm
            arg = fget(t.function, "arguments")
            if arg:
                a["args"] += arg
                seq.append((t.index, arg))
    print("%s: deltas=%d arg_fragments=%d content=%r" % (name, n, len(seq), content))
    for k, v in sorted(acc.items()):
        print("    idx=%s name=%s id=%s args=%r" % (k, v["name"], bool(v["id"]), v["args"]))
    print("    first 6 fragments: %s" % (seq[:6],))
