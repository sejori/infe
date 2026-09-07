"""SGLang parity probe: stock qwen25 vs infe_hermes over real token chunks.

Covers B6 (marker-less continuation call) and B8 (nameless argument fragments).
"""
import infe_parsers_sglang  # noqa: F401  registers infe_* detectors
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool, Function
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tools = [
    Tool(type="function", function=Function(name="get_weather", parameters={"type": "object"})),
    Tool(type="function", function=Function(name="get_time", parameters={"type": "object"})),
]

CASES = {
    "both_markers": (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London", "units": "celsius"}}\n</tool_call>'
        '\n<tool_call>\n{"name": "get_time", "arguments": {"city": "London"}}\n</tool_call>'
    ),
    # B6: second call arrives as bare JSON with no opener (what Qwen2.5 actually emits under SGLang)
    "markerless_second": (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London", "units": "celsius"}}\n</tool_call>'
        '\n{"name": "get_time", "arguments": {"city": "London"}}\n}'
    ),
}

for case, out in CASES.items():
    chunks = [tok.decode([i]) for i in tok.encode(out, add_special_tokens=False)]
    print("=== case: %s (%d token chunks)" % (case, len(chunks)))
    for name in ["qwen25", "infe_hermes"]:
        p = FunctionCallParser(tools, name)
        normal = ""
        items = []
        for c in chunks:
            nt, cs = p.parse_stream_chunk(c)
            normal += nt
            items += [(x.tool_index, x.name, x.parameters) for x in cs]
        try:
            nt, cs = p.parse_stream_end()
            normal += nt
            items += [(x.tool_index, x.name, x.parameters) for x in cs]
        except AttributeError:
            pass
        acc = {}
        for idx, nm, params in items:
            a = acc.setdefault(idx, {"name": None, "args": ""})
            if nm:
                a["name"] = nm
            if params:
                a["args"] += params
        print("  %-12s items=%-3d calls=%d normal=%r" % (name, len(items), len(acc), normal))
        for k, v in sorted(acc.items()):
            print("      idx=%s name=%s args=%r" % (k, v["name"], v["args"]))
