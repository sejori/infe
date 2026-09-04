"""Launch SGLang with the infe shim registered before ServerArgs parses --tool-call-parser.

SGLang validates --tool-call-parser against FunctionCallParser.ToolCallParserEnum
at arg-parse time, so the shim must be imported into the server process *before*
ServerArgs.add_cli_args runs. SGLang has no plugin flag, so we wrap the launch:

    python -m infe_parsers_sglang.launch -- <sglang args> --tool-call-parser infe_hermes

File-based (not -c/stdin) so multiprocessing spawn children can re-import
__main__ safely.
"""
import sys
import runpy

if __name__ == "__main__":
    # Register infe detectors before SGLang parses CLI args
    import infe_parsers_sglang  # noqa: F401
    sys.argv = ["sglang.launch_server"] + sys.argv[1:]
    runpy.run_module("sglang.launch_server", run_name="__main__")
