"""Launch SGLang with the infe shim registered before ServerArgs parses --tool-call-parser.
File-based (not -c/stdin) so multiprocessing spawn children can re-import __main__ safely."""
import sys, runpy
if __name__ == "__main__":
    sys.path.insert(0, "/shims")
    import sglang_shim  # noqa: F401  registers infe_* detectors
    sys.argv = ["sglang.launch_server"] + sys.argv[1:]
    runpy.run_module("sglang.launch_server", run_name="__main__")
