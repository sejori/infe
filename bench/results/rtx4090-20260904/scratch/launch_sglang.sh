#!/bin/bash
pip install -q /wheels/*.whl
exec python3 /shims/launch_sglang.py "$@"
