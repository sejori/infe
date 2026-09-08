"""infe-sched profiling plugin for SGLang.

Two load paths:
  1. PYTHONPATH=/path/to/infe_sched_probe + SGLANG_PLUGINS=infe_sched_probe
  2. pip install -e /path/to/infe_sched_probe + SGLANG_PLUGINS=infe_sched_probe

Both cause SGLang's load_plugins() → entry_point.load() → import plugin →
register AROUND hooks on Scheduler.get_next_batch_to_run / run_batch /
process_batch_result.

When SGLANG_SCHED_TIMER_OUT is set, per-phase wall-clock time is accumulated
and written as JSON at exit. When unset, hooks are zero-overhead no-ops.
"""
from infe_sched_probe.plugin import *  # noqa: F401,F403  (triggers registration)
