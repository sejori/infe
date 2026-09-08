#!/usr/bin/env bash
# Drive the infe-sched M0 probe: run SGLang with the phase-timer plugin
# enabled, alongside a stock control arm, across the three workload modes.
#
# The timer plugin is loaded via PYTHONPATH (no pip install needed — mirrors
# the infe-parsers shim approach). SGLANG_SCHED_TIMER_OUT triggers data
# collection; without it the hooks are zero-overhead no-ops.
#
# Usage:
#   INFE_BENCH_DIR=~/infe-bench PORT=18000 bash run_sched_probe.sh <gpu> [concurrency...]
#
# Produces:
#   $B/results/sched_probe_<mode>_<arm>_<timestamp>.json      — e2e metrics
#   $B/results/sched_probe_<mode>_<arm>_<timestamp>.cpu.txt   — CPU samples
#   $B/results/sched_probe_<mode>_<arm>_<timestamp>.timer.json — phase timing (probe arm only)
#
# Workload modes: mixed_lengths, short_burst, tool_call
# Arms: stock (no plugin, no timer) | probe (timer plugin loaded)
#
# The probe arm and stock arm use the SAME SGLang image and flags — the only
# difference is SGLANG_PLUGINS=infe_sched_probe + SGLANG_SCHED_TIMER_OUT.
# This ensures the timer overhead is the only variable.
set -euo pipefail

GPU=${1:-0}; shift; CONC=${*:-"8 64 256"}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}; PORT=${PORT:-8000}
B=${INFE_BENCH_DIR:-$HOME/infe-bench}; HF=$B/hf
SGLANG_IMG_TAG=${SGLANG_IMAGE_TAG:-v0.5.19}
IMG=lmsysorg/sglang:$SGLANG_IMG_TAG
HARNESS="$(cd "$(dirname "$0")" && pwd)"
REPO=${INFE_REPO:-$(cd "$HARNESS/../.." 2>/dev/null && pwd || echo "$HOME/infe")}
ROUNDS=${ROUNDS:-3}
MODES=${MODES:-"mixed_lengths short_burst tool_call"}

ARGS=(--model-path $MODEL --port 8000 --host 0.0.0.0 --context-length 4096 --mem-fraction-static 0.85)
COMMON_OPTS="--gpus device=$GPU --ipc=host -p 127.0.0.1:$PORT:8000 -v $HF:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -v $REPO/shims:/shims:ro"

mkdir -p "$B/results"

run_arm() {
  local mode=$1 arm=$2
  local OUT="$B/results/sched_probe_${mode}_${arm}_$(date +%Y%m%d-%H%M%S)"
  local NAME="infe-sched-$mode-$arm"

  docker rm -f "$NAME" >/dev/null 2>&1 || true

  if [ "$arm" = "probe" ]; then
    # Timer plugin loaded via PYTHONPATH. SGLANG_PLUGINS whitelists it.
    # SGLANG_SCHED_TIMER_OUT triggers data collection.
    docker run -d --name "$NAME" $COMMON_OPTS \
      -e PYTHONPATH=/shims/sglang/infe_sched_probe \
      -e SGLANG_PLUGINS=infe_sched_probe \
      -e SGLANG_SCHED_TIMER_OUT=/tmp/timer.json \
      "$IMG" python3 -m sglang.launch_server "${ARGS[@]}" --tool-call-parser qwen25 >/dev/null
  else
    # Stock control: same image, same flags, no plugin.
    docker run -d --name "$NAME" $COMMON_OPTS \
      "$IMG" python3 -m sglang.launch_server "${ARGS[@]}" --tool-call-parser qwen25 >/dev/null
  fi

  echo "waiting for $NAME on :$PORT"
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || {
    echo "server failed to start"; docker logs --tail 60 "$NAME"; docker rm -f "$NAME"; exit 1
  }

  # Verify timer plugin loaded (probe arm only)
  if [ "$arm" = "probe" ]; then
    docker logs "$NAME" 2>&1 | grep -i "infe.sched\|hook\|Registered.*infe" | head -5 || true
  fi

  # CPU sampler
  python3 "$HARNESS/cpu_sampler.py" --container-name "$NAME" --output "${OUT}.cpu.txt" --interval 1.0 &
  local SAMPLER_PID=$!

  # Run the workload
  local SHARED_PREFIX="short"
  if [ "$mode" = "mixed_lengths" ]; then
    SHARED_PREFIX="long"
  fi

  python3 "$HARNESS/e2e_high_admission.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --arm "$arm" \
    --engine sglang \
    --mode "$mode" \
    --shared-prefix "$SHARED_PREFIX" \
    --concurrency $CONC \
    --requests "$ROUNDS" \
    --output "${OUT}.json"

  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true

  # Extract timer data (probe arm only)
  if [ "$arm" = "probe" ]; then
    docker cp "$NAME:/tmp/timer.json" "${OUT}.timer.json" 2>/dev/null || \
      echo "WARNING: timer.json not found — check if plugin loaded"
    echo "=== timer summary ==="
    python3 -c "
import json, sys
try:
    with open('${OUT}.timer.json') as f:
        r = json.load(f)
    f = r.get('fractions', {})
    s = r.get('get_next_batch_to_run', {})
    fwd = r.get('run_batch', {})
    res = r.get('process_batch_result', {})
    steps = r.get('step_span', {}).get('calls', 0)
    print(f'  steps={steps}')
    print(f'  sched:   {f.get(\"sched_of_step_pct\", 0):5.1f}% of step (mean {s.get(\"mean_ns\", 0)/1000:.0f}µs, p99 {s.get(\"p99_ns\", 0)/1000:.0f}µs)')
    print(f'  forward: {f.get(\"forward_of_step_pct\", 0):5.1f}% of step (mean {fwd.get(\"mean_ns\", 0)/1000:.0f}µs)')
    print(f'  result:  {f.get(\"result_of_step_pct\", 0):5.1f}% of step (mean {res.get(\"mean_ns\", 0)/1000:.0f}µs)')
    sched_pct = f.get('sched_of_step_pct', 0)
    if sched_pct < 5.0:
        print(f'  >>> KILL CRITERION: sched={sched_pct:.1f}% < 5% — scheduler NOT on critical path')
    else:
        print(f'  >>> sched={sched_pct:.1f}% >= 5% — scheduler IS a meaningful fraction of step time')
except FileNotFoundError:
    print('  timer.json not found')
except Exception as e:
    print(f'  error: {e}')
"
  fi

  echo "cpu samples: $(wc -l < "${OUT}.cpu.txt" 2>/dev/null || echo 0)  mean: $(sed 's/%//' "${OUT}.cpu.txt" 2>/dev/null | awk '{s+=$1;n++} END{if(n) printf "%.0f%%", s/n; else print "N/A"}')"
  docker rm -f "$NAME" >/dev/null
  echo "done -> ${OUT}.json"
  echo ""
}

# Run each mode × arm, interleaved: stock, probe, stock, probe ...
# (interleaving defeats thermal/clock drift, same as run_ab_docker.sh)
for mode in $MODES; do
  echo "===== mode: $mode ====="
  for arm in stock probe; do
    run_arm "$mode" "$arm"
  done
done

echo ""
echo "========== SUMMARY =========="
echo "Raw results in $B/results/sched_probe_*"
echo ""
echo "To compare arms:"
echo "  cd $B/results && python3 $HARNESS/summarize_sched_probe.py 'sched_probe_*.json'"
