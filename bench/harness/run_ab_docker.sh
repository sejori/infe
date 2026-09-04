#!/usr/bin/env bash
# Drive one arm (stock|infe) of one engine (vllm|sglang) in Docker on a chosen GPU, run the
# e2e client, snapshot container CPU, tear down. Any Linux host with Docker + the NVIDIA container toolkit. Layout under $INFE_BENCH_DIR (default ~/infe-bench): hf/ wheels/ shims/ results/ plus this script and e2e_tool_stream.py.
# Usage: INFE_BENCH_DIR=... PORT=18000 ROUNDS=3 run_ab_docker.sh <engine> <arm> <gpu> [concurrency list...]
set -euo pipefail
ENGINE=$1; ARM=$2; GPU=$3; shift 3; CONC=${*:-"8 64 256"}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}; PORT=${PORT:-8000}; ROUNDS=${ROUNDS:-3}
B=${INFE_BENCH_DIR:-$HOME/infe-bench}; HF=$B/hf; SHIMS=$B/shims
NAME=infe-$ENGINE-$ARM; OUT=$B/results/${ENGINE}_${ARM}_$(date +%Y%m%d-%H%M%S).json
docker rm -f $NAME >/dev/null 2>&1 || true
COMMON=(--name $NAME --gpus "device=$GPU" --ipc=host -p 127.0.0.1:$PORT:8000 -v $HF:/hf -e HF_HOME=/hf -v $B/wheels:/wheels:ro -v $SHIMS:/shims:ro -e HF_HUB_OFFLINE=1)
if [ $ENGINE = vllm ]; then
  IMG=vllm/vllm-openai:latest
  ARGS=($MODEL --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-auto-tool-choice)
  if [ $ARM = stock ]; then
    docker run -d "${COMMON[@]}" $IMG "${ARGS[@]}" --tool-call-parser hermes >/dev/null
  else
    docker run -d "${COMMON[@]}" --entrypoint bash $IMG -c "pip install -q /wheels/*.whl && python3 -c 'import infe_parsers; print(\"infe_parsers\", infe_parsers.__version__)' && exec vllm serve ${ARGS[*]} --tool-call-parser infe_hermes --tool-parser-plugin /shims/vllm_shim.py" >/dev/null
  fi
else
  IMG=lmsysorg/sglang:latest
  ARGS=(--model-path $MODEL --port 8000 --host 0.0.0.0 --context-length 4096 --mem-fraction-static 0.85)
  if [ $ARM = stock ]; then
    docker run -d "${COMMON[@]}" $IMG python3 -m sglang.launch_server "${ARGS[@]}" --tool-call-parser qwen25 >/dev/null
  else
    docker run -d "${COMMON[@]}" $IMG bash /shims/launch_sglang.sh "${ARGS[@]}" --tool-call-parser infe_hermes >/dev/null
  fi
fi
echo "waiting for $NAME on :$PORT"; for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "server failed to start"; docker logs --tail 60 $NAME; docker rm -f $NAME; exit 1; }
docker logs $NAME 2>&1 | grep -iE "infe|version|Registered" | head -5 || true
# CPU sampler: docker stats every 2s during the run
( while docker ps -q -f name=$NAME >/dev/null 2>&1 && [ -f $B/results/.running-$NAME ]; do docker stats --no-stream --format "{{.CPUPerc}}" $NAME 2>/dev/null; sleep 2; done ) > ${OUT%.json}.cpu.txt &
touch $B/results/.running-$NAME
python3 $B/e2e_tool_stream.py --base-url http://127.0.0.1:$PORT --model $MODEL --arm $ARM --engine $ENGINE --concurrency $CONC --requests $ROUNDS --output $OUT
rm -f $B/results/.running-$NAME; sleep 3
echo "cpu samples: $(wc -l < ${OUT%.json}.cpu.txt)  mean: $(sed 's/%//' ${OUT%.json}.cpu.txt | awk '{s+=$1;n++} END{if(n) printf "%.0f%%", s/n}')"
docker rm -f $NAME >/dev/null; echo "done -> $OUT"
