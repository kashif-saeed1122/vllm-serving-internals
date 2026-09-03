#!/usr/bin/env bash
# week2_sweep.sh — concurrency sweep, prompt/output length held fixed
set -euo pipefail
mkdir -p results/week2

for c in 1 2 5 10 20; do
  echo "=== concurrency $c ==="
  vllm bench serve \
    --backend openai-chat \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --ignore-eos \
    --num-prompts 100 \
    --max-concurrency "$c" \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --seed 42 \
    --save-result \
    --result-dir results/week2 \
    --result-filename "week2_conc${c}.json"
done
