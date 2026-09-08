

mkdir -p results/week3


# ---- 0. warm-up. No --save-result: this run is thrown away on purpose. ----
# Pays the one-time costs (CUDA graph capture, clock ramp) before any measured run.
echo "RUN warmup_not_saved"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --ignore-eos \
  --num-prompts 32 \
  --max-concurrency 16 \
  --request-rate inf \
  --seed 7
sleep 10


# ---- 1. anchor: identical to Week 2's concurrency-20 run ----
echo "RUN anchor_in512_out256_c20"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --ignore-eos \
  --num-prompts 100 \
  --max-concurrency 20 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename anchor_in512_out256_c20.json
sleep 5


# ---- 2. ceiling, concurrency 8 ----
echo "RUN ceiling_in6144_out1024_c8"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 6144 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 32 \
  --max-concurrency 8 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename ceiling_in6144_out1024_c8.json
sleep 5


# ---- 3. ceiling, concurrency 16 ----
echo "RUN ceiling_in6144_out1024_c16"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 6144 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 32 \
  --max-concurrency 16 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename ceiling_in6144_out1024_c16.json
sleep 5


# ---- 4. ceiling, concurrency 32 ----
echo "RUN ceiling_in6144_out1024_c32"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 6144 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 64 \
  --max-concurrency 32 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename ceiling_in6144_out1024_c32.json
sleep 5


# ---- 5. ceiling, concurrency 48 ----
echo "RUN ceiling_in6144_out1024_c48"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 6144 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 96 \
  --max-concurrency 48 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename ceiling_in6144_out1024_c48.json
sleep 5


# ---- 6. ceiling, concurrency 64 ----
echo "RUN ceiling_in6144_out1024_c64"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 6144 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 128 \
  --max-concurrency 64 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename ceiling_in6144_out1024_c64.json
sleep 5


# ---- 7. cold-start test: concurrency 1, run LAST while the GPU is hot ----
# Week 2's conc-1 run went first and had the highest TTFT of the sweep (98 ms).
# Same config, run last. Near 25 ms -> Week 2 was a cold-start artifact.
echo "RUN coldstart_in512_out256_c1"
date +%s
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --ignore-eos \
  --num-prompts 40 \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename coldstart_in512_out256_c1.json


echo "SWEEP COMPLETE"
date +%s