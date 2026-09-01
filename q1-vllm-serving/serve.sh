#!/bin/bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000