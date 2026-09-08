#!/bin/bash
# metrics-sampler.sh -- record vLLM's /metrics twice a second, raw, into one log.
#
# Start in its own pane BEFORE the sweep. Stop with Ctrl-C AFTER it.
#     ./q1-vllm-serving/week3/metrics-sampler.sh
#
# Output: results/week3/metrics.log. It looks like this, repeated every 0.5 s:
#
#     1757340000                                          <- unix time, one per sample
#     vllm:num_requests_running{...} 16.0
#     vllm:num_requests_waiting{...} 0.0
#     vllm:kv_cache_usage_perc{...} 0.31                  <- fraction 0..1, not percent.
#     vllm:num_preemptions_total{...} 0.0                    (older vLLM: gpu_cache_usage_perc)
#     vllm:prefix_cache_queries_total{...} 12345.0
#     vllm:prefix_cache_hits_total{...} 678.0
#
# Nothing is parsed here on purpose. summarize.py reads this file afterwards.
#
# Why one curl per sample: all seven numbers come from the same instant, so
# "waiting" and "kv_cache_usage" on the same sample are comparable. That
# comparison IS the memory-ceiling finding.
#
# If the server is down, curl prints nothing, the timestamp is still written,
# and the loop carries on. A gap in the data, not a dead sampler.

mkdir -p results/week3

while true
do
  date +%s >> results/week3/metrics.log
  curl -s --max-time 2 localhost:8000/metrics \
    | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|num_preemptions|prefix_cache_queries|prefix_cache_hits)" \
    >> results/week3/metrics.log
  sleep 0.5
done