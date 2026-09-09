#!/bin/bash
# gpu-sampler.sh -- nvidia-smi once a second, every column NAMED in the header row.
#
# Start in its own pane before the sweep. Ctrl-C after.
#     ./q1-vllm-serving/week3/gpu-sampler.sh
#
# Output: results/week3/gpu.csv
#
# Columns, and the trap Week 2 fell into:
#   memory.used         MiB of VRAM allocated. vLLM grabs ~90% at boot and never
#                       gives it back, so this is a FLAT LINE all session. Not a load signal.
#   utilization.gpu     % of time a kernel was running.  <- this is what Week 2's ~35% was
#   utilization.memory  % of time the memory BUS was busy. NOT % of VRAM in use.
#   power.draw, clocks.sm   for the cold-start question: does the clock ramp up over the session?
#
# Fields reference: run  nvidia-smi --help-query-gpu  on the pod.

mkdir -p results/week3

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
  --format=csv,nounits \
  -l 1 > results/week3/gpu.csv