# vLLM Serving Internals

Q1 of a self-directed AI/ML infrastructure learning plan. Goal: understand,
measure, and explain how a real LLM serving stack behaves under load — and
where/why it breaks — using vLLM and Qwen2.5-7B-Instruct-AWQ on a single GPU.

## Why this exists
Most LLM projects stop at "I called an API." This one goes the other
direction: benchmark a serving stack rigorously, read the source that
implements it, then deliberately break it and explain the failure modes
with data, not guesses.

## Structure
- **Month 1 — Benchmarking baseline**: TTFT, ITL, throughput, p99 latency
  under varying load, measured with `vllm bench serve` / GuideLLM.
- **Month 2 — Source trace**: annotated walkthrough of one request's full
  lifecycle through vLLM internals (scheduler, paged attention, KV cache).
- **Month 3 — Failure modes**: systematic perturbation (batch size, context
  length, quantization, cache config) with explained degradation points.

## Environment
See [ENVIRONMENT.md](./ENVIRONMENT.md) for exact hardware, driver, CUDA,
vLLM version, and model revision hash used to produce all results here.

## Status
🚧 Week 1 — environment verified, benchmarking harness not yet built.

## Reports
(Populated as each month's writeup is completed.)