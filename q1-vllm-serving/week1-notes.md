### when i deployed the gpu it auto picked the Qwen3-0.6B model infact i specified the Model Env var in the templete though 
### how i resolved it - I put this command in the Container image Command "--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq" 


## Concurrency test: single request vs 5 concurrent

| Metric                      | 1 request (800 tok) | 5 concurrent (~950 words ea.) |
|------------------------------|----------------------|-------------------------------|
| Generation throughput        | 39.9 tok/s           | 228.6 tok/s                   |
| Per-request effective rate   | 39.9 tok/s           | ~45.7 tok/s                   |
| GPU KV cache usage            | 0.2%                 | 0.9%                          |
| Prefix cache hit rate         | 34.4%                | 58.6%                         |

**Key finding**: 5x concurrent requests produced ~5.7x the aggregate
throughput with *no drop* in per-request speed. This matches the expected
behavior of continuous batching: single-sequence decode is memory-bandwidth
bound, not compute bound, so the GPU has spare capacity that batching
multiple sequences' decode steps absorbs almost for free. To be confirmed
more rigorously in Month 1 with `vllm bench serve` across a range of
concurrency levels (not just 1 vs 5).

**Caveat**: engine log stats are 10-second rolling-window snapshots, not
instantaneous rates — a snapshot taken right after a burst completes can
report a *higher* throughput than one taken mid-burst, since the sampling
window doesn't align with actual request start/stop times. Treat these
numbers as directional, not precise, until proper benchmarking tooling
is in place.

**Also observed**: GPU KV cache usage stayed under 1% even at 5 concurrent
long-generation requests — cache-pressure failure modes (Month 3 target)
will require much higher concurrency than tested so far to reproduce.