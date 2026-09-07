# vLLM Serving Internals

Q1 of a self-directed AI/ML infrastructure learning plan. Goal: understand,
measure, and explain how a real LLM serving stack behaves under load — and
where/why it breaks — using vLLM and Qwen2.5-7B-Instruct-AWQ on a single GPU.

## Why this exists

Most LLM projects stop at "I called an API." This one goes the other
direction: benchmark a serving stack rigorously, read the source that
implements it, then deliberately break it and explain the failure modes
with data, not guesses.

Every number in this repo was measured on the exact stack recorded in
[ENVIRONMENT.md](./q1-vllm-serving/ENVIRONMENT.md), and the raw
`vllm bench serve` JSON output is committed alongside the notes so any claim
can be checked against its source data.

## Structure

- **Month 1 — Benchmarking baseline**: TTFT, ITL, throughput, p50/p95/p99, and
  the KV-cache memory ceiling under varying load, measured with
  `vllm bench serve`.
- **Month 2 — Source trace**: annotated walkthrough of one request's full
  lifecycle through vLLM internals (API server → scheduler → block manager →
  KV cache → batching loop → output).
- **Month 3 — Failure modes**: systematic perturbation (batch size, context
  length, quantization, cache config) with explained degradation points.

## Status

🚧 **Week 3 of 12** (Month 1 — Baseline)

| Week | State |
|---|---|
| 1 — Environment verification | ✅ Complete |
| 2 — Real benchmarking tooling | ✅ Complete |
| 3 — Expand the sweep, find the memory ceiling | 🚧 In progress |
| 4 — Month 1 checkpoint | ⬜ Not started |

**Month 1 requires five metrics. Four are captured:**

| Metric | State |
|---|---|
| TTFT (mean, p50/p95/p99) | ✅ Week 2 |
| ITL / TPOT | ✅ Week 2 |
| Throughput (req/s, tok/s) | ✅ Week 2 |
| p50 / p95 / p99 | ⚠️ Week 2, but from 100 samples per config — p99 is effectively the maximum. Week 3 adds repeats. |
| **Memory ceiling** | ⬜ **Outstanding — Week 3's primary target** |

## Results so far

### Week 2 — concurrency sweep

Config held fixed across all runs: 512 input / 256 output tokens,
`--ignore-eos`, 100 prompts, `--request-rate inf`, `--seed 42`. Only
`--max-concurrency` varies.

| Concurrency | TTFT mean | TTFT p95 | TTFT p99 | ITL mean | Req/s | Output tok/s |
|---|---|---|---|---|---|---|
| 1  | 98.41 ms | 103.13 ms | 103.86 ms | 8.826 ms  | 0.424 | 108.55 |
| 2  | 24.36 ms | 26.51 ms  | 27.47 ms  | 8.846 ms  | 0.874 | 223.65 |
| 5  | 32.10 ms | 41.93 ms  | 44.59 ms  | 9.092 ms  | 2.117 | 542.04 |
| 10 | 44.21 ms | 60.32 ms  | 66.18 ms  | 10.017 ms | 3.826 | 979.55 |
| 20 | 71.39 ms | 108.45 ms | 128.50 ms | 12.033 ms | 6.333 | 1621.29 |

**Headline finding:** across a 20× increase in concurrency, mean inter-token
latency rose only **36%** (8.83 → 12.03 ms) while output throughput rose
**~15×** (108 → 1621 tok/s). That is continuous batching made numerically
visible — decode is memory-bandwidth-bound, so batching more sequences into one
step barely lengthens the step, and tokens-per-step scales far faster than
time-per-step.

**But throughput scaling bends past concurrency ~5.** Throughput gain per unit
of concurrency gain: 1→2 is 2.06×, 2→5 is 2.42×, 5→10 drops to 1.81×, 10→20 to
1.66×. Aggregate efficiency keeps improving, but **TTFT is the metric that
actually degrades** — a request at concurrency 20 waits ~3× longer for its first
token than one at concurrency 2.

Full analysis, PagedAttention paper notes, and the source-level resolution of a
benchmark-tool measurement artifact: **[week2-notes.md](./q1-vllm-serving/week2-notes.md)**

### Known open questions

Tracked honestly rather than hidden — see the notes for detail.

- **Concurrency 1 has the highest TTFT in the sweep** (98 ms), breaking an
  otherwise clean monotonic trend, with a tight distribution (std ≈ 13 ms) — so
  the whole run was slow, not a few outliers. Leading hypothesis: one-time
  warm-up cost paid by whichever run went first. Tested in Week 3.
- **The KV-cache memory ceiling has not been measured.** Cache usage stayed
  under 1% through Weeks 1–2, so no cache-pressure behaviour was ever
  triggered. Week 3's primary target.
- **`max_concurrent_requests` reports exactly 2× the configured concurrency** at
  every level. Resolved as a benchmark-tool artifact — integer-second bucketing
  double-counts wave handoffs when every request is the same length — traced to
  the responsible code and cross-checked against measured run duration.

## Repository layout

```
├─ Readme.md                        this file
├─ q1-vllm-serving/                 code and notes
│  ├─ ENVIRONMENT.md                locked stack: GPU, driver, CUDA, vLLM, model revision
│  ├─ SECURITY_NOTES.md             standing observation-only side-channel log
│  ├─ serve.sh                      launches vLLM with the pinned config
│  ├─ week2-sweep.sh                Week 2 benchmark script
│  ├─ week1-notes.md                environment verification, wrong-model incident
│  └─ week2-notes.md                PagedAttention notes, sweep results, teardown
├─ results/                         raw `vllm bench serve` JSON, one file per configuration
│  └─ week2/
├─ week-wise-plan/                  execution plans, written before each week
└─ week2-sweep.log                  raw console output from the Week 2 session
```

**Conventions.** Notes are per-week (`week1-notes.md`, `week2-notes.md`, …) so
each week can be read in isolation and the progression is visible.
`ENVIRONMENT.md` and `SECURITY_NOTES.md` are standing documents maintained
across the whole quarter. Benchmark scripts are run **from the repository
root**, so results land in `results/weekN/`.

## Reproducing the numbers

```bash
# 1. On a GPU host, launch the server with the pinned config
./q1-vllm-serving/serve.sh

# 2. Verify the served model — do not trust the logs
curl -s localhost:8000/v1/models

# 3. From the repository root, run the sweep
./q1-vllm-serving/week2-sweep.sh
```

Step 2 is not optional. Week 1's first session silently served a different model
than the configuration claimed, because the container start command — not the
environment variable — controls which model loads. The full incident is written
up in [week1-notes.md](./q1-vllm-serving/week1-notes.md).

## Environment

See [ENVIRONMENT.md](./q1-vllm-serving/ENVIRONMENT.md) for exact hardware,
driver, CUDA, vLLM version, and model revision hash used to produce all results
here. The stack is deliberately locked for the whole quarter so results stay
comparable across weeks; configuration variables are perturbed on purpose in
Month 3, not incidentally before then.

## Security notes

[SECURITY_NOTES.md](./q1-vllm-serving/SECURITY_NOTES.md) is an
**observation-only** log of security-relevant behaviour noticed while doing the
performance work — currently prefix-cache state being observable through an
unauthenticated `/metrics` endpoint, and measurably shaped by prompt overlap
across requests. Nothing is investigated or exploited this quarter; the log
exists so the observations are not lost.

## Planning documents

Each week is planned before it is executed, so no week loses context relative to
what came before:

- [Week 3 plan](./week-wise-plan/week3-plan.md) — design and rationale: capacity
  arithmetic, experiment axes, what distinguishes a scheduler limit from a
  memory limit
- [Week 3 day-by-day guide](./week-wise-plan/week3-day-by-day.md) — execution:
  daily steps, sources, and the tooling built from scratch

## Reports

Monthly writeups, populated as each checkpoint is completed.

- Month 1 — baseline numbers report *(due end of Week 4)*
- Month 2 — annotated request lifecycle *(due end of Week 8)*
- Month 3 — failure modes and quarter-final report *(due end of Week 12)*
