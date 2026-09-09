# Week 3 Notes

**Goal:** find the KV cache memory ceiling for this GPU + model + config, and
tell prefill effects apart from decode effects.

Plan: `../week-wise-plan/week3-plan.md` · Daily guide: `../week-wise-plan/week3-day-by-day.md`

**The one-sentence result:** requests start queuing at concurrency 8 because the
per-step token budget binds first, and the KV cache does not run out until
concurrency ~35, where preemption starts at 97.5% occupancy.

---

## 1. Model config (verified Day 1)

From `Qwen/Qwen2.5-7B-Instruct-AWQ` `config.json`.

| Field | Value | Why it matters |
|---|---|---|
| `num_hidden_layers` | 28 | multiplier in the KV formula |
| `num_attention_heads` | 28 | **not** the one you want |
| `num_key_value_heads` | 4 | GQA. Using 28 here overestimates KV by 7x |
| `hidden_size` | 3584 | head_dim = 3584 / 28 = 128 |
| `tie_word_embeddings` | false | embed and lm_head are separate weights |

```
KV bytes per token = 2 (K and V) x layers x KV heads x head_dim x dtype bytes
                   = 2 x 28 x 4 x 128 x 2
                   = 57,344 B = 56 KiB per token
```

Two traps. Use `num_key_value_heads` (4), not `num_attention_heads` (28). And
AWQ quantizes the weights, not the KV cache, so the cache is still fp16 and the
`2 bytes` is right.

Serve config, locked all week (`serve.sh`): `--max-model-len 8192`,
`--gpu-memory-utilization 0.90`, `--quantization awq`.

---

## 2. Paper notes: Orca (OSDI '22)

Yu et al., *ORCA: A Distributed Serving System for Transformer-Based Generative
Models*. Five points, in my own words, with my own numbers where they apply.

### 2.1 Iteration-level scheduling

Older servers pick a batch and don't get control back until every request in it
finishes. Two costs: a request that finishes early keeps generating throwaway
tokens until its batchmates are done, and a request that arrives mid-batch waits
for the whole batch to drain before anything starts.

Orca schedules one *iteration* at a time. An iteration is one forward pass
through all layers, producing one token per request. Control returns to the
scheduler every few tens of milliseconds, so finished requests can leave and new
ones can join at every boundary. The batch becomes a rolling window.

**My Week 2 numbers show the throughput half of this** (`week2_conc1.json` vs
`week2_conc20.json`, same config, only concurrency differs):

```
                     conc 1        conc 20      change
mean ITL           8.826 ms      12.033 ms      +36%
output tok/s      108.55        1621.29         ~15x
duration          235.83 s        15.79 s       14.9x faster
```

20x the concurrency bought ~15x the throughput for 36% slower tokens.

**Limit of this evidence:** the sweep ran `--request-rate inf`, which fires all
100 prompts at once. Nothing arrived mid-flight, so this says nothing about the
queueing-delay half. Untested here.

**Oddity carried forward:** `Peak concurrent requests` in the Week 2 log is
exactly 2x `max_concurrency` in all five runs. Resolved in Week 2 as a
client-side bucketing artifact in `calculate_metrics()`, not the server holding
double.

### 2.2 Selective batching

Once the batch is rebuilt every iteration, requests are no longer in lockstep,
so tensors are ragged and won't combine into a rectangle.

Only attention needs to know which token belongs to which request. Linear
layers, layer norm, residual adds and activations treat each row independently.
So: flatten everything into one *total tokens x hidden size* tensor with no batch
dimension, split per request for attention only, merge back afterwards. Nothing
is padded. The unit of batching becomes the token, not the request.

```
batched (one flat tensor):  QKV proj / layernorm / residual / activation / out proj
   [ a1  b1  c1  c2  d1  d2  d3 ]
              | split
NOT batched (per request):  attention
   [a1]+cache  [b1]+cache  [c1 c2]  [d1 d2 d3]
              | merge
   [ a1  b1  c1  c2  d1  d2  d3 ]
```

KV is cached separately per request, and held until the scheduler explicitly
frees it. That is what makes a ragged batch safe, and it is what couples this to
the memory problem below.

**Cost:** at fixed batch size Orca's engine is slightly slower, because it does
not batch attention. Defensible: attention has no model parameters, and reading
parameters out of memory is the actual bottleneck.

### 2.3 How the scheduler admits requests

FCFS defined per iteration: for any two requests, if one arrived earlier it must
have completed at least as many iterations as the later one. A later arrival can
still *finish* first if it needs fewer iterations.

Selection: drop anything already running, sort by arrival, walk the list, take at
most `max_bs`, and check the memory reservation for anything not yet started.
When one doesn't fit, the scan **stops** rather than skipping. Skipping would
break the FCFS guarantee. So head-of-line blocking is a deliberate price.

`max_bs` caps requests, not tokens.

### 2.4 Memory reservation — the weak point

At first schedule, Orca reserves slots for the request's entire token budget up
front. A slot is one token's K and V. If it doesn't fit, the scan stops.

Reserving the worst case is what makes progress provable — otherwise the
scheduler could admit a request, generate for a while, then find no room for the
next token, with nothing able to move and free anything.

**Three ways to spend the pool** (56 KiB/token, ~14 GiB = 262,144 slots):

| | scheme | per request | capacity |
|---|---|---|---|
| A | reserve max-model-len for everyone (older systems) | 8,192 slots = 448 MiB, 90.6% wasted on Week 2's workload | ~32 |
| B | reserve this request's budget (Orca) | 512 + 256 = 768 slots = 42 MiB | ~341 |
| C | 16-token blocks on demand (vLLM) | 48 blocks, divides evenly | ~341 |

**The ~10x gap is A versus C, not Orca versus vLLM.** This corrects
`week3-day-by-day.md:538`, which says "a slot must hold max-model-len = 8,192
tokens" — that mixes up a *server* setting with a *per-request* reservation.

**B and C tie at ~341, so Week 2 is the worst possible advertisement for
paging.** Three properties of that workload cause the tie, all visible in the
log: every request is the same length, 768 divides evenly by the 16-token block
so no block is part-used, and `ignore_eos=True` forces the declared output
length to be exactly right. Confirmed from the log: `Total generated tokens:
25600` = exactly 100 x 256, zero variance over 500 requests. Nothing in Week 2
could have shown a difference.

Measured prompt length is 541, not 512 — the chat template adds ~29 tokens. So
real usage is ~797 tokens per request.

**So where does reserving up front actually lose?** Not on reservation size:

1. **Output length isn't knowable in advance.** The budget is set before
   generation starts, so a real client sets it high to be safe and holds that
   worst case for the request's whole life. `ignore_eos=True` hides this
   completely. This is the deep reason and it survives the correction above.
2. **No sharing.** Two requests with the same system prompt fill separate slots
   with identical K and V.
3. **No taking memory back.** Slots are held to completion. vLLM can *react* to
   block exhaustion by evicting; Orca can only *prevent* it by refusing.
4. **Head-of-line blocking**, from the stop-don't-skip rule plus reservation.

### 2.5 What Orca does not solve

Orca re-decides batch membership every iteration but reserves memory once per
request. It fixed the time axis and left the memory axis to PagedAttention.

The paper names one open problem itself: it fuses scheduler and engine together,
giving up the clean serving-layer / engine-layer separation, and leaves the
interface design as future work.

### Glossary

| Term | Meaning |
|---|---|
| Iteration | One pass through all layers; one token per request in the batch |
| Prompt / generating phase | Processing the whole prompt at once / one token per iteration after |
| Queueing delay | Time before any computation starts on a request |
| Split / Merge | Separate requests for attention, rejoin afterwards |
| Slot | One token's K and V — 56 KiB here |
| Reservation | Slots promised at admission, held until the request finishes |
| `max_bs` | Max requests in one batch |
| Head-of-line blocking | One unadmittable request stalling everything behind it |

---

## 3. Predicted capacity (pre-registered Day 4, before the pod)

| # | Prediction | Predicted |
|---|---|---|
| P1 | KV pool size from startup log | ~250,000–270,000 tokens |
| P2 | Max concurrency for 8,192 tokens/request | ~31x |
| P3 | Anchor conc-20 output tok/s vs Week 2's 1621.29 | within ±5% |
| P4 | Cold-start conc-1 TTFT, run last | ~25 ms (Week 2: 98.41 ms) |
| P5 | First ceiling level where `waiting > 0` | 16 or below |
| P6 | KV usage at that first `waiting > 0` | low (<30%) → token budget binds first |
| P7 | Any preemptions | yes, at 48 or 64 |

P6 was the one that mattered: seeing `waiting > 0` and calling it "the memory
ceiling" would be wrong if the pool were still mostly empty.

---

## 4. Measured capacity

Engine startup, 09-09 12:07:22–12:08:48, transcribed to
`results/week3/capacity-config.txt`:

```
non-default args: {'model': 'Qwen/Qwen2.5-7B-Instruct-AWQ', 'max_model_len': 8192,
                   'quantization': 'awq', 'gpu_memory_utilization': 0.9}
Available KV cache memory: 13.42 GiB
GPU KV cache size: 251,344 tokens, Maximum concurrency for 8,192 tokens: 30.68x
```

Cross-checked against `vllm:cache_config_info`: `block_size=16`,
`num_gpu_blocks=15709`, `enable_prefix_caching=True`,
`prefix_caching_hash_algo=sha256`. 15,709 x 16 = **251,344 tokens**. Two
independent sources, exact agreement.

| Quantity | Value |
|---|---|
| Available KV cache memory | 13.42 GiB |
| KV pool | 251,344 tokens / 15,709 blocks |
| Block size | 16 tokens |
| Engine's own max concurrency @8192 | 30.68x |
| Prefix caching | **enabled by default** |
| Chunked prefill | enabled |
| `max_num_seqs` / `max_num_batched_tokens` | **not printed** — see open questions |

**P1 and P2 both hit.** The Part 2 arithmetic was sound: 56 KiB x 251,344
≈ 13.4 GiB against the reported 13.42 GiB.

Also recorded for `ENVIRONMENT.md`: `dtype=torch.float16`, attention backend
`FLASH_ATTN`, `AutoAWQMarlinLinearMethod`, `cudagraph_mode=FULL_AND_PIECEWISE`,
engine `seed=0`, `revision=main`.

---

## 5. Axes A and B — dropped in the Week 3 re-cut

Both are Month 3 "Perturb" work, not Month 1 baseline.

Partial substitute for B from data that does exist: a single 541-token prefill
took 108 ms (~5,000 tok/s); 20 concurrent 541-token prefills took 1,033 ms
(~10,500 tok/s aggregate). Directional only.

Axis D turned out to be prefill-bound by accident, which covers some of what B
was for.

---

## 6. Axis C — Week 2 anchor (512 in / 256 out)

| Metric | Week 2 conc-20 | Week 3 anchor conc-20 | Δ |
|---|---|---|---|
| Prefix cache hit rate | not measured | **3.0%** | — |
| Output tok/s | 1621.29 | **1066.60** | **−34.2%** |
| TTFT mean | 71.39 ms | **1033.30 ms** | **14.5x worse** |
| TTFT p50 / p99 | 72.64 / 128.50 ms | 1107.89 / 1811.71 ms | 15.3x / 14.1x |
| ITL mean | 12.033 ms | 14.694 ms | +22% |
| **ITL p50** | (mean 12.033) | **11.839 ms** | **−1.6%** |
| Peak KV usage | not measured | **6.0%** | — |

| Metric | Week 2 conc-1 | Week 3 cold-start conc-1 | Δ |
|---|---|---|---|
| Output tok/s | 108.55 | **108.39** | −0.15% |
| ITL mean | 8.826 ms | 8.811 ms | −0.17% |
| TPOT mean | 8.861 ms | 8.836 ms | −0.28% |
| TTFT mean | 98.41 ms | **108.35 ms** | +10.1% |
| Peak KV usage | not measured | **0.3%** | — |

**C1. conc-1 reproduces almost perfectly** — throughput within 0.15%, ITL within
0.17%, across two pods, a week apart, re-downloaded weights and a different run
position. The rig is sound.

**C2. conc-20 does not reproduce** — 34% less throughput, 14.5x worse TTFT — yet
ITL p50 matches Week 2's ITL mean to 1.6%. Decode is identical. The entire
deficit is prefill.

**C3. Week 2's conc-1 TTFT was not a cold-start artifact. P4 refuted.** The
cold-start run was placed last, on a GPU that had been at 100% for 17 minutes
with clocks at 2.4 GHz, and still produced 108 ms — 10% *higher* than Week 2, not
the ~25 ms predicted. 108 ms is simply what a 541-token prefill costs here.

**C4. Both short-prompt runs measured 3.0%, and that is exactly the chat
template.** Anchor: 54,100 queries / 1,600 hits over 100 prompts. Cold-start:
21,640 / 640 over 40. That is exactly 16 tokens — one block — per request. It's
16 rather than the ~29 the template occupies because a partial block is not
cacheable. The Week 2 security note predicted this.

**C5. So the Week 2 / Week 3 gap comes from Week 2's side.** `week2-sweep.sh`
runs 1/2/5/10/20 in one loop with `--seed 42` and fixed lengths, so all five runs
sent the **same 100 prompts**. Runs 2–5 re-sent prompts run 1 had already cached,
and conc-20 was the fifth. Week 2 has no hit-rate data to prove this directly,
but the table carries its own fingerprint: TTFT *falls* from 98.41 ms at
concurrency 1 to 24.36 ms at concurrency 2, which is impossible unless the
prefill work differed between runs.

**Consequence: four of five TTFT rows in the Week 2 table cannot be read as
prefill cost.** ITL and TPOT are unaffected — decode is per-token work no cache
can skip.

**C6. Neither short-prompt run came near the ceiling.** Peak KV 6.0% at conc 20,
0.3% at conc 1. At 797 tokens per request, 20 concurrent requests reserve 1,000
of 15,709 blocks = 6.4%, matching the measured 6.0%. Yet `num_requests_waiting`
still reached 11. **Requests queued with the KV cache 94% empty.** That is P6's
scenario, measured.

---

## 7. Axis D — ceiling hunt (6144 in / 1024 out)

Five runs, `--ignore-eos`, seed 42, prompts = 2 x concurrency (min 32). All
completed with `failed: 0`.

| conc | blocks needed | % pool | **hit %** | out tok/s | TTFT mean | TPOT mean | ITL p95 | peak run | peak wait | peak KV % | **preempts** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 3,600 | 22.9% | **7.5** | 361.67 | 3332.7 ms | 18.86 ms | 14.49 ms | 8 | 6 | 22.8 | 0 |
| 16 | 7,200 | 45.8% | **99.8** | 811.58 | 255.7 ms | 19.43 ms | 20.44 ms | 16 | **0** | 45.6 | 0 |
| 32 | 14,400 | 91.7% | **50.0** | 626.66 | 9091.1 ms | 41.83 ms | 34.06 ms | 32 | 26 | 91.3 | 0 |
| 48 | 21,600 | 137.5% | **0.3** | 397.06 | 34906.7 ms | 78.69 ms | 496.29 ms | 40 | 46 | **100.0** | **10** |
| 64 | 28,800 | 183.3% | **0.3** | 391.11 | 66188.1 ms | 82.93 ms | 497.93 ms | 40 | 61 | **100.0** | **15** |

### D1. The prefix cache explains the whole series

| run | queries | hits | rate | why |
|---|---|---|---|---|
| c8 | 197,536 | 14,864 | 7.5% | first exposure to these 32 prompts |
| c16 | **197,536** | **197,120** | **99.8%** | same 32 prompts, fully resident |
| c32 | 395,072 | 197,408 | **50.0%** | 64 prompts, first 32 still cached |
| c48 | 592,608 | 1,536 | 0.3% | 96 prompts, cache overrun |
| c64 | 790,144 | 2,048 | 0.3% | 128 prompts, cache overrun |

Query counts equal the JSONs' `total_input_tokens` exactly, so the hit rate
directly measures the fraction of prefill skipped. c8 and c16 issued the
identical query count because they sent identical prompts — c8 paid, c16 got
99.8% free. c32's 50.0% is arithmetic: its 197,408 hit tokens are within 0.15% of
c16's 197,120, so the cache returned exactly the same 32 prompts and nothing
more. Retention collapsed at 96 prompts because 592,608 tokens does not fit the
251,344-token pool alongside live requests.

### D2. The throughput curve is unusable, and that is the finding

Reported: 361.67 → 811.58 → 626.66 → 397.06 → 391.11 tok/s. That is not a
scaling curve — it tracks the hit rate. The 812 "peak" is a run that skipped
99.8% of its prefill.

Only three points are cache-comparable:

| conc | hit rate | out tok/s | mean TTFT |
|---|---|---|---|
| 8 | 7.5% | 361.67 | 3.3 s |
| 48 | 0.3% | 397.06 | 34.9 s |
| 64 | 0.3% | 391.11 | 66.2 s |

**Throughput is flat from concurrency 8 to 64 (under 10%) while mean TTFT rises
20x.** At 6,144-token prompts this server saturates at or below concurrency 8.
Every slot past that buys queueing delay and nothing else. Axis D was
unintentionally a prefill-bound experiment, and prefill does not batch its way
out of a bandwidth limit the way decode does.

**Retraction.** An earlier draft read "peak useful throughput is at concurrency
16" and built a "2.1x theory-versus-measurement gap" on it. Both were artifacts
of the 99.8% hit rate, from reading the client JSONs without cache telemetry.
Withdrawn.

### D3. The KV arithmetic is accurate to within one sequence

15,709 blocks ÷ 450 blocks per full request = **34.9 concurrent requests**.

| conc | peak running | mean | **modal** |
|---|---|---|---|
| 48 | 40 | 28.5 | **35** (102 of 512 samples) |
| 64 | 40 | 30.2 | **35** (175 of 678 samples) |

Both over-subscribed runs settle at 34–35. Peak 40 exceeds 34.9 because a request
early in decode holds 6,144 + n tokens rather than the full 7,168, so more fit
briefly.

The formula predicts capacity. It does **not** predict where throughput peaks —
D2 shows saturation around concurrency 8, well below the 35 the memory holds.

### D4. Preemption threshold: 97.5% KV occupancy

25 events, all in c48 (10) and c64 (15), zero elsewhere. Occupancy at every
event: min 97.4%, max 97.7%, mean 97.53%. Concurrency 32 peaked at 91.3% and
never preempted. **No preemption at 91.3%, preemption at 97.4%.**

The mechanism, step by step from the counters:

```
preempt 1  KV 97.6%  running 39  waiting 9
preempt 2  KV 97.6%  running 38  waiting 10
preempt 3  KV 97.5%  running 37  waiting 11
preempt 4  KV 97.4%  running 36  waiting 12
preempt 5  KV 97.4%  running 35  waiting 13
```

Running walks down one sequence per step while waiting climbs, ~7 s apart, KV
pinned at 97.5%, then repeats ~70 s later. This is Orca's memory-reservation weak
point in live telemetry.

Time near saturation: c32 touched 90% for ~5 s and never hit 95%; c48 spent ~95 s
above 95%. That is the boundary. It also explains the 496 ms ITL p95 — that is a
preempted sequence waiting for blocks.

### D5. Queuing is not a ceiling signal — P6, measured

| run | KV % at first `waiting > 0` | peak KV % | preempts |
|---|---|---|---|
| warm-up c16 | 1.9% | 4.9% | 0 |
| anchor c20 | **1.0%** | 6.0% | 0 |
| ceil c8 | 1.8% | 22.8% | 0 |
| ceil c16 | **never queued** | 45.6% | 0 |
| ceil c32 | 75.9% | 91.3% | 0 |
| ceil c48 | 1.6% | 100.0% | 10 |
| ceil c64 | 2.5% | 100.0% | 15 |

In five of six runs that queued, first queuing happened with KV **under 3% full**
— arrival burst, not capacity. `--request-rate inf` fires all requests at once
and the scheduler admits them over several steps.

**So `waiting > 0` is nearly worthless as a ceiling indicator.** Reporting "the
ceiling is at concurrency 8, where requests began queuing" would have been wrong
by a factor of four. The signals that work are peak KV occupancy and
`num_preemptions_total`, neither of which was pre-registered.

**c16 never queued at all** — zero waiting across 116 samples. With 99.8% of its
prefill cached there was almost no admission work to queue for. The cache did not
just speed the run up; it removed the queue.

`num_requests_waiting_by_reason` was `capacity` for every waiting request and
`deferred` for none, across all eight runs.

---

## 8. THE MEMORY CEILING FINDING

On this GPU (RTX PRO 4000 Blackwell, 24 GB) serving Qwen2.5-7B-Instruct-AWQ at
`--max-model-len 8192 --gpu-memory-utilization 0.90`, the engine allocates
**251,344 tokens (15,709 blocks x 16)** from the **13.42 GiB** left after
weights. A 7,197-token request (6144 + 1024 + ~29) reserves **450 blocks**, so
the pool holds **34.9**.

1. **The server settles at 34–35 concurrent sequences.** Asked for 48 it ran 35;
   asked for 64 it ran 35 again. Block arithmetic predicts capacity to within one
   sequence.

2. **The ceiling is enforced by preemption at 97.5% KV occupancy.** 25 events,
   all at 97.4–97.7%, all in the two over-subscribed runs.

3. **Requests queue long before memory is the constraint.** Five of six queuing
   runs did so with KV under 3%. At concurrency 20 with 512-token prompts, 11
   requests waited while the pool was 94% empty. Queuing and the memory ceiling
   are unrelated at this scale; conflating them would have put the ceiling at
   concurrency 8 instead of 35.

4. **Throughput saturates far below the memory ceiling.** Among the three
   cache-comparable runs, throughput is 362 / 397 / 391 tok/s at concurrency
   8 / 48 / 64 — flat within 10% — while mean TTFT rises from 3.3 s to 66.2 s.
   **The useful operating point and the memory capacity differ by more than 4x,
   and only the second is what the KV formula describes.**

5. **c16 and c32 are excluded** from the throughput conclusion. At 99.8% and
   50.0% hit rates they measure cache reuse, not scaling, and must not be quoted
   as scaling results.

---

## 9. Why VRAM is the wrong ceiling signal

`gpu.csv`, 2,472 samples at 1 Hz; `metrics.log`, 4,892 samples at 0.5 Hz.

| Window | samples | memory.used MiB | util.gpu | util.memory | mean KV % | peak KV % | temp max |
|---|---|---|---|---|---|---|---|
| idle, pre-sweep | 193 | 20,874 | 0% | 0% | 0.0 | 0.0 | 28 |
| warm-up c16 | 34 | 20,874–21,103 | 24% | 19% | 0.8 | 4.9 | 43 |
| anchor c20 | 45 | 20,874 | 53% | 39% | 2.3 | 6.0 | 56 |
| ceiling c8 | 112 | 20,874 | 80% | 63% | 15.1 | 22.8 | 67 |
| ceiling c16 | 62 | 20,874 | 65% | 60% | 27.4 | 45.6 | 64 |
| ceiling c32 | 128 | 20,874 | 81% | 67% | 56.7 | 91.3 | 68 |
| ceiling c48 | 272 | 20,874 | 91% | 70% | 74.6 | **100.0** | 68 |
| ceiling c64 | 361 | 20,874–20,888 | 93% | 71% | 78.8 | **100.0** | 69 |
| cold-start c1 | 110 | 20,874 | 85% | **75%** | 0.2 | 0.3 | 68 |

**V1. `memory.used` is a flat line while KV occupancy covers its whole range.**
VRAM ranged 20,874–21,103 MiB of 24,467 — a spread of 1.1% — while KV occupancy
went 0.0% to 100.0% and the server went from idle to preempting 25 times. vLLM
claims 90% of VRAM at boot and never releases it, so `nvidia-smi` reports the
reservation, not the occupancy. **It cannot detect the ceiling.**

This resolves the apparent contradiction in Week 1 — "~87% VRAM reserved"
alongside "KV cache usage under 1%". Both true, measuring different things: what
the process holds from the driver, versus what the engine uses of what it holds.

**V2. Which column was Week 2's ~35%.** At anchor conc-20, `utilization.gpu`
averaged 53% and `utilization.memory` 39%. The latter is the plausible source,
but this is an attribution, not a confirmation — Week 2's reading was never
labelled. Both are *time* percentages: the share of the interval in which a
kernel was running or the memory bus was active. Neither means "35% of the GPU"
or "35% of VRAM", which is how such a number is usually read.

**V3. `utilization.memory` is highest at concurrency 1 — Week 1's decode claim
confirmed.** The cold-start run shows the highest memory-bus utilisation of the
session (75%) while producing the lowest throughput (108 tok/s, 3.6x less than
concurrency 64) and using 0.3% of the pool. One sequence decoding alone reads the
full weight matrix to emit a single token: it saturates the bus while compute
idles. Batching amortises that read, which is why c64 gets 3.6x the throughput at
*lower* bus utilisation. Week 1 inferred this from a throughput ratio; here all
three instruments agree.

**V4. SM clocks ramp 13.8x** — 172–180 MHz idle, 2,467–2,490 MHz under load.
Large enough to make the cold-start hypothesis plausible, and it was still
refuted: the cold-start run executed on a 68 °C GPU after 17 minutes of load and
reproduced Week 2's number to 0.15%. Its own SM max was 2,242 MHz, the lowest of
any load window, because one decoding sequence never demands full clocks.

---

## 10. Weekend teardown: what triggers queuing

**Not started.** Deferred to the weekend, 1-hour hard cap.

Telemetry already answered more than the teardown was scoped to: queuing driven
by arrival burst at KV under 3% (D5), ceiling enforced by preemption at 97.5%
(D4). What source reading would still add:

- **T3** — the preemption path. Where a failing `allocate_slots` leads to
  eviction, and why the threshold sits at ~97.5% rather than 100%.
- **T1/T2** — `max_num_seqs` and `max_num_batched_tokens` are not printed by this
  build. Readable from 0.28.0's defaults in the local clone. Peak running hit
  exactly 40 in both over-subscribed runs, which is suspicious enough to check.
- **The admission burst** — why 20 simultaneous arrivals produce 11 waiting when
  the pool is 94% empty. Almost certainly the per-step token budget.

---

## 11. Predictions: scorecard

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| P1 | ~250–270k tokens | **251,344** (two sources) | hit |
| P2 | ~31x | **30.68x** | hit |
| P3 | within ±5% of 1621.29 | **1066.60, −34.2%** | **miss** |
| P4 | ~25 ms | **108.35 ms** (Week 2: 98.41) | **miss** |
| P5 | `waiting > 0` at 16 or below | **conc 8**, warm-up at 16 | hit |
| P6 | KV <30% at first queuing | **1.0–2.5%** in 5 of 6 runs | hit |
| P7 | preemptions at 48 or 64 | **10 at c48, 15 at c64** | hit |

**Five hits, two misses, nothing unresolved.**

**P6 was the prediction worth writing, and it paid.** It existed to stop this
week reporting "requests started queuing, therefore the memory ceiling is here."
Queuing began at 1.0% occupancy; the ceiling was at 97.5%. A factor of 97, and
the note that closed it existed before any data did.

**P5 was a hit and a badly chosen question.** `waiting > 0` fires on arrival
burst in almost every run, so predicting where it first appears measured the load
generator, not the server. Choose the metric before predicting its value.

**Why P3 was wrong.** It assumed repeating a configuration reproduces its number.
It does not, when the engine carries state between runs. **A benchmark against a
stateful server is only reproducible if the state is cleared or recorded, and
this repo did neither.**

**Why P4 was wrong.** It over-fitted one anomalous point to a convenient
explanation. Run last on a hot GPU, the same config reproduced throughput to
0.15%. The ordering was coincidence, and the real cause ran the other way — not
warm-up inflating conc-1, but cache reuse deflating every other run. **P4 being
wrong is what produced the week's main finding.**

---

## 12. Open questions for Week 4

1. **Both sweep scripts need per-run seeds.** Highest priority; blocks quoting
   any further TTFT number. Both pass `--seed 42` with fixed lengths to every
   run, guaranteeing identical prompts and cache reuse from run 2 on. Either vary
   the seed per run, or keep it and report the hit rate beside every TTFT. The
   second is better science and now costs nothing.
2. **Is the Week 2 table salvageable?** ITL/TPOT reproduced to 1.6% and are
   sound. TTFT is contaminated for four of five rows with no Week 2 hit-rate data
   to quantify it. Re-measuring is ~10 minutes of pod time. The Month 1 report
   must not present that TTFT column without re-measurement or an explicit
   warning.
3. **`max_num_seqs` and `max_num_batched_tokens` still unknown.** Needed to
   explain the peak `running` of exactly 40, and to attribute D5's queuing to the
   token budget rather than by elimination. Free from the local clone.
4. **Why does preemption trigger at 97.5% rather than 100%?** Tight enough
   (97.4–97.7% across 25 events) to be a real boundary. Candidates: reserved
   headroom for the next prefill chunk, block fragmentation, a watermark
   constant. One grep in the scheduler.
5. **Why is c8's hit-rate floor 7.5% rather than 3.0%?** 14,864 hit tokens over
   32 requests is 929 blocks, ~29 each, not the single template block. Either the
   `random` dataset shares a longer prefix at 6144 tokens, or chunked prefill
   re-queries the cache within the same request. The second would mean the metric
   partly measures intra-request behaviour at long prompts, which changes how
   99.8% should be read. One look at `vllm/benchmarks/datasets.py`.
6. **The model revision is not pinned.** The engine reports `revision=main` and
   weights were re-downloaded this session (15.58 s, `OVERLAY` filesystem).
   `ENVIRONMENT.md` records `b25037543e...` as though enforced. Either add
   `--revision` or downgrade that line to "observed once, unverified".
7. **Sampling was not greedy.** `vllm bench serve` 0.28.0 no longer defaults to
   `temperature=0`. Throughput is unaffected because `--ignore-eos` fixes output
   length, but generated text is not reproducible. Add `--temperature 0`.
8. **Axis D was accidentally prefill-bound.** A decode-bound ceiling hunt — short
   prompts, long outputs, so the pool fills from generation — is a better Month 3
   design. The 34–35 capacity result is unaffected: it rests on block arithmetic
   and `running`, not throughput.

---

## 13. Operational findings

**O1. The container start command is PID 1.** The RunPod template auto-starts
vLLM, so a server holds port 8000 before any shell exists. Running `serve.sh` on
top fails with `Address already in use`, and `pkill -f "vllm serve"` **terminated
the entire pod**, because killing PID 1 stops the container. Fix: serving flags
moved into the start command itself, so the auto-started server *is* the intended
config. `serve.sh` stays as the version-controlled record.

**O2. The default start command served the wrong context length.** Before O1's
fix, `/v1/models` reported `max_model_len: 32768` against the specified 8192.
**Second time this template has served a configuration other than the one
specified** — Week 1 was the same failure with the model itself.

**O3. The network volume was not mounted.** The repo was at `/vllm-workspace/`,
not `/workspace/`; the log reports `OVERLAY` and 15.58 s downloading weights that
should have been resident. Everything written lived on container disk. **A repo
clone was lost to this earlier the same day, and `metrics.log` was nearly lost
the same way.** `df -h /workspace` is now the first command of every session:
`overlay` means the volume is absent.

**O4. The image lacks `git` and `file`.** Both are container filesystem, so
`apt-get install -y git file` runs every session. Belongs in the checklist.

**O5. An SSH disconnect killed the sweep's shell but not the samplers.** Wi-Fi
dropped during c64. On reconnect both samplers were still running on orphaned
ptys and the sweep had completed, all seven JSONs present with `failed: 0`. Their
surviving is why `metrics.log` covers the full 2,598 s. Long jobs belong under
`tmux`.

**O6. Startup output goes to container stdout, not a file.** No local
`server-startup.log` this session; capacity numbers transcribed by hand from the
RunPod Logs tab. Full engine log exported separately as `logs.txt`.

**O7. The samplers were the highest-value thing built this week.** Four scripts,
three of them trivial. `metrics.log` turned three unresolved predictions into
measurements, proved the prefix-cache finding, located the preemption threshold
to 0.3 points, and forced the retraction of an incorrect throughput conclusion.
The Week 2 method gap — "`/metrics` was never sampled during the runs" — was the
right thing to have identified, and closing it was worth more than any additional
run.

**Session cost:** pod up ~12:07, terminated ~13:08. Sweep proper 12:29–12:48.
`metrics.log`: 4,892 samples over 2,598 s.

---

## 14. For the Month 1 report's method section

The serving configuration is asserted by a version-controlled script and verified
at runtime from `/v1/models` and the engine's `non-default args` line every
session, because the platform was twice observed to serve a configuration other
than the one specified. Every latency figure is reported alongside the
prefix-cache hit rate of the run that produced it, because a fixed seed across
runs was found to make TTFT a measure of cache reuse rather than of prefill.