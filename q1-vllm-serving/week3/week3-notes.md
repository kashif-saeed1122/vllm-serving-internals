# Week 3 Notes

Goal: separate prefill-bound from decode-bound effects by sweeping prompt
length and output length independently, and find the KV-cache memory ceiling
for this GPU + model + config.

Plan: ../week-wise-plan/week3-plan.md
Daily guide: ../week-wise-plan/week3-day-by-day.md



## Model config (verified Day 1)

https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/config.json

#### Useful info

| Field | Verified | Why it matters |
|---|---|---|
| `num_hidden_layers` | 28 | multiplier in the KV formula |
| `num_attention_heads` | 28 | NOT the one you want |
| `num_key_value_heads` | 4 | GQA. Using 28 here overestimates KV by 7× |
| `hidden_size` | 3584 | → `head_dim` = 3584 / 28 = 128 |
| `tie_word_embeddings` | false | embed + lm_head are separate → ~2.2 GiB of fp16 weights paid twice |

**The one number the whole week hangs on.** These four feed the KV formula used in ④ and in
Day 5's capacity prediction:

```
KV bytes per token = 2 (K and V) × num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes
                   = 2 × 28 × 4 × 128 × 2
                   = 57,344 B = 56 KiB / token
```

Serve config it is paired with ([serve.sh](serve.sh)): `--max-model-len 8192`,
`--gpu-memory-utilization 0.90`, `--quantization awq`. AWQ quantizes weights, not the KV cache,
so the `dtype_bytes = 2` above is correct.

## Paper Notes: Orca
<!-- Day 2 -->

Yu et al., *ORCA: A Distributed Serving System for Transformer-Based Generative Models*
(OSDI '22). Five things only, in my own words. Every number below is either from this
repo's own runs or arithmetic on this machine's config — sources named inline.

**Evidence this note draws on**

| Source | What it gives |
|---|---|
| [serve.sh](serve.sh) | `--max-model-len 8192`, `--gpu-memory-utilization 0.90`, `--quantization awq` |
| [week2-sweep.sh](week2-sweep.sh) | the sweep flags: 512 in / 256 out, `--ignore-eos`, `--request-rate inf` |
| `results/week2/week2_conc{1,2,5,10,20}.json` | the metrics quoted below |
| [week2-sweep.log](../week2-sweep.log) | per-run config echo + the serving-result blocks |
| model `config.json` (linked above) | 28 layers, 4 KV heads, head_dim 128 |

---

### 1. Iteration-level scheduling

**The problem.** Older serving systems pick a batch, hand it to the GPU, and don't get
control back until every request in that batch is finished. The batch is frozen for its
whole lifetime. Two things go wrong.

First, requests need different numbers of steps. If one request stops at token 10 and a
batchmate runs to token 500, the finished one stays in the tensor for 490 more steps,
generating tokens nobody wants, which get thrown away. That is GPU time spent on nothing.
It also can't return its answer early, because the system only collects results when the
whole batch is done — so its latency includes its batchmates' remaining work.

Second, a request that arrives while a batch is running waits for that entire batch to
drain before any computation starts on it. That wait is pure queueing delay.

**The fix.** Schedule one *iteration* at a time instead of one batch at a time. An
iteration is one forward pass through all the model's layers, producing exactly one new
token for each request in the batch. The loop becomes: pick who runs, run exactly one
iteration, collect one token per request, repeat.

Because control comes back to the scheduler every iteration — tens of milliseconds — a
finished request can leave and a newly arrived one can join at every boundary. The batch
stops being a locked group and becomes a rolling window.

Four pieces do this: an endpoint that takes requests in and sends responses out; a
request pool holding every live request and its tokens; the scheduler, which picks the
batch each iteration and files the returned tokens back into the pool; and the engine,
which is the abstraction over the actual GPU math.

A request finishes when the model emits an end-of-sequence token or hits its token cap —
not when an iteration ends. Every request in a batch finishes its iteration at the same
moment, because that is what an iteration is.

**What Week 2's own numbers show.**

Comparing `results/week2/week2_conc1.json` against `results/week2/week2_conc20.json` —
same everything, only `--max-concurrency` differs:

```
                        conc 1        conc 20       change
mean ITL              8.826 ms      12.033 ms       +36%
mean TPOT             8.861 ms      12.080 ms       +36%
output throughput    108.55 tok/s  1621.29 tok/s    ~15x
benchmark duration   235.83 s        15.79 s        14.9x faster
```

Twenty times the concurrency bought roughly fifteen times the throughput while each token
got only 36% slower. That is the payoff of re-composing the batch every iteration: the
GPU stays full, so extra load turns into throughput instead of per-token latency. If the
batch were frozen per-request instead, those twenty requests would have been served in
sequence and the duration would have stayed near 235 s.

**The honest limit of this evidence.** The sweep ran `--request-rate inf`
([week2-sweep.sh](week2-sweep.sh)), which fires all 100 prompts at once. Nothing arrived
mid-flight. So these numbers demonstrate the *throughput* half of iteration-level
scheduling — a re-composed batch keeps the GPU busy — but say nothing about the
*queueing-delay* half, which is the part about late arrivals joining a batch already in
progress. That half is still untested here. Axis E's length-variance test is where it
gets exercised.

One measured oddity worth carrying forward: `Peak concurrent requests` in
[week2-sweep.log](../week2-sweep.log) is exactly twice `max_concurrency` in all five runs
— 1 to 2, 2 to 4, 5 to 10, 10 to 20, 20 to 40. Too clean to be noise. Week 2 read this as
request waves overlapping at the handoff. Worth confirming it is a client-side accounting
artifact and not the server actually holding double.

---

### 2. Selective batching

**Why it's needed.** Iteration-level scheduling creates its own problem. Once the batch
can be rebuilt freely every iteration, requests in it are no longer in lockstep, and
three combinations stop being batchable the normal way:

1. two requests both still processing their prompts, but with different prompt lengths
2. two requests both generating, but at different positions in their output — their
   cached keys and values are different sizes
3. one request processing its prompt while another is generating — the first hands the
   model its whole prompt at once, the second hands it a single token

Batching means running the identical operation on identically shaped tensors. The normal
approach feeds each layer one tensor shaped *batch size x tokens per request x hidden
size*. A tensor is a rectangle, so every request has to contribute the same number of
tokens. No rectangle fits 1, 1, 2 and 3. The paper's word for this is an irregularly
shaped tensor: it won't combine.

**The asymmetry that makes a fix possible.** Only attention needs to know which token
belongs to which request. The linear layers, layer norm, the residual adds and the
activation function all treat each token independently — a row's output depends only on
that row, so they never look sideways at other rows. Attention is the single operation
where tokens look at each other, and if you run it on a flattened tensor, one request's
prompt would attend to another request's tokens. Attention also normally runs as a GPU
routine that does many matrix multiplications at once and demands they all be the same
shape.

**The mechanism.** Flatten everything into one tensor of *total tokens x hidden size*,
with no batch dimension at all. For a batch contributing 1, 1, 2 and 3 tokens that's 7
rows. Nothing is padded, nothing is wasted. The unit of batching becomes the token, not
the request.

```
Batched together (one flat tensor, request identity irrelevant):
  QKV projection / layer norm / residual add / activation / output projection
        [ a1  b1  c1  c2  d1  d2  d3 ]        7 rows x hidden size

                        | Split
                        v

NOT batched (must respect request boundaries):
  attention, run once per request
        [a1]        [b1]        [c1 c2]    [d1 d2 d3]
       + cached     + cached    (still on its prompt,
         K/V for      K/V for    no cache yet)
         a's earlier  b's earlier
         tokens       tokens

                        | Merge
                        v
        [ a1  b1  c1  c2  d1  d2  d3 ]        back to 7 rows
```

Split slices the queries, keys and values apart per request — the projection that
produces them has already run on the flat tensor at that point. Merge stacks the
attention outputs back together so the rest of the layer is one big matrix multiply
again. This split/merge pair sits inside *every* layer, on *every* iteration.

**The per-request key/value cache.** A request that is generating contributes only its
one newest token, but attention still has to look at all its earlier tokens. Recomputing
those every iteration would be quadratic waste, so the keys and values are cached — and
cached *separately per request*. That separation is exactly what makes a ragged batch
safe. Importantly, the cache is held until the scheduler explicitly asks for it to be
freed; nothing reclaims it automatically. That is what couples this mechanism to the
memory section below.

**What it costs, honestly.** At a fixed batch size Orca's engine is *slightly slower*
than the system it compares against, precisely because it does not batch attention. The
defence is that attention has no model parameters, so batching it wins nothing in
parameter-reuse terms — and reading parameters out of GPU memory is the actual
bottleneck. They also fuse the per-request attention kernels together to claw back the
cost of launching many small ones.

---

### 3. How the scheduler decides what to admit

**State machine.** A request is waiting, then gets selected and has memory reserved for
it, then is running on the GPU, then a token comes back and it is now generating, then it
is selectable again with no new reservation, and so on until it finishes, its memory is
released, and it leaves the pool.

The selection step filters out anything currently running on the GPU. That filter only
ever matters when the scheduler is allowed to inject several batches before waiting for
one to come back, which is a multi-worker pipelining thing. On a single GPU the scheduler
waits every iteration, so that filter never fires. This machine is single-GPU
([serve.sh](serve.sh)), so it never fires here.

**First-come-first-served, defined per iteration.** The guarantee is: for any two
requests in the pool, if one arrived earlier than the other, the earlier one must have
completed at least as many iterations as the later one. Note the paper's own caveat — a
later arrival can still *finish* first if it needs fewer iterations. Arrival order
constrains progress, not completion.

**Selection.** Drop anything running, sort by arrival time, walk the list, take at most
`max_bs` requests, and for each request that hasn't started yet, check its memory
reservation fits.

The detail worth writing down: when a not-yet-started request doesn't fit, the scan
*stops* rather than skipping past it. Skipping would let a later arrival complete an
iteration while an earlier one had completed zero, which breaks the FCFS guarantee above.
So head-of-line blocking is a deliberate price, not a bug — one request that can't get
memory stalls everything behind it, including requests that need no new memory at all.

**`max_bs`** is just the cap on how many requests can be in one batch. Raising it trades
latency for throughput with diminishing returns, and the operator tunes it against a
latency budget. Week 2's own throughput-per-unit-concurrency ratios show that curve
bending on this machine: 2.06x from concurrency 1 to 2, 2.42x from 2 to 5, then 1.81x
from 5 to 10 and 1.66x from 10 to 20.

**What Orca lacks compared to vLLM (for the Day 7 teardown).** Orca has two admission
constraints: `max_bs`, and the memory reservation check. vLLM has three triggers —
`max_num_seqs`, `max_num_batched_tokens`, and running out of KV blocks. Two gaps:

- **No cap on tokens per iteration.** Orca has nothing like `max_num_batched_tokens`.
  `max_bs` counts requests, not tokens, so admitting four requests with 512-token prompts
  means one iteration chewing through 2,048 tokens — a latency spike the scheduler cannot
  see coming. This is directly relevant to Axis B (4,096-token prompts), where a handful
  of requests is a very large number of tokens.
- **No preemption.** Once memory is reserved it is held until the request completes.
  vLLM's block exhaustion is a *dynamic* trigger that can evict or swap a request out.
  Orca's check is a *static* promise made once, at admission.

---

### 4. Memory reservation — the weak point

**Mechanism.** The first time a request is scheduled, the scheduler reserves enough slots
for that request's entire token budget up front, where a slot is the memory for one
token's attention key and value. If the reservation doesn't fit in what's left, the scan
stops.

Reserving the worst case immediately is what makes progress provable. The stated reason is
deadlock avoidance: without it, the scheduler could admit a request, generate happily for
a while, then discover there is no room for the next token's key and value — with no
request able to move forward and free anything.

The total pool size is the easy knob. Given the model and how it's sharded, GPU memory
use depends mostly on that number, so the operator just takes the largest value that
fits. Unlike `max_bs`, it needs no experiments to tune.

**Slot size on this machine.** From the model's `config.json` — 28 layers, 4 key/value
heads, head_dim 128 — and fp16:

```
bytes per token = 2 (one K, one V) x 28 layers x 4 KV heads x 128 head_dim x 2 bytes
                = 57,344 bytes
                = 56 KiB per token
```

Two traps in that formula. Use `num_key_value_heads` (4), not `num_attention_heads` (28)
— this model shares key/value heads across query heads, and using 28 overestimates by 7x.
And AWQ quantizes the *weights*, not the KV cache, so the cache is still fp16 and the
`2 bytes` is right.

**Three ways to spend that memory.**

```
one slot = one token's K and V = 56 KiB
~14 GiB KV pool = 14,336 MiB / 56 KiB = 262,144 slots

(A) reserve the MODEL's maximum length for every request
    (what the older system Orca compares against does)
      8,192 x 56 KiB                     = 448 MiB reserved per request
      Week 2 actually needs 768 x 56 KiB =  42 MiB
      wasted = 448 - 42 = 406 MiB        = 90.6% of the reservation
      capacity: 262,144 / 8,192          = ~32 requests at once

(B) reserve THIS REQUEST's budget: its prompt + how many tokens it asked for
    (what Orca does)
      Week 2 asks for 256 output tokens, so 512 + 256 = 768
      768 x 56 KiB                       =  42 MiB reserved per request
      wasted on this workload            = ~0%
      capacity: 262,144 / 768            = ~341 requests at once

(C) hand out 16-token blocks on demand as the sequence grows
    (what vLLM does)
      768 / 16 = 48 blocks, divides evenly, so no part-used block
      capacity: 16,384 blocks / 48       = ~341 sequences at once
                                           (Plan Part 2.2 rounds to ~330)
```

**The ~10x gap is (A) versus (C) — not Orca versus vLLM.** This is a correction to
[week3-day-by-day.md:538](../week-wise-plan/week3-day-by-day.md#L538), which says "a slot
must hold max-model-len = 8,192 tokens." That mixes up two different numbers:

- `--max-model-len 8192` is a *server* setting in [serve.sh](serve.sh) — the longest
  sequence this server will accept from anyone.
- Orca's reservation is a *per-request* quantity — this request's prompt plus the number
  of output tokens this particular client asked for.

Orca reserves the second. So 448 MiB and 90.6% are correct arithmetic, but they describe
row (A), the older system that sizes every reservation by the model's limit regardless of
what the request needs. Describe (A) as "~32 requests at once, each holding 8,192 slots"
— a slot is per-token, not per-request.

**What the harness actually asked for, from the logs.** The config echo at the top of
each run in [week2-sweep.log](../week2-sweep.log) records `random_input_len=512`,
`random_output_len=256`, `ignore_eos=True`. And every one of the five result blocks
reports `Total generated tokens: 25600` — exactly 100 x 256, with zero variance across
500 requests. So the client declared 256 output tokens and got exactly 256 every single
time. Orca would have reserved 768 slots and used all 768.

Measured prompt length was slightly larger than nominal: `Total input tokens: 54100` over
100 prompts is 541 each, not 512 — the chat template adds about 29 tokens. So real usage
is ~797 tokens against the 768 the arithmetic above uses.

**So Week 2 is the worst possible advertisement for on-demand blocks**, and that is the
honest thing to record. Rows (B) and (C) tie at ~341. Three properties of this workload
cause that tie, all three visible in the log: every request is the same length, 768
divides evenly by the 16-token block size so no block is left part-used, and
`ignore_eos=True` forces the declared output length to be exactly correct. Under those
conditions, reserving a contiguous chunk up front is already optimal and paging buys
nothing. Nothing in Week 2 could have shown a difference.

**So where does reserving up front actually lose?** Not on the size of the reservation —
on four things a promise made once, at admission, cannot do:

1. **Output length isn't knowable in advance.** The budget has to be set before generation
   starts. A chat client that can't predict how long its answer will be sets the cap high
   to be safe, and that worst case is then held for the request's entire life.
   `ignore_eos=True` hides this completely — it forces exactly 256 tokens, so declared and
   actual coincide by construction. Real traffic has neither property. This is the deep
   reason, and it survives the correction above: **you cannot know the output length in
   advance**, so anything that commits memory at admission must either over-reserve or
   risk deadlock. The waste is the price of that impossibility — and this workload is
   rigged never to charge it.
2. **No sharing between requests.** Two requests with the same system prompt reserve and
   fill separate slots for identical keys and values. Blocks can be shared instead, and
   only copied when one request diverges.
3. **No taking memory back.** Slots are held to completion. Running out of blocks is a
   condition vLLM can *react* to, by evicting or swapping a request out; running out of
   reservations is something Orca can only *prevent*, by refusing to admit.
4. **Head-of-line blocking**, from the stop-don't-skip rule in the section above combined
   with reservation — one request that can't get memory stalls everyone behind it.

**Three steps, not a two-way fight.** The older system sizes reservations by the model,
Orca by the request, vLLM by the token. Orca is the middle step, not the villain.

**Testable on this machine in Week 3.** If the reasoning above is right, the tie between
(B) and (C) should break as soon as request lengths stop being uniform. Axis E is the
test: mixed lengths mean part-used blocks and mispredicted budgets, which is where
on-demand allocation should pull ahead. Prediction to check on Day 7: uniform-length axes
(A, B, C, D) should land near the Part 2.2 capacity estimates, and Axis E should come in
below them.

---

### 5. What Orca does not solve

> Orca schedules time at iteration granularity but still reserves memory at request
> granularity — it fixed the time axis and left the memory axis for PagedAttention.

The support for that sentence is that Orca's scheduling loop does two things at two
different granularities. Batch membership is re-decided on every single iteration, inside
the main loop. Memory is reserved once per request lifetime, only when a request starts.
That mismatch is the seam PagedAttention cuts along.

The paper also names one open problem itself: Orca deliberately fuses the scheduler and
the execution engine together, giving up the clean separation between a serving layer and
an engine layer that other systems keep. It leaves as an open question how to design an
interface that supports both scheduling and memory techniques without losing that
separation.

---

### Glossary

| Term | Meaning |
|---|---|
| Iteration | One pass through all layers; produces one token per request in the batch |
| Prompt phase | Processing the whole input prompt, in one iteration (paper: initiation) |
| Generating phase | Producing one token per iteration thereafter (paper: increment) |
| Queueing delay | Time a new request waits before any computation starts on it |
| Irregularly shaped | Ragged — rows of different lengths, so no rectangle fits |
| Token-wise batching | The unit of batching is the token, not the request |
| Split / Merge | Ops that separate requests for attention and rejoin them afterwards |
| K/V cache | Saved keys and values from earlier tokens, so they aren't recomputed |
| Slot | Memory for one token's attention key and value — 56 KiB on this machine |
| Reservation | Slots promised to a request at admission, held until it finishes |
| `max_bs` | Max requests in one batch |
| Iteration-level FCFS | An earlier arrival has run at least as many iterations as any later one |
| Head-of-line blocking | One unadmittable request stalling everything queued behind it |


## Predicted capacity (before measuring)

Pre-registered Day 4, before the pod was deployed.

| # | Prediction | Predicted |
|---|---|---|
| P1 | `GPU KV cache size` from startup log | ~250,000–270,000 tokens |
| P2 | `Maximum concurrency for 8,192 tokens per request` | ~31x |
| P3 | Anchor conc-20 output tok/s vs Week 2's 1621.29 | within ±5% |
| P4 | Cold-start conc-1 TTFT, run last | ~25 ms (Week 2 measured 98.41 ms) |
| P5 | First ceiling level where `waiting > 0` | 16 or below |
| P6 | KV usage at that first `waiting > 0` | low (<30%) → token budget binds first, not memory |
| P7 | Any preemptions | yes, at 48 or 64 |

Reasoning recorded at the time: from Part 2's arithmetic (56 KiB/token, 13–14 GiB
available after weights), the pool should hold roughly a quarter-million tokens. P4
assumed Week 2's high conc-1 TTFT was a cold-start artifact, since conc-1 ran first
in that sweep. P6 was the one that mattered: seeing `waiting > 0` and calling it
"the memory ceiling" would be wrong if the pool were still mostly empty.

**All seven resolved. Five hits, two misses.** Scorecard and post-mortem below.

## Measured capacity (from the startup log)

Source: RunPod Logs tab, engine startup 09-09 12:07:22–12:08:48. The container start
command auto-starts the server, so there is no local `server-startup.log` this session
(see Operational findings). Transcribed to `results/week3/capacity-config.txt`.

```
(APIServer pid=51) INFO 09-09 12:07:22 [api_utils.py:272] non-default args:
  {'model_tag': 'Qwen/Qwen2.5-7B-Instruct-AWQ', 'model': 'Qwen/Qwen2.5-7B-Instruct-AWQ',
   'max_model_len': 8192, 'quantization': 'awq', 'gpu_memory_utilization': 0.9}
(EngineCore pid=671) INFO 09-09 12:08:48 [gpu_worker.py:578] Available KV cache memory: 13.42 GiB
(EngineCore pid=671) INFO 09-09 12:08:48 [kv_cache_utils.py:1869] GPU KV cache size: 251,344 tokens,
   Maximum concurrency for 8,192 tokens per request: 30.68x
```

Cross-checked against `vllm:cache_config_info` in `metrics-idle-snapshot.txt`:

```
block_size="16"  num_gpu_blocks="15709"  enable_prefix_caching="True"
prefix_caching_hash_algo="sha256"  num_gpu_blocks_override="None"
```

15,709 blocks × 16 tokens = **251,344 tokens**. Two independent sources, exact
agreement.

| Quantity | Value | Source |
|---|---|---|
| Available KV cache memory | 13.42 GiB | startup log |
| KV pool | 251,344 tokens / 15,709 blocks | log + `cache_config_info` |
| Block size | 16 tokens | `cache_config_info` |
| Engine's own max concurrency @8192 tok | 30.68x | startup log |
| Prefix caching | **enabled by default** | `cache_config_info` |
| Chunked prefill | enabled | engine config line |
| `max_num_seqs` / `max_num_batched_tokens` | **not printed** | see Open questions |

**P1 hit** (251,344 vs ~250–270k). **P2 hit** (30.68x vs ~31x). The Part 2 arithmetic
was sound: 56 KiB/token × 251,344 tokens ≈ 13.4 GiB against the reported 13.42 GiB.

Also recorded from the engine config line, for `ENVIRONMENT.md`:
`dtype=torch.float16`, attention backend `FLASH_ATTN` (FlashAttention 2),
`AutoAWQMarlinLinearMethod` via `MarlinLinearKernel`, `cudagraph_mode=FULL_AND_PIECEWISE`,
`max_cudagraph_capture_size=512`, engine `seed=0`, `revision=main`.

## Results — Axis A (decode-bound, 128 in / 1024 out)

**Dropped in the Week 3 re-cut.** Month 3 "Perturb" work, not Month 1 baseline.

## Results — Axis B (prefill-bound, 4096 in / 32 out)

**Dropped in the Week 3 re-cut.** Deferred to Month 3.

Partial substitute from data that does exist: single-request prefill of 541 tokens
took 108 ms (cold-start run) → ~5,000 tok/s. Twenty concurrent 541-token prefills took
1,033 ms → ~10,500 tok/s aggregate. Directional only.

Axis D turned out to be a prefill-bound experiment by accident, which is the main
reason its throughput numbers behave as they do.

## Results — Axis C (Week 2 anchor, 512 in / 256 out)

Two runs at 512-in/256-out bracketing the ceiling series. Prefix-cache hit rates from
`metrics.log` are what make the comparison interpretable.

| Metric | Week 2 conc-20 | **Week 3 anchor conc-20** | Δ |
|---|---|---|---|
| Prefix cache hit rate | not measured | **3.0%** | — |
| Output tok/s | 1621.29 | **1066.60** | **−34.2%** |
| Req/s | 6.333 | **4.166** | **−34.2%** |
| TTFT mean | 71.39 ms | **1033.30 ms** | **14.5x worse** |
| TTFT p50 | 72.64 ms | 1107.89 ms | 15.3x |
| TTFT p99 | 128.50 ms | 1811.71 ms | 14.1x |
| ITL mean | 12.033 ms | 14.694 ms | +22% |
| **ITL p50** | (mean 12.033) | **11.839 ms** | **−1.6%** |
| Peak KV usage | not measured | **6.0%** | — |

| Metric | Week 2 conc-1 | **Week 3 cold-start conc-1** | Δ |
|---|---|---|---|
| Prefix cache hit rate | not measured | **3.0%** | — |
| Output tok/s | 108.55 | **108.39** | **−0.15%** |
| Req/s | 0.424 | 0.4234 | −0.14% |
| ITL mean | 8.826 ms | 8.811 ms | −0.17% |
| TPOT mean | 8.861 ms | 8.836 ms | −0.28% |
| TTFT mean | 98.41 ms | 108.35 ms | +10.1% |
| Peak KV usage | not measured | **0.3%** | — |

**C1. The conc-1 configuration reproduces almost perfectly.** Throughput within 0.15%,
ITL within 0.17%, req/s within 0.14% — across two pods, a week apart, a re-downloaded
copy of the weights, and a different position in the run order. The measurement rig is
sound.

**C2. The conc-20 configuration does not reproduce** — 34% less throughput, 14.5x
worse TTFT — yet **ITL p50 matches Week 2's ITL mean to within 1.6%**. Decode speed is
identical; the entire deficit is TTFT. Not "the GPU was slower this time"; something
specific to prefill differed.

**C3. Week 2's conc-1 TTFT was NOT a cold-start artifact. P4 refuted.** The cold-start
run was placed last, on a GPU that had been at 100% utilisation for 17 minutes with SM
clocks at 2.4 GHz, and produced 108 ms — 10% *higher* than Week 2's 98 ms, not the
~25 ms predicted. The Week 2 open question is answered: 108 ms is what a single
541-token prefill costs on this GPU.

**C4. Both short-prompt runs measured a 3.0% hit rate, and that number is exactly the
chat template.** From `metrics.log`: anchor 54,100 queries / 1,600 hits; cold-start
21,640 / 640. With 100 and 40 prompts, that is **exactly 16 tokens — one block — per
request** (100 × 16 = 1,600; 40 × 16 = 640). The Week 2 security note predicted this:
"that wrapper alone is a shared prefix, so a small non-zero hit rate is expected even
with fully random prompts." Confirmed and quantified to the block. It is 16 tokens
rather than the ~29 the template occupies because a partial block is not cacheable.

**C5. So the Week 2 / Week 3 anchor gap is not caused by *this* run's cache** — both
Week 3 short-prompt runs were effectively cold at 3.0%. It comes from Week 2's side.
`week2-sweep.sh` runs 1/2/5/10/20 in one loop with `--seed 42`, `--num-prompts 100` and
fixed lengths, so all five runs sent the **same 100 prompts**; runs 2–5 re-sent prompts
already cached by run 1, and conc-20 was the fifth of five.

Week 2 has no hit-rate data to prove this directly — that is the gap the Week 2 note
itself identified. But the mechanism is now measured at a different prompt length in
Axis D, where an identical setup produced a 99.8% hit rate and a 13x TTFT drop. And the
Week 2 table carries its own fingerprint: TTFT *falls* from 98.41 ms at concurrency 1 to
24.36 ms at concurrency 2, which is impossible unless the prefill work differed between
runs.

**Consequence: four of five TTFT rows in the Week 2 table cannot be read as prefill
cost.** ITL and TPOT are unaffected — decode is per-token work no cache can skip, which
is exactly why ITL reproduced to 1.6% while TTFT did not.

**C6. Neither short-prompt run came near the memory ceiling.** Peak KV usage 6.0% at
conc 20, 0.3% at conc 1. At 797 tokens per request (512 + 256 + 29), 20 concurrent
requests reserve 1,000 of 15,709 blocks — 6.4%, matching the measured 6.0% almost
exactly. Yet `num_requests_waiting` still reached 11 during the anchor run. **Requests
queued with the KV cache 94% empty.** This is P6's scenario, measured.

## Results — Axis D (ceiling hunt, 6144 in / 1024 out)

Five runs, 6144-in/1024-out, `--ignore-eos`, seed 42, prompts = 2 × concurrency (min
32). All completed with `failed: 0`. Client metrics from the run JSONs; scheduler state
from `metrics.log` at 0.5 s resolution.

| conc | blocks needed | % pool | **prefix hit %** | out tok/s | TTFT mean ms | TPOT mean ms | ITL p95 ms | peak running | mean running | peak waiting | peak KV % | **preemptions** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 3,600 | 22.9% | **7.5** | 361.67 | 3332.7 | 18.86 | 14.49 | 8 | 5.9 | 6 | 22.8 | 0 |
| 16 | 7,200 | 45.8% | **99.8** | 811.58 | 255.7 | 19.43 | 20.44 | 16 | 10.3 | **0** | 45.6 | 0 |
| 32 | 14,400 | 91.7% | **50.0** | 626.66 | 9091.1 | 41.83 | 34.06 | 32 | 21.8 | 26 | 91.3 | 0 |
| 48 | 21,600 | 137.5% | **0.3** | 397.06 | 34906.7 | 78.69 | 496.29 | 40 | 28.5 | 46 | **100.0** | **10** |
| 64 | 28,800 | 183.3% | **0.3** | 391.11 | 66188.1 | 82.93 | 497.93 | 40 | 30.2 | 61 | **100.0** | **15** |

### D1. The prefix cache is measured, and it explains the run series

Query and hit counts from `metrics.log`, in tokens:

| run | queries | hits | rate | interpretation |
|---|---|---|---|---|
| ceil c8 | 197,536 | 14,864 | 7.5% | first exposure to these 32 prompts |
| ceil c16 | **197,536** | **197,120** | **99.8%** | same 32 prompts, fully resident |
| ceil c32 | 395,072 | 197,408 | **50.0%** | 64 prompts, first 32 still cached |
| ceil c48 | 592,608 | 1,536 | 0.3% | 96 prompts, cache overrun |
| ceil c64 | 790,144 | 2,048 | 0.3% | 128 prompts, cache overrun |

Three points worth stating plainly.

**Query counts equal the JSONs' `total_input_tokens` exactly** (c8: 197,536; c32:
395,072). Every prompt token is queried against the cache, so the hit rate directly
measures the fraction of prefill work skipped.

**c8 and c16 issued the identical query count — 197,536 — because they sent the
identical 32 prompts.** Same seed, same lengths, same `--num-prompts`. c8 paid for them;
c16 got 99.8% free.

**c32's 50.0% is arithmetic, not coincidence.** 64 prompts of which the first 32 were
cached; its 197,408 hit tokens are within 0.15% of c16's 197,120. The cache returned
exactly the same 32 prompts and nothing more.

**Why retention collapsed between 32 and 96 prompts.** 32 × 6,173 = 197,536 tokens fits
inside the 251,344-token pool. 96 × 6,173 = 592,608 does not — and live requests need
the pool at the same time — so retention falls to the 0.3% template floor.

### D2. The throughput curve is unusable, and that is the finding

Reported output throughput: 361.67 → 811.58 → 626.66 → 397.06 → 391.11 tok/s.

That is **not a scaling curve.** It tracks the hit rate. The 812 tok/s "peak" at
concurrency 16 is a run that skipped 99.8% of its prefill; the 627 at concurrency 32
skipped 50%. Neither describes what this server does with real traffic.

Only three points are cache-comparable — c8 at 7.5%, c48 and c64 at 0.3%:

| conc | hit rate | out tok/s | mean TTFT |
|---|---|---|---|
| 8 | 7.5% | **361.67** | 3.3 s |
| 48 | 0.3% | **397.06** | 34.9 s |
| 64 | 0.3% | **391.11** | 66.2 s |

**Throughput is flat from concurrency 8 to 64 — 362 to 397 tok/s, under 10% — while
mean TTFT rises 20x, from 3.3 s to 66.2 s.** At a 6,144-token prompt length this server
saturates at or below concurrency 8. Every slot past that buys queueing delay and
nothing else.

That makes Axis D unintentionally a prefill-bound experiment: at 6,144 in to 1,024 out,
prefill dominates, and prefill does not batch its way out of a bandwidth limit the way
decode does.

**Retraction.** An earlier draft of these notes read "peak useful throughput is at
concurrency 16" and built a "2.1x theory-versus-measurement gap" on it. Both were
artifacts of the 99.8% hit rate, produced by reading the client-side JSONs without the
cache telemetry. Withdrawn. The correct reading is D3.

### D3. The KV arithmetic is accurate to within one sequence

15,709 blocks ÷ 450 blocks per full-length request = **34.9 concurrent requests**.

Measured `num_requests_running` at the two over-subscribed levels:

| conc | peak running | mean running | **modal running** |
|---|---|---|---|
| 48 | 40 | 28.5 | **35** (102 of 512 samples), then 34 (87) |
| 64 | 40 | 30.2 | **35** (175 of 678 samples), then 34 (114) |

**The server settles at 34–35 concurrent sequences at both levels. The arithmetic
predicted 34.9.** Two runs asking for 48 and 64 slots converged on the same number, and
it is the number the block formula gives.

Peak 40 exceeds 34.9 because a request early in decode holds 6,144 + n tokens rather
than the full 7,168, so more fit briefly. Means are lower because runs include ramp-up
and drain.

So the KV capacity formula works. What it does **not** predict is where throughput
peaks — D2 shows saturation around concurrency 8, well below the 35 the memory holds.
Capacity and useful concurrency are different quantities, and only the first is what
the arithmetic answers.

### D4. Preemption threshold: 97.5% KV occupancy

25 preemption events, all in c48 (10) and c64 (15), zero elsewhere. KV occupancy at
every event: **min 97.4%, max 97.7%, mean 97.53%**.

Concurrency 32 peaked at 91.3% and preempted zero times. The threshold is bracketed
tightly: **no preemption at 91.3%, preemption at 97.4%.**

Events arrive in bursts of five, and the counters trace the mechanism step by step:

```
ts 1788957450  preempt 1   KV 97.6%  running 39  waiting 9
ts 1788957457  preempt 2   KV 97.6%  running 38  waiting 10
ts 1788957463  preempt 3   KV 97.5%  running 37  waiting 11
ts 1788957470  preempt 4   KV 97.4%  running 36  waiting 12
ts 1788957477  preempt 5   KV 97.4%  running 35  waiting 13
```

Running walks 39 → 35 while waiting climbs 9 → 13, one request per step, ~7 s apart,
KV pinned at 97.5% throughout — then the pattern repeats ~70 s later. The scheduler
evicts one sequence at a time until the pool has headroom. This is Orca's
memory-reservation weak point (Part 4 of the paper notes) in live telemetry.

Time spent near saturation:

| conc | samples ≥90% | ≥95% | ≥99% | mean KV % |
|---|---|---|---|---|
| 8 | 0 | 0 | 0 | 15.1 |
| 16 | 0 | 0 | 0 | 27.4 |
| 32 | 10 | **0** | 0 | 56.7 |
| 48 | 272 | 190 | 57 | 74.6 |
| 64 | 404 | 283 | 84 | 78.8 |

Concurrency 32 touched 90% for 10 samples (~5 s) and never reached 95%. Concurrency 48
spent ~95 s above 95%. That is the boundary.

**P7 hit, directly measured.** It also confirms what p95 ITL implied: 496 ms of dead air
between two tokens of one request is a preempted sequence waiting for blocks.

### D5. Queuing is not a ceiling signal — P6, measured

`num_requests_waiting` exceeded zero in six of eight runs, including the warm-up. KV
occupancy at the *first* such moment:

| run | KV % at first `waiting > 0` | peak KV % | preemptions |
|---|---|---|---|
| warm-up c16 | 1.9% | 4.9% | 0 |
| anchor c20 | **1.0%** | 6.0% | 0 |
| ceil c8 | 1.8% | 22.8% | 0 |
| ceil c16 | **never queued** | 45.6% | 0 |
| ceil c32 | 75.9% | 91.3% | 0 |
| ceil c48 | 1.6% | 100.0% | 10 |
| ceil c64 | 2.5% | 100.0% | 15 |

In five of six runs that queued, the first queuing occurred with the KV cache **under
3% full**. The cause is arrival burst, not capacity: `--request-rate inf` fires
`max-concurrency` requests simultaneously at run start and the scheduler admits them
over several steps.

**So `waiting > 0` is nearly worthless as a ceiling indicator, and P6 was right to
insist on reading occupancy alongside it.** Had this week reported "the memory ceiling
is at concurrency 8, where requests began queuing," it would have been wrong by a factor
of four. The signals that work are **peak KV occupancy** and **`num_preemptions_total`**,
neither of which was pre-registered.

One clean detail: **ceiling c16 never queued at all** — zero waiting across 116 samples.
With 99.8% of its prefill served from cache there was almost no admission work to queue
for. The cache did not merely speed the run up; it removed the queue.

`num_requests_waiting_by_reason` was `capacity` for every waiting request and `deferred`
for none, across all eight runs. No LoRA or KV-transfer effects.

## THE MEMORY CEILING FINDING

On this GPU (RTX PRO 4000 Blackwell, 24 GB) serving Qwen2.5-7B-Instruct-AWQ at
`--max-model-len 8192 --gpu-memory-utilization 0.90`, the engine allocates a KV pool of
**251,344 tokens (15,709 blocks × 16)** from the **13.42 GiB** of VRAM remaining after
weights. A 7,197-token request (6144 in + 1024 out + ~29 template) reserves **450
blocks**, so the pool holds **34.9** of them.

**Measured: the server settles at 34–35 concurrent sequences.** Asked for 48 slots it
ran 35 (modal, 102 of 512 samples); asked for 64 it ran 35 again (modal, 175 of 678).
The block arithmetic predicts this configuration's capacity to within one sequence.

**The ceiling is enforced by preemption at 97.5% KV occupancy.** Twenty-five events, all
at 97.4–97.7%, all in the two over-subscribed runs. Concurrency 32 peaked at 91.3% and
preempted zero times; concurrency 48 spent 95 seconds above 95% and preempted ten times.
The mechanism is visible in the counters: running walks down one sequence per scheduler
step while waiting climbs, KV pinned at 97.5%, until headroom returns.

**Requests queue long before memory is the constraint.** Five of six runs that queued did
so with KV under 3% — arrival burst, not capacity. At concurrency 20 with 512-token
prompts, 11 requests waited while the pool was 94% empty. Queuing and the memory ceiling
are unrelated phenomena at this scale; conflating them would have put the ceiling at
concurrency 8 instead of 35.

**Throughput saturates far below the memory ceiling.** Among the three cache-comparable
runs (7.5%, 0.3%, 0.3%), output throughput is 362 / 397 / 391 tok/s at concurrency
8 / 48 / 64 — flat within 10% — while mean TTFT rises from 3.3 s to 66.2 s. At a
6,144-token prompt this server is prefill-bound and gains nothing from concurrency past
~8. **The useful operating point and the memory capacity differ by more than 4x, and only
the second is what the KV formula describes.**

**What limits the other two runs is the prefix cache, not the hardware.** Concurrency 16
reported 812 tok/s on a 99.8% hit rate and concurrency 32 reported 627 on 50.0%. Both are
measurements of cache reuse. They are excluded from the throughput conclusion above and
should not be quoted as scaling results.

## Why VRAM is the wrong ceiling signal

`gpu.csv`, 2,472 samples at 1 Hz; `metrics.log`, 4,892 samples at 0.5 Hz. Both span idle,
all eight runs, and idle again.

| Window | samples | memory.used MiB | util.gpu avg | util.memory avg | **mean KV %** | **peak KV %** | SM MHz | temp max |
|---|---|---|---|---|---|---|---|---|
| idle, pre-sweep | 193 | 20,874 | 0% | 0% | 0.0 | 0.0 | 172–180 | 28 |
| warm-up c16 | 34 | 20,874–21,103 | 24% | 19% | 0.8 | 4.9 | 172–2490 | 43 |
| anchor c20 | 45 | 20,874 | 53% | 39% | 2.3 | 6.0 | 180–2482 | 56 |
| ceiling c8 | 112 | 20,874 | 80% | 63% | 15.1 | 22.8 | 180–2467 | 67 |
| ceiling c16 | 62 | 20,874 | 65% | 60% | 27.4 | 45.6 | 180–2475 | 64 |
| ceiling c32 | 128 | 20,874 | 81% | 67% | 56.7 | 91.3 | 180–2467 | 68 |
| ceiling c48 | 272 | 20,874 | 91% | 70% | 74.6 | **100.0** | 180–2467 | 68 |
| ceiling c64 | 361 | 20,874–20,888 | 93% | 71% | 78.8 | **100.0** | 180–2467 | 69 |
| cold-start c1 | 110 | 20,874 | 85% | **75%** | 0.2 | 0.3 | 180–2242 | 68 |

**V1. `memory.used` is a flat line while `kv_cache_usage` covers its entire range.**
Across the session VRAM ranged **20,874 to 21,103 MiB of 24,467 — a spread of 229 MiB,
or 1.1%**. Over the same period KV occupancy went from **0.0% to 100.0%**, a full-scale
excursion, and the server went from idle to preempting twenty-five times.

Those two columns sit side by side in the table above, and that pairing is the whole
argument. vLLM claims 90% of VRAM at boot and never releases it, so `nvidia-smi` reports
the reservation, not the occupancy. **It cannot detect the ceiling; it did not move by
even 1% while the ceiling was being hit.**

This resolves the apparent contradiction in the Week 1 notes — "~87% VRAM reserved"
alongside "KV cache usage under 1%". Both were true, measuring different things: what the
process holds from the driver, versus what the engine is using of what it holds.

**V2. Which column Week 2's ~35% was.** At anchor conc-20, `utilization.gpu` averaged 53%
and `utilization.memory` 39%; the latter is the plausible source. Both are *time*
percentages — the share of the interval in which a kernel was running, or the memory bus
was active. Neither means "35% of the GPU" or "35% of VRAM", which is how such a number is
usually read.

**V3. `utilization.memory` is highest at concurrency 1 — Week 1's decode claim,
confirmed.** The cold-start run shows the **highest average memory-bus utilisation of the
session (75%)** while producing the **lowest throughput (108 tok/s, 3.6x less than
concurrency 64)** and using **0.3% of the KV pool**. One sequence decoding alone reads the
full weight matrix to emit a single token: it saturates the bus while compute units idle
and the cache sits empty. Batching amortises that read, which is why concurrency 64 gets
3.6x the throughput at *lower* bus utilisation. Week 1 inferred this from a throughput
ratio; here all three instruments agree.

**V4. SM clocks ramp 13.8x, and this bounds the cold-start question.** Idle SM clock is
172–180 MHz, under load 2,467–2,490 MHz. Clock ramp is real and large, which is why the
cold-start hypothesis (P4) was plausible. It was refuted anyway: the cold-start run
executed on a GPU at 68 °C after 17 minutes of sustained load and reproduced Week 2's
cold conc-1 number to 0.15% on throughput. Its own SM max was 2,242 MHz — the lowest of
any load window — because one decoding sequence never demands full clocks.

## Weekend Teardown: what triggers queuing

Not started. Deferred to the weekend, 1-hour hard cap.

The telemetry has already answered more than the teardown was scoped to: queuing driven
by arrival burst at KV under 3% (D5), and the ceiling enforced by preemption at 97.5%
(D4). What source reading would add:

- **T3 confirmed** — the preemption path. Find where a failing `allocate_slots` leads to
  eviction, and why the threshold sits at ~97.5% rather than 100% (block fragmentation,
  or reserved headroom).
- **T1/T2 still unquantified** — `max_num_seqs` and `max_num_batched_tokens` are not
  printed by this build, so the sequence cap and per-step token budget are unknown.
  Readable from 0.28.0's defaults in the local clone. Peak running hit exactly 40 in both
  over-subscribed runs, which is suspicious enough to check against `max_num_seqs`.
- **The admission burst** — why 20 simultaneous arrivals produce 11 waiting when the pool
  is 94% empty. Almost certainly the per-step token budget, which would make it a T2
  observation.

## Predictions: measured vs predicted

| # | Prediction | Predicted | Measured | Verdict |
|---|---|---|---|---|
| P1 | `GPU KV cache size` | ~250–270k tokens | **251,344** (15,709 × 16, two sources) | **hit** |
| P2 | Max concurrency @8192 tok | ~31x | **30.68x** | **hit** |
| P3 | Anchor conc-20 tok/s vs 1621.29 | within ±5% | **1066.60, −34.2%** | **miss** |
| P4 | Cold-start conc-1 TTFT | ~25 ms | **108.35 ms** (Week 2: 98.41) | **miss** |
| P5 | First level where `waiting > 0` | 16 or below | **conc 8**, and warm-up at 16 | **hit** |
| P6 | KV usage at first `waiting > 0` | <30% → token budget binds | **1.0–2.5%** in 5 of 6 queuing runs | **hit** |
| P7 | Any preemptions | yes, at 48 or 64 | **10 at c48, 15 at c64, 0 elsewhere** | **hit** |

**Five hits, two misses, nothing unresolved.**

**P6 was the prediction worth writing, and it paid.** It existed to stop this week
reporting "requests started queuing, therefore the memory ceiling is here." Queuing began
at 1.0% KV occupancy; the ceiling was at 97.5%. The gap between those is a factor of 97,
and the note that closed it existed before any data did.

**P5 was a hit and a badly chosen question.** `waiting > 0` fires on arrival burst in
almost every run including the warm-up, so predicting where it first appears measured the
load generator rather than the server. The signals that work — peak KV occupancy and
preemption count — were not pre-registered. Worth carrying forward as a lesson about
choosing the metric before predicting its value.

**Why P3 was wrong.** It assumed that repeating a configuration reproduces its number. It
does not, when the engine carries state between runs. Week 2's conc-20 figure was the
fifth of five runs sharing one prompt set; Week 3's anchor measured a 3.0% hit rate, i.e.
effectively cold. The ±5% was reasoning about hardware variance and ignored engine state
entirely. **A benchmark against a stateful server is only reproducible if the state is
cleared or recorded, and this repo did neither.**

**Why P4 was wrong.** It over-fitted one anomalous point to the convenient explanation:
Week 2's conc-1 TTFT was the highest of that sweep and conc-1 ran first, so warm-up looked
causal. Run last, on a thoroughly hot GPU, the same configuration reproduced throughput to
0.15% and TTFT to within 10%. The ordering was coincidence, and the real cause ran the
other way — not warm-up inflating conc-1, but cache reuse deflating every other run.
**P4 being wrong is what produced the week's main finding**, which is the argument for
pre-registering predictions rather than interpreting after the fact.

## Operational findings

**O1. The container start command is the container's PID 1.** The RunPod template
auto-starts vLLM from its start command, so a server holds port 8000 before any shell
exists. Running `serve.sh` on top fails with `OSError: [Errno 98] Address already in use`,
and `pkill -f "vllm serve"` **terminated the entire pod**, because killing PID 1 stops the
container. Resolution: serving flags moved into the start command itself —
`--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90`
— so the auto-started server *is* the intended configuration. `serve.sh` remains in the
repo as the version-controlled record of it.

**O2. The default start command served the wrong context length.** Before O1's fix,
`/v1/models` reported `max_model_len: 32768` while `serve.sh` and `ENVIRONMENT.md` specify
8192. The template ignored the flags in its own container command. **Second occasion this
template has served a configuration other than the one specified** — Week 1 was the same
failure with the model itself. After the fix, `non-default args` and `models.json` both
confirmed 8192.

**O3. The network volume was not mounted.** The repo was at `/vllm-workspace/`, not
`/workspace/`; the startup log reports `Filesystem type for checkpoints: OVERLAY` and
15.58 s spent **downloading** weights that should have been resident on volume
`q1-serving`. Everything written lived on container disk. **A repo clone was lost to this
earlier the same day, and `metrics.log` was nearly lost the same way.** `df -h /workspace`
is the first command of every session: a network mount is required, `overlay` means the
volume is absent.

**O4. The container image lacks git and file.** Both are container filesystem, not volume,
so `apt-get install -y git file` runs every session. Not a defect — it is what a stateless
container means — and it belongs in the checklist rather than in memory.

**O5. An SSH disconnect killed the sweep's shell but not the samplers.** Wi-Fi dropped
during the conc-64 run. On reconnect, `metrics-sampler.sh` (pid 4405) and `gpu-sampler.sh`
(pid 5656) were still running on orphaned ptys, and the sweep had in fact completed — all
seven JSONs present with `failed: 0`. The samplers surviving is why `metrics.log` covers
the full 2,598-second session. Long jobs belong under `tmux` regardless.

**O6. Startup output goes to container stdout, not a file.** With the server auto-started
there is no local `server-startup.log`, so capacity numbers were transcribed by hand from
the RunPod Logs tab into `capacity-config.txt`. The full engine log was separately exported
as `logs.txt` and is the better artifact, covering 12:07:22–12:11:57 with the complete
engine config line.

**O7. The samplers were the highest-value thing built this week.** Four scripts, three of
them trivial. `metrics.log` turned three unresolved predictions into measurements, proved
the prefix-cache finding outright, located the preemption threshold to 0.3 percentage
points, and forced the retraction of an incorrect throughput conclusion drawn from the
client-side JSONs alone. The Week 2 method gap — "`/metrics` was never sampled during the
runs" — was the correct thing to have identified, and closing it was worth more than any
additional run would have been.

**Session cost:** pod up ~12:07, terminated ~13:08. Sweep proper 12:29–12:48.
`metrics.log`: 4,892 samples over 2,598 s.

**For the Month 1 report's method section:** the serving configuration is asserted by a
version-controlled script and verified at runtime from `/v1/models` and the engine's
`non-default args` line every session, because the platform was twice observed to serve a
configuration other than the one specified; and every latency figure is reported alongside
the prefix-cache hit rate of the run that produced it, because a fixed seed across runs was
found to make TTFT a measure of cache reuse rather than of prefill.
