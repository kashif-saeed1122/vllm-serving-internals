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

# ORCA (OSDI '22) — reading notes

Yu et al., *ORCA: A Distributed Serving System for Transformer-Based Generative Models*.
Five things only. Section and page refs are to the paper's own numbering (521–538).

---

## ① Iteration-level scheduling

**The problem (§3, C1, p.524 + Figure 3).** Existing systems schedule at *request* granularity: the
serving system and the engine interact only when a batch is dispatched or when the whole batch
finishes. The batch is fixed until every request in it completes. Two consequences:

- **Early-finished requests.** Each request needs a different number of iterations. A request that
  finishes at token 10 while a batchmate runs to token 500 stays in the tensor for 490 more
  iterations, producing padding tokens that are discarded (the `-` cells in Figure 3). Wasted GPU.
- **Late-joining requests.** A request arriving mid-batch waits for the entire batch to drain before
  any computation starts on it. That wait is the **queueing delay**.

Its answer also can't be returned early — the engine only returns results when the whole batch is
done, so the finished request's latency includes its batchmates' remaining work.

**The fix (§3, S1, p.525 + Figure 4).** Schedule one **iteration** at a time. An iteration is defined
in §2 (p.523) as "the run of all layers of the model" — one forward pass, one output token per
request in the batch. The scheduler loop:

1. select requests to run next
2. invoke the engine for exactly one iteration
3. receive one token per request, append each to the pool

Because control returns every iteration (tens of ms), finished requests can leave and new arrivals
can join at every boundary. The batch becomes a rolling window rather than a locked group.

**Components (Figure 4).**

| Component | Job |
|---|---|
| Endpoint | HTTPS/gRPC front door; requests in, responses out |
| Request pool | Holds every live request and its tokens, across its whole lifetime |
| Scheduler | Picks the batch each iteration, applies returned tokens to the pool |
| Engine | Abstraction over GPU tensor math; may span multiple GPUs/machines |

A request "finishes" when the model emits `<EOS>` or hits `max_tokens` — not when an iteration ends.
All requests in a batch finish their iteration simultaneously; that's what an iteration is.

**Connecting to Finding 3 (Week 2, own hardware).**

> ORCA's iteration-level scheduling is the mechanism behind Finding 3. Across a 20× concurrency
> increase, mean ITL rose only 8.826 → 12.033 ms (+36%) while output throughput rose
> 108.55 → 1621.29 tok/s (~15×). Because the batch is re-composed at every iteration boundary,
> a late-arriving request joins the batch already in flight instead of waiting for it to drain —
> the paper's own phrasing is that late arrivals "hitch a ride with the current ongoing batch"
> (§6.2, p.532). Added concurrency therefore converts almost entirely into throughput rather than
> per-token latency. §6.2 reports the same shape on their hardware: increasing `max_bs` raises
> throughput "without affecting the latency."

Source for those two numbers, so they stay traceable: [week2-notes.md](week2-notes.md) Sweep
Results table, concurrency 1 and 20 rows. Verified against the raw harness output rather than the
table — `results/week2/week2_conc1.json` gives `mean_itl_ms` 8.82642624725122 and
`output_throughput` 108.55454745698806; `results/week2/week2_conc20.json` gives 12.032717359397793
and 1621.2932922657196. Same config both runs (`--random-input-len 512 --random-output-len 256
--ignore-eos --num-prompts 100 --request-rate inf --seed 42`), only `--max-concurrency` varies, so
the comparison isolates concurrency.

One caveat to carry into Week 3: `--request-rate inf` fires all 100 prompts at once, so this sweep
never actually tested mid-flight arrivals. It measures the *throughput* half of iteration-level
scheduling (a re-composed batch keeps the GPU full) but not the *queueing-delay* half, which is the
half §6.2 is really about. Axis E's length-variance test is where late joining gets exercised.

---

## ② Selective batching

**Why it's needed (§3, C2, p.525).** Iteration-level scheduling creates the problem. Once the batch
is re-composed freely, three pairs of requests cannot be batched by the canonical mechanism:

1. both in **initiation** phase with different numbers of input tokens (`x3`: 2, `x4`: 3)
2. both in **increment** phase but at different token indices — their K/V tensors differ in shape
3. one in initiation, one in increment — initiation takes all prompt tokens at once, increment takes one

Batching requires identical operations on identically-shaped tensors. Canonical batching feeds each
Transformer layer a `[B, L, H]` tensor: **B** = batch size, **L** = tokens per request this
iteration, **H** = hidden size. A tensor is rectangular, so every request must contribute the same
L. No rectangle fits 1, 1, 2, 3. The paper calls this an **irregularly shaped** tensor — it cannot
**coalesce**.

**The key asymmetry.** Only Attention needs to know which token belongs to which request.
Linear, LayerNorm, Add and GeLU act on each token independently — a row's output depends only on
that row, so they never look sideways. Attention is the one operation where tokens look at each
other; run it on a flattened tensor and `x3`'s prompt would attend to `x2`'s tokens. On GPUs
Attention normally uses cuBLAS **batched matrix multiplication**, which demands identical shapes.

**The mechanism (§3, S2, p.526 + Figure 5).** Flatten to `[∑L, H]` — no batch dimension at all.
For the Figure 4 batch: 1 + 1 + 2 + 3 = 7 tokens, so `[7, H]`. This is **token-wise** batching
instead of request-wise: nothing padded, nothing wasted.

```
Batched (one flat tensor, request identity irrelevant):
  QKV Linear / LayerNorm / Add / GeLU / Attn Out Linear
        [ x14  x22  x31  x32  x41  x42  x43 ]        shape [7, H]
          ^ all four requests flattened together

                        | Split
                        v

NOT batched (must respect request boundaries):
  Attention, one op per request
        [x14]        [x22]        [x31 x32]    [x41 x42 x43]
         x1           x2           x3           x4
        [1, H]       [1, H]       [2, H]       [3, H]
         + K/V for    + K/V for    (initiation, no cache yet)
         x11,x12,x13  x21
         from K/V mgr from K/V mgr

                        | Merge
                        v
        [ x14  x22  x31  x32  x41  x42  x43 ]        back to [7, H]
```

Split slices Q/K/V per request (the QKV Linear has already run at that point — Figure 5 shows
`[7, 3H]` entering Split). Merge stacks the Attention outputs back to `[7, H]` so the rest of the
layer is one big matmul again. This pair sits inside **every** Transformer layer, every iteration.

**Attention K/V manager.** An increment-phase request supplies only one token (`x14`) but must
attend to `x11, x12, x13` as well. Recomputing them each iteration would be quadratic waste
(§2, p.523: fairseq-style **incremental decoding**), so keys and values are cached **separately per
request**. That separation is what makes a ragged batch safe. Critically, they're held until "the
scheduler explicitly asks to remove" them — memory is not reclaimed automatically. This couples the
mechanism to ④.

**The cost, honestly.** §6.1 / Figure 9a: at fixed batch size the ORCA engine is *slightly worse*
than FasterTransformer, because it doesn't batch Attention. The paper's defence (§1, p.522) is that
Attention has **no model parameters**, so batching it gains nothing in parameter-read reuse — and
parameter reads from GPU global memory are the real bottleneck (§7, p.533). §5 (p.529) adds that
they fuse the split Attention kernels by concatenating thread blocks, recovering launch overhead.

---

## ③ The scheduler's admission decision

§4.2 (pp.527–528) + Algorithm 1. §4.1 is skippable; §4.2 is not.

**State machine.** `INITIATION` → (selected, reserve `max_tokens` slots) → `RUNNING` → (engine
returns a token) → `INCREMENT` → selectable again, no new reservation → ... → finished, slots
released, leaves pool.

Algorithm 1 line 19 filters `state ≠ RUNNING`. `RUNNING` is only non-empty because of pipelining:
lines 9–10 let the scheduler inject up to `n_workers` batches before waiting for a return. On a
single GPU `n_workers = 1`, so the scheduler waits every iteration and that filter never fires.

**Iteration-level FCFS (p.528).** Defined as: for any pair `(xi, xj)` in the pool, if `xi` arrived
earlier than `xj`, then `xi` must have run the same or more iterations than `xj`. Note the paper's
own qualifier — a late request may still *return* earlier if it needs fewer iterations.

**`Select` (lines 17–28).** Filter out `RUNNING`, sort by arrival time, walk the list, take at most
`max_bs`, and for each `INITIATION`-phase request check the reservation fits.

Line 25 is `break`, **not `continue`** — this is the detail worth writing down. If a request in the
initiation phase can't be reserved, the scan *stops* rather than skipping ahead. Skipping would let
a later arrival run an iteration while an earlier one has run zero, violating iteration-level FCFS.
So head-of-line blocking is the deliberate price of the FCFS guarantee: one unreservable request
stalls everything behind it, including requests needing no new memory.

**`max_bs`.** Largest number of requests in a batch. Increasing it trades latency for throughput
with diminishing returns; the operator tunes it against a latency budget (§4.2 p.528, §6.2 p.532).

**Gap vs vLLM (Day 7 teardown).** ORCA has two admission constraints — `max_bs`, and the
`n_rsrv > n_slots` reservation check. vLLM's three triggers are `max_num_seqs`,
`max_num_batched_tokens`, and KV block exhaustion. Two things ORCA lacks:

- **No cap on tokens per iteration.** Nothing analogous to `max_num_batched_tokens`. `max_bs` counts
  requests, not tokens, so four requests admitted with 512-token prompts means one iteration
  processing 2,048 tokens — a latency spike the scheduler cannot see coming.
- **No preemption.** Once slots are reserved they're held to completion. vLLM's block exhaustion is
  a *dynamic* trigger that can evict or swap; ORCA's is a *static* promise made at admission.

---

## ④ KV reservation — the weak point

§4.2 p.528 (paragraph beginning "When the scheduler considers a request in the initiation phase"),
Algorithm 1 lines 23–26 (reserve) and line 14 (release), plus §4.2 p.529 first paragraph.

**Mechanism.** On first scheduling, the scheduler reserves `req.max_tokens` slots, where a **slot**
is the memory for one token's Attention key and value. If `n_rsrv + max_tokens > n_slots`, break.
Reserving the worst case up front is what makes progress provable — the stated motivation is
**deadlock avoidance**: without it the scheduler could admit a request and later find no space for
the next token's key and value, with no request able to proceed.

`n_slots` is the easy knob (p.529): given the model spec and parallelism degrees, GPU memory usage
depends mostly on `n_slots`, so the operator just takes the largest value that fits. Unlike
`max_bs`, it needs no experiments.

**Waste on own hardware.**

Slot size is fixed by the Day 1 config above — 2 (K and V) × 28 layers × 4 KV heads × 128 head_dim
× 2 B (fp16) = 57,344 B = **56 KiB/token**. AWQ quantizes the weights, not the KV cache, so this
stays fp16. Note it uses `num_key_value_heads = 4`, not `num_attention_heads = 28`; using 28 would
overestimate by 7×.

```
a slot = K/V for one token = 56 KiB                          (Plan Part 1.2)
~14 GiB KV pool = 14,336 MiB / 56 KiB = 262,144 slots

(A) FasterTransformer — reserves the MODEL's max seq len, per request
      8,192 × 56 KiB                       = 448 MiB reserved
      Week 2 actually uses 768 × 56 KiB     =  42 MiB
      wasted = 448 − 42 = 406 MiB           = 90.6% of the reservation
      capacity: 262,144 / 8,192             = ~32 concurrent requests

(B) ORCA — reserves THIS REQUEST's max_tokens (input + max_gen_tokens)
      Week 2 harness sends max_completion_tokens = 256, so
      max_tokens = 512 + 256 = 768          =  42 MiB reserved
      wasted on this workload               = ~0%
      capacity: 262,144 / 768               = ~341 concurrent requests

(C) PagedAttention — allocates 16-token blocks on demand
      768 / 16 = 48 blocks, divides evenly — no internal fragmentation
      capacity: 16,384 blocks / 48          = ~341 concurrent sequences
                                              (Plan Part 2.2 rounds to ~330)
```

**The ~10× gap is (A) vs (C), not ORCA vs PagedAttention.** This is the correction to
[week3-day-by-day.md:538](../week-wise-plan/week3-day-by-day.md#L538), which says "a slot must hold
max-model-len = 8,192 tokens." That conflates two different quantities: `--max-model-len 8192` is
*server-side* config ([serve.sh](serve.sh)), while ORCA's `max_tokens` is a *per-request client
attribute* (footnote 6, p.528). ORCA reserves the latter. So 448 MiB and 90.6% are correct
arithmetic describing **FasterTransformer** (§6.1, pp.530–531 — hence the OOM gaps in Figure 9 at
batch 8 for 13B and batch 16 for 101B), not ORCA on this workload. Phrase (A) as "~32 concurrent
requests, each holding 8,192 slots" — a slot is per-token, not per-request.

**Week 2 is the worst possible advertisement for PagedAttention**, and that is the honest thing to
record: (B) and (C) tie at ~341. Every request is the same length, 768 divides evenly into 16-token
blocks, and the client declares its exact output length up front. Under those three conditions
contiguous per-request reservation is already optimal and paging buys nothing. Harness check:
`--random-input-len 512 --random-output-len 256 --ignore-eos`
([week2-sweep.sh](week2-sweep.sh)); measured `total_input_tokens` was 54,100 / 100 prompts = 541
actual (the chat template adds ~29), so ~797 tokens/request against the 768 nominal.

**So where does ORCA actually lose?** Not on reservation size — on the four things a static promise
made at admission cannot do:

1. **Unpredictable output length.** The client must set `max_tokens` before generation. A chat client
   that cannot predict its answer sets it high "just in case," and ORCA reserves that worst case for
   the whole request lifetime. `--ignore-eos` hides this entirely: it forces exactly 256 tokens, so
   the declared and actual lengths coincide by construction. Real traffic has neither property.
2. **No prefix sharing.** Two requests with the same system prompt reserve and fill separate slots.
   PagedAttention shares blocks copy-on-write.
3. **No preemption.** Slots are held to completion; block exhaustion in vLLM is a *dynamic* trigger
   that can evict or swap.
4. **Head-of-line blocking**, from the line 25 `break` in ③ combined with reservation — one
   unreservable request stalls everything behind it.

The stronger framing, which survives the correction: **you cannot know the output length in
advance**, so any scheme committing memory at admission must either over-reserve or risk deadlock.
The number is the price of that impossibility — and this workload is rigged to never charge it.

**Three-step progression, not a two-way comparison.** FasterTransformer sizes by the model, ORCA by
the request, PagedAttention by the token. ORCA is the middle step, not the villain.

---

## ⑤ What ORCA does not solve

> ORCA schedules at iteration granularity but still reserves memory at request granularity — it
> fixed the time axis and left the memory axis for PagedAttention.

Supporting clause: Algorithm 1 has one loop body with two granularities. Line 4 re-decides batch
membership every iteration, inside `while true`; lines 23–26 reserve memory once per request
lifetime, guarded by `state = INITIATION`. That asymmetry is the seam PagedAttention cuts along.

Also open, per the paper itself (§7, p.533, "Interface between serving systems and execution
engines"): ORCA tightly couples scheduler and engine, abandoning the clean serving-layer /
engine-layer separation that Triton-style systems have. They call a general interface supporting
both techniques without losing that separation an open question.

---

## Glossary

| Term | Meaning |
|---|---|
| Iteration | One run of all layers; yields one token per request |
| Initiation / increment phase | Prefill (whole prompt, one iteration) vs decode (one token per iteration) |
| Queueing delay | Time a new request waits before any computation starts on it |
| `[B, L, H]` | Batch size × tokens per request × hidden size |
| Coalesce | Combine per-request tensors into one larger tensor |
| Irregularly shaped | Ragged — rows of different lengths, no rectangle fits |
| `[∑L, H]` | Flattened tensor, all tokens stacked, no batch dimension |
| Token-wise batching | Batch unit is the token, not the request |
| Split / Merge | Ops that separate requests for Attention and rejoin them after |
| cuBLAS batched matmul | GPU routine for many identically-shaped matmuls at once |
| Incremental decoding | Caching K/V across iterations instead of recomputing (fairseq) |
| K/V manager | Per-request cache of keys and values from earlier tokens |
| Slot | Memory for one token's Attention key and value |
| `n_slots` | Total slots allocated to the K/V manager; operator-tuned |
| `n_rsrv` | Slots currently reserved across all admitted requests |
| `max_tokens` | Per-request cap: input tokens + `max_gen_tokens` |
| `max_bs` | Max requests in one batch |
| Iteration-level FCFS | Earlier arrival has run ≥ as many iterations as any later arrival |


## Predicted capacity (before measuring)
<!-- Day 5 Step 5.5: the pre-registration table -->

## Measured capacity (from the startup log)
<!-- Day 6 Step 6.2: capacity-config.txt pasted verbatim -->

## Results — Axis A (decode-bound, 128 in / 1024 out)
## Results — Axis B (prefill-bound, 4096 in / 32 out)
## Results — Axis C (Week 2 anchor, 512 in / 256 out)
## Results — Axis D (ceiling hunt, 6144 in / 1024 out)
## Results — Axis E (length-variance test)
<!-- Day 7 -->

## THE MEMORY CEILING FINDING
<!-- Day 7 Step 7.4 -- the week's actual deliverable -->

## Why VRAM is the wrong ceiling signal
<!-- Day 7 -->

## Weekend Teardown: what triggers queuing
<!-- Day 7 -->

## Predictions: measured vs predicted
<!-- Day 7 -->

## Open questions carried into Week 4
<!-- Day 7 -->