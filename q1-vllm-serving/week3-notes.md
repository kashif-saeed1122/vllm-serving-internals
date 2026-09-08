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