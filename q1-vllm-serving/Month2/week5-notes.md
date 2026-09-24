# Week 5 — Paper notes: SARATHI-Serve (chunked prefills, stall-free batching)

Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024.
*Written from memory first, then checked against the paper. Corrections marked. Same method as Week 2.*
*Term note: the paper says **TBT** (time-between-tokens) where I say ITL — same thing.*

---

## Q1 — What is a generation stall? Who suffers?

Generation stalls mean high latency when a new request arrives while a decode
phase is in progress. It happens in iteration-level (continuous) batching
systems like Orca and vLLM, where the batch is re-decided every iteration and a
new request can join at the next iteration boundary.

**Who suffers:** ❗ *not* the request being prefilled — its TTFT is fine. The
ones **already decoding** suffer. When a big prefill lands in the same iteration
as running decodes, the prefill saturates the GPU, the decodes wait, and their
ITL/TBT spikes.

**Correction to my first draft — I had the prioritization backwards.**
- ❌ I wrote that these systems "prefer prefill prioritization, which increases
  throughput." That describes the **problem**, not the fix. Default vLLM/Orca
  **prioritize prefills** (admit a new prompt eagerly), and *that* is what
  *causes* the stall.
- ✅ SARATHI-Serve's contribution is to **stop** doing that: chunk the prefill
  and interleave the chunks with the decodes so no single iteration is
  prefill-dominated. It trades a little throughput for **stable ITL**, not the
  other way round.
- 🔵 Also: "new request joins the batch when an existing request is done" is not
  quite it. In iteration-level batching the batch is rebuilt **every iteration** —
  a new request joins at the next boundary, not only when an old one finishes.

*Connects to my Week 3 data:* my c8→c64 finding was "throughput flat, TTFT up
20×." That is this exact prefill/decode interference, measured on my own rig.

---

## Q2 — Token budget per iteration, and prompts longer than it

The token budget per iteration is a fixed amount of tokens that can be processed
in one iteration. Decode tokens are processed normally (~1 token each). When a
new request comes in, its prompt is chunked and fitted into what's left of the
budget; the overflow chunks wait for the next iterations.

**If the prompt is longer than the remaining budget**, it is split — this
iteration's chunk goes now, the rest rides later iterations until the whole
prompt is processed.

**Correction to my worked example — the numbers didn't add up.**
- ❌ I wrote "512 budget, 512 prompt, chunked into 4." A 512 prompt with a 512
  budget **fits in one iteration** — no chunking needed. Chunking only happens
  when the prompt is longer than what's **left** of the budget.
- ✅ The case I was reaching for, made consistent — budget 512, 32 decodes already
  running:

  ```
  budget                = 512
  decodes this step     = 32 tokens (1 each)      -> 480 left for prefill
  new prompt            = 512 tokens              -> doesn't fit in 480
  chunk it:  480 now  +  32 next iteration
  this iteration total  = 32 decode + 480 prefill = 512  (exactly the budget)
  ```
  So decodes first, prefill fills the rest, overflow waits — my instinct was
  right, the arithmetic just has to sum to the budget.

*Connects forward:* this budget is `max_num_batched_tokens` in the vLLM
scheduler (Week 6 code reading).

---

## Q3 — What does stall-free cost?

Stall-free ensures decodes never experience a generation stall due to a
co-running prefill chunk. It is **not free** — you pay in three places, plus one
deeper cost I missed at first.

**Cost 1 — higher TTFT.** When the request doesn't fit the token budget, its
whole prompt does not process in a single shot, so the first token arrives later.
✅ Confirmed: the paper's ablation shows chunked-prefills-only raises TTFT.

**Cost 2 — KV-cache re-fetch.** Each chunk re-reads all prior chunks' KV from GPU
memory during attention. For N chunks, chunk *i* reads chunks 0..*i*−1, so reads
grow ~N(N−1)/2. Computation stays the same; it's the repeated HBM reads that pile
up. Stays modest — **~25% slower at chunk=512, near-zero at 2048** — because
prefill stays **compute-bound** even at small chunks, so the extra reads hide
behind compute. ✅ Both numbers confirmed (Yi-34B, Fig 14).

**Cost 3 — tile quantization.** GPUs do matmuls in fixed-size tiles; a chunk size
that doesn't divide the tile wastes thread-block cycles, so the budget needs
hardware profiling, not a round number. ✅ Concept confirmed.
⚠ **Verify:** I wrote "chunk 257 ~32% slower than 256." Mechanism right, exact
32% unconfirmed against the paper's figure — check or soften before quoting.

**The deeper cost I first missed.** Stall-free *forces* you to set a token
budget, and that budget is a two-sided tension:
- too small → low arithmetic intensity + more KV re-reads → prefill overhead
- too large → the prefill chunk grows → decode stalls come back → higher ITL
The paper's strongest claim: the optimal budget sits at the **linear-layer
compute/memory crossover (the arithmetic-intensity knee)**, roughly
**workload-independent**. So the cost is really: stall-free turns serving into a
one-knob tuning problem, and the knob's right answer is set by the GPU, not the
traffic.

**Truck analogy.** The shipment now takes many small trips instead of one big
one. Two costs stack: the full order arrives later (**TTFT**), and every trip
re-checks the manifest of everything already delivered before adding the new box
(**KV re-read**). And the truck's size has to match the loading dock, not just
any round number (**tile quantization / the budget**).

---

## Does this predict my Week 3 anomaly? (my rule: hold the data against the paper)

**20 requests queued with KV 94% empty.** SARATHI explains half of it: queuing is
driven by the **token budget**, not by KV capacity. A burst of prompts can't all
clear the budget in one iteration, so they wait — while the KV pool is still
nearly empty. The KV ceiling (97.5%) and the queue are unrelated, which is
exactly what I measured. ✅ The paper predicts the *mechanism*; Week 6's scheduler
code is where I confirm it line by line.