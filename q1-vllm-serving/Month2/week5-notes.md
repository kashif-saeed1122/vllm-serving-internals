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

## vllm Repo code reading to find so ans

Read these 2 files 
vllm/entrypoints/openai/api_server.py: 
vllm/entrypoints/openai/chat_completion/serving.py: 

I tried to find answers for these questions 
## Q. What does the server do to the messages list before the engine ever sees it?
Before seding to engine the message are sent to a funtion called create_chat_completion method that exist nthe api_router.py file which then utlizes the Chat(OpenAIServingChat) Instance which is in serving.py file

## Q. Is the template applied to text, or to token ids?
The flow is is like this 
_create_chat_completion -> render_chat_request -> in here it says it returns convo that means it applies on the text not on the token ids.

## Q. What is the streaming function's return type? What does that tell you about how FastAPI sends it?
It returns the type of AsyncGenerator, the router wraps that generator in StreamingResponse(content=generator, media_type="text/event-stream") — Starlette iterates the generator lazily and writes each yielded string to the response as it's produced, rather than waiting for the whole thing and sending it in one shot.


### Thursday read code repo
# 1. step() does three things in order
Answer (core.py:583-613):

Schedule: self.scheduler.schedule(...) decides which requests run this step, how many tokens each gets, and which KV blocks they use. It returns a SchedulerOutput.
Execute: self.model_executor.execute_model(scheduler_output, non_block=True) runs the forward pass, and sample_tokens(grammar_output) picks the next token ids.
Update: self.scheduler.update_from_output(scheduler_output, model_output) appends the new tokens to each request, checks stop conditions, and builds the EngineCoreOutputs that go back to the frontend.


# 2. Request states and transitions

The states (request.py:351-367): WAITING, WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, WAITING_FOR_REMOTE_KVS, WAITING_FOR_STREAMING_REQ, RUNNING, PREEMPTED, then six FINISHED_* states.

The order matters: is_finished() is status > PREEMPTED, which is why the comment says anything after PREEMPTED counts as finished.

 new request
   │  (structured output?)
   ├──────────► WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
   │                 │ grammar compiled        │ compile error
   ▼                 ▼                         ▼
 WAITING ◄───────────┘                   FINISHED_ERROR
   │  ▲
   │  └──────── WAITING_FOR_REMOTE_KVS ◄── (from WAITING or PREEMPTED,
   │                 │                      async KV load started)
   │                 └──► back to WAITING, or PREEMPTED if it was preempted before
   ▼
 RUNNING ◄──────────── PREEMPTED
   │   └── out of KV blocks ──► PREEMPTED
   │
   ├─ EOS / stop token ───────► FINISHED_STOPPED
   ├─ max_tokens / max len ───► FINISHED_LENGTH_CAPPED
   ├─ repetition detected ────► FINISHED_REPETITION
   ├─ grammar rejects token ──► FINISHED_ERROR
   └─ (resumable session) ────► WAITING_FOR_STREAMING_REQ ──► WAITING (next chunk)
                                                          └─► FINISHED_ABORTED (session ends)

 any unfinished state ──► FINISHED_ABORTED   (client abort, shutdown)

# 3. What num_computed_tokens counts, and when it goes down
Answer: it is how many tokens at the start of the request's sequence (prompt plus output) already have their KV cache entries, so they need no forward pass. Each step the scheduler asks for the rest: num_new_tokens = num_tokens - num_computed_tokens (scheduler.py:932).

It goes up in two places:

At admission: it is set to the prefix-cache hit, local plus external (scheduler.py:881-883, scheduler.py:1136).
After scheduling: += num_scheduled_token, before the GPU has finished (scheduler.py:1379-1393). The comment there gives the reason: a long prefill can be scheduled again in the very next step.
So it is optimistic: it means "scheduled to be computed", not "confirmed computed".

It goes down in four cases:

Preemption (the answer the hint wants): the request's blocks are freed, so the counter is reset to 0 (scheduler.py:1352-1356). On resume, the prefix-cache lookup runs again because the counter is 0 (scheduler.py:811), so some blocks may be recovered without recomputing.
Rejected speculative tokens: -= num_rejected, because the count was advanced for draft tokens that turned out wrong (scheduler.py:1844-1846).
Invalid or failed KV blocks: truncated to the first bad block, idx * block_size (scheduler.py:2908).
Full remote hit on the whole prompt: set to num_tokens - 1, so the last token is recomputed to produce logits to sample from (scheduler.py:2761-2762).

# 4. Why detokenisation is incremental, and why a token can produce no text
Where it happens: not in the engine core. The core sends only token ids. The frontend's OutputProcessor.process_outputs calls detokenizer.update(...) (output_processor.py:669), which calls decode_next once per token (detokenizer.py:118-120).

Why incremental:

Streaming needs the new text after every token.
Decoding the whole output each step would cost more and more as the output grows.
Decoding each token alone and joining the strings is wrong, because the text of a token depends on its neighbours (leading spaces, and characters split across tokens).
So the detokenizer keeps a small window. In the slow path (detokenizer_utils.py:241-268) it decodes the window without the new token (prefix_text) and with it (new_text), and emits only the difference. The fast path does the same job with the tokenizers library's DecodeStream.

Why one token id can give no text:

Incomplete UTF-8 character (the main one). In byte-level BPE a token is a run of bytes, not characters. Urdu "ا" is two bytes (D8 A7) and 😀 is four (F0 9F 98 80). If the model emits only the first part, decoding gives "�". The code detects this and returns "" without moving its offsets (detokenizer_utils.py:260-265), so the next token is decoded together with the held one. The fast path returns None, turned into "" at detokenizer.py:222.
Special tokens such as EOS, when skip_special_tokens is on.
Stop strings. The last max(len(stop)) - 1 characters are held back in case they are the start of a stop string (detokenizer.py:85-90, detokenizer.py:149-165). The text exists but is not released yet.
Out-of-vocabulary id: decoded as "" (detokenizer_utils.py:218-231).