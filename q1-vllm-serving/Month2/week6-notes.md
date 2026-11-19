# Week 6 — Paper notes: SGLang / RadixAttention

Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024.
*Read only RadixAttention (§3) and its evaluation; skipped the frontend language.
Answered from memory first, then checked against the paper and vLLM v0.28.0 source. Corrections marked.*

---

## Q1 — What data structure does SGLang use to find shared prefixes, and what does vLLM use instead?

**SGLang: a radix tree** (compressed trie) over token IDs. Edges hold token
runs of any length; each node points to the KV slots for its tokens. A finished
request's KV stays in the tree instead of being freed. A new request walks from
the root, matches the longest cached prefix, and computes only the rest.
Shared prefixes are **inner nodes**; each request's unique tail is a **leaf**.

**vLLM: a hash table.** Each *full* 16-token block is hashed from
(parent block hash, its tokens, extra keys like LoRA / `cache_salt`)
(`hash_block_tokens`). A dict maps hash → physical block
(`cached_block_hash_to_block`). Lookup walks the block hashes in order and
stops at the first miss. Chaining the parent hash makes it a prefix tree
flattened into a hash table, at 16-token steps.

**Corrections to my first draft:**
- ❌ I wrote that shared tokens are reused via a single leaf. It's backwards:
  shared = inner node, unique = leaf.
- ❌ PagedAttention is vLLM's **memory layout**, not its prefix lookup.

---

## Q2 — Which design can reuse part of a block? Connect to "16 tokens, not 29".

**Only SGLang.** Page size = 1 token, so a match can stop at any token. vLLM
hashes only full blocks, so a cache hit is always rounded down to a multiple
of 16.

**My data:** Qwen2.5's template puts **24** identical tokens before the user
text. The other ~5 of the ~29 template tokens come *after* the user text, so
they are not a prefix.
- Block 0 = 16 template tokens → hit.
- Block 1 = 8 template + 8 user tokens → different hash → miss, so 8 shared tokens are recomputed.
- Measured **16 tokens/request** (1,600 hits / 100 prompts = 3.0% of 541).
- SGLang would reuse all **24** (~4.4%). Neither design would get 29.

**Corrections to my first draft:**
- ❌ I never stated which design wins. That was the actual question.
- 🔵 "29" was the wrong number to compare against: the shared *prefix* is 24.

---

## Q3 — What eviction policy does RadixAttention use, and what does it evict first: leaves or inner nodes?

**LRU, leaves first.** The paper's wording is "evicts the least recently used leaf first" (§3).
- A child's KV is useless without its parent's KV, because attention needs the whole prefix.
- So evicting a leaf never breaks another cached path, and shared ancestors
  survive while any descendant is cached.
- Once its children are gone, a parent becomes a leaf and can be evicted, so eviction runs bottom-up.

**Reference counter:** per node, it counts the running requests using that node. A node can
be evicted only when the count is 0. The count protects nodes in use; it does not rank them.

**vLLM analog:** blocks with `ref_cnt` = 0 go into an LRU free queue. A finished
request's blocks are freed in reverse, so tail blocks are evicted first:
leaf-first, at block level.

**Corrections to my first draft:**
- ❌ "Least used" → it's least **recently** used.
- ❌ I mixed in "prompts with larger shared prefixes". That's the scheduling trick (Q4), not eviction.
- ❌ I didn't answer leaves vs inner nodes.

---

## Q4 — What scheduling trick does the paper use to raise the cache hit rate?

*TODO — my answer pending.*

---

# Tuesday — Code: where the budget comes from

*Read `vllm/config/scheduler.py`, `vllm/engine/arg_utils.py` (`get_batch_defaults`), and the top of
`Scheduler.schedule()` in `vllm/v1/core/sched/scheduler.py`, all from vLLM v0.28.0.
My setup: `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192` with no budget flags, on an RTX PRO 4000 Blackwell (24 GB).
Everything here is a **prediction from reading the code**. The pod decides on Saturday. Corrections marked.*

**Short version:** Every scheduler step, vLLM has a purse of **2048 tokens** to spend and **256 seats**
for running requests. Both numbers come from a hardware table in `arg_utils.py`, not from the
config class. My 24 GB card falls into the untuned `else` branch of that table. A burst of
20 × 541 = 10,820 tokens is more than five purses, so prefilling it takes **6 steps**. The purse runs
out long before the seats do.

---

## The fields (`config/scheduler.py`)

| Field | Class default | What it controls |
|---|---|---|
| `max_num_batched_tokens` | 2048 | Tokens processed per step (the purse). The comment says "mainly for convenience when testing" |
| `max_num_seqs` | 128 | Max **running** requests (the seats) |
| `max_num_scheduled_tokens` | `None`, which means "same as `max_num_batched_tokens`" | Tokens the scheduler may issue per step. Only smaller when spec decoding is on |
| `watermark` | 0.0 | Fraction of KV blocks to keep free when admitting waiting or preempted requests. 0.0 = off |
| `policy` | `"fcfs"` | Order of arrival. `"priority"` = lowest priority value first, ties broken by arrival time |

**Where `token_budget` is set:** the first lines of `schedule()`:
`token_budget = self.max_num_scheduled_tokens` (`scheduler.py:496`). The step spends it in two
loops, both guarded by `token_budget > 0`:
1. **running** requests first (decodes and in-progress prefill chunks),
2. **waiting** requests second (new prompts).

So decode tokens are always paid for before any new prefill.

---

## Q1 — Which default wins for `vllm serve` on a 24 GB card?

**The `arg_utils.py` table.** The full order of precedence:

1. **My CLI flag.** A default is used only when the arg is still `None` (`EngineArgs.max_num_seqs: int | None = None`).
2. **The table**, keyed by GPU size and launch mode (usage context): `default_max_num_batched_tokens.get(usage_context, ...)`
   (`arg_utils.py:2777`). `vllm serve` passes `UsageContext.OPENAI_API_SERVER` (`cli/serve.py:296`).
3. **The class constant** (`SchedulerConfig.DEFAULT_*`). This is the fallback in that `.get()`, used only when the launch mode isn't in the table.

After that, three adjustments can change the number. None of them applies to my setup:
- `--performance-mode throughput` doubles both values (the default mode is `balanced`).
- With chunked prefill off, the budget is raised to `max(max_model_len, budget)`.
- The budget is capped at `max_num_seqs × max_model_len`.

A default in vLLM isn't one number. It depends on the hardware and the launch mode, so the same model served
two different ways gets two different budgets.

**Predicted: `max_num_batched_tokens = 2048`, `max_num_seqs = 256`.**

**The trap:** 2048 is also the class constant, so the budget alone can't show which path ran.
**`max_num_seqs` tells them apart:** 256 means the table won. 128 means the class constant won.

**How to check on the pod:**
- **Budget:** the startup log prints `Chunked prefill is enabled with max_num_batched_tokens=2048.`
- **Seats:** the startup log doesn't print it. Print it from the config instead. *(I haven't run this yet. Try it on the pod.)*
  ```bash
  python -c "from vllm.engine.arg_utils import EngineArgs; from vllm.usage.usage_lib import UsageContext; \
  c = EngineArgs(model='Qwen/Qwen2.5-7B-Instruct-AWQ', quantization='awq', max_model_len=8192) \
      .create_engine_config(UsageContext.OPENAI_API_SERVER).scheduler_config; \
  print(c.max_num_batched_tokens, c.max_num_seqs)"
  ```


## Q2 — What does the device-memory check decide? Where does my card land?

**It sorts the GPU into one of three buckets and returns both defaults: the purse and the seats.**
It looks at total memory and the device name. It doesn't measure the card's speed.

| Bucket (`arg_utils.py:2582–2612`) | Budget (`LLM` / server) | Seats |
|---|---|---|
| ≥ 160 GiB (B200/B300) | 16384 / 16384 | 1024 |
| ≥ 70 GiB **and not an A100** (H100/H200) | 16384 / 8192 | 1024 |
| everything else: `# TODO(woosuk): Tune the default values for other hardware.` | 8192 / **2048** | **256** |

**24 GB is under 70 GiB, so my card lands in the `else` branch.** Nothing in that branch was tuned for my card.
A T4, a 4090, my card, and even an **A100-80GB** all get the same 2048 / 256.
The A100 is moved down on purpose: a big budget made its throughput worse (PR #17885). So bigger batches
aren't always better.

My defaults aren't tuned for me. They're the fallback, which means there's room to beat them.

**Corrections to my first draft:**
- ❌ I wrote that it decides how much prefill work the card can handle per step, "nothing more". It also
  picks **`max_num_seqs`** (256 vs 1024). It also checks the **device name** (the A100 exclusion), not just memory.
- 🔵 It reads **total** memory, measured in **GiB**, not free memory. So `--gpu-memory-utilization` can't move
  my card into another bucket.
- 🔵 The prediction holds even if the check fails. If the device query throws an error, the code sets `device_memory = 0`.
  That also lands in `else`, so I get the same numbers either way.

---

## Q3 — Is `max_num_seqs` a cap on running requests or on waiting requests?

**Running.** It's the number of seats in the room, not the length of the queue outside.

The scheduler renames it `max_num_running_reqs` (`scheduler.py:110`). It is checked in one place that matters:
inside the **waiting** loop (`scheduler.py:757`): `if num_running >= self.max_num_running_reqs: break`.
So it controls admission. This setting puts no limit on the waiting queue: 1,000 requests can wait
while 256 run.

`num_running` also counts paused streaming sessions (`num_waiting_for_streaming_input`). Those are no longer
in `running`, but each one still holds a model-runner slot. A seat is a slot, not "a request computing right now".

**Additions:**
- 🔵 The running loop never checks the cap. It only matters when a request is admitted. That includes preempted requests:
  they go back to `waiting` and have to be admitted again. An
  `assert len(self.running) <= self.max_num_running_reqs` at the end of `schedule()` backs this up.
- 🔵 Startup **requires** `max_num_batched_tokens >= max_num_seqs` (`config/scheduler.py:263` raises a
  `ValueError` otherwise). Every running request needs at least 1 token per step to decode. With more seats than
  tokens, the seats could never all be served in one step. Here 2048 ≥ 256 ✓.

---

## Q4 — The burst: 20 requests of 541 tokens arrive at once

```
burst     = 20 × 541                = 10,820 tokens
budget    = 2,048 tokens / step
steps     ≥ 10,820 / 2,048          = 5.28  → at least 6
per step  = 2,048 / 541             = 3.78  → 3 whole prompts (1,623) + a 425-token chunk of the 4th
slack     = 6 × 2,048 − 10,820      = 1,468 tokens unused across the 6 steps
decodes   = 0+3+7+11+15+18          = 54 tokens   (each finished prompt decodes 1 token/step, paid for first)
54 < 1,468                          → exactly 6 steps, not 7
seats     = 20 of 256               → the seat limit is never reached
```

**First token by step** (from a small simulation: first-come first-served, running requests scheduled first, chunked prefill on, 1 decode token per step):

| Step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Requests getting their first token | 3 | 4 | 4 | 4 | 3 | 2 |

So TTFT should look like a **staircase with 6 levels**. The last two requests wait through all six steps
before their first token.

**The purse is the limit, not the seats.** This is where intuition goes wrong: "too many requests at once" sounds like a concurrency limit,
but here it is a token limit.


  A bigger budget wins on every measure only if the GPU is *underused* at 2048. **That's the real question for the pod.**

---

## Pre-registration for Saturday

| # | Prediction | How I check it | What would prove me wrong |
|---|---|---|---|
| 1 | `max_num_batched_tokens = 2048` | startup log line | any other number |
| 2 | `max_num_seqs = 256` (from the table, not the class) | the config print above | 128: the class constant won, so my reading of the launch-mode lookup is wrong |
| 3 | The burst prefills in 6 steps. Sorted TTFTs form a staircase (3-4-4-4-3-2) | per-request TTFT, sorted | clearly fewer than 6 levels at a 2048 budget |
| 4 | `--max-num-seqs 512` changes nothing (control) | same burst, 3 runs per setting | TTFT moves by more than the run-to-run noise |
| 5 | `--max-num-batched-tokens 8192` gives 2 levels instead of 6, the last TTFT about the same, first and mean TTFT **higher** | same burst | the last TTFT drops a lot: the GPU was underused at 2048, so my "already fully busy" assumption is wrong |

*Caveat: "at once" means the client sends all 20 together, but they reach the engine a few ms apart. If step 1
starts before all 20 have arrived, the staircase shifts.*

# Week 6 · Wednesday: the running loop and preemption

Files: `vllm/v1/core/sched/scheduler.py` (`schedule()` L691, running loop L767–972, `_preempt_request` L1820).
Line numbers can drift between versions.

## One sentence
Running requests are served first. If one can't get KV blocks, the scheduler kicks out another running
request (the victim), frees its blocks, and retries. The victim recomputes from zero later.

## Two limits (do not mix them up)
| | token_budget | KV blocks |
|---|---|---|
| What | compute allowed this step | memory (fixed pool, sized from VRAM at startup) |
| Resets | every step | held until the request finishes or is preempted |
| When short | request just gets **fewer tokens** (`min`, L820) | `allocate_slots` returns `None` -> **victim** |

Only memory failure causes preemption. A spent budget never does (the loop just ends, L769).

## Why running before waiting (Q1)
- Running requests hold blocks and computed progress. Waiting ones hold nothing.
- Decode is cheap (about 1 token) and users watch the stream. Waiting only delays the first token.
- The waiting pass gets the leftover budget.
- Waiting-first would thrash: admit, prefill, evict, recompute.
- L1030: if anything was preempted this step, the waiting pass is skipped.
- A waiting request never preempts. On `None` it just `break`s (L1504).

## The call that decides (Q2)
`self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)` at L888.
- Returns the new blocks, or `None` if blocks needed > free blocks (`kv_cache_manager.py` L604–608).
- Needs a new block only when the next token crosses a block boundary.
  `needed = ceil((computed + new) / block_size) - blocks_held`
- Watermark headroom applies only to WAITING/PREEMPTED requests, not running ones.

## Preemption path: ask, pick, remove, reset, retry
1. **Ask:** `allocate_slots` returns `None` (blocks needed > free).
2. **Pick victim (L906–912):**
   - FCFS: `running[-1]`, the newest admitted.
   - PRIORITY: `max(running, key=(priority, arrival_time))`. Largest number = least important.
3. **Remove from `self.running`:** done by the loop, not by `_preempt_request`.
   - FCFS: `pop()` (L948).
   - PRIORITY: `index` + `del`, then `if victim_index < req_index: req_index -= 1` (L919–922).
   - PRIORITY only: if the victim was already scheduled this step, refund it (L924–946).
4. **Reset (`_preempt_request`, L1820):**
   - free KV blocks (maybe deferred if a GPU step may still write them)
   - free encoder cache
   - `status = PREEMPTED`
   - `num_computed_tokens = 0` (keeps prompt and output tokens)
   - `num_preemptions += 1`
   - `waiting.prepend_request`: front under FCFS; re-inserted by (priority, arrival) under PRIORITY
5. **Retry:** back to `while True` (L887) with the same request.
   - Success: `break` (L896), schedule it.
   - Give up: L902 pending frees, L916 victim can't be freed, L958 victim is the request itself.
   - Then FCFS outer `break` (L972); PRIORITY self-preempt `continue` (L970).

## Q3: why evict a running request at all
Memory is oversubscribed because admission is optimistic (only the next chunk is checked).
Without eviction: every request holds blocks and waits for one more, so nobody finishes. Deadlock.
Evicting one victim breaks the circle.

## Q4: what the victim loses
| | before | after |
|---|---|---|
| blocks | held | returned to pool |
| status | RUNNING | PREEMPTED |
| num_computed_tokens | e.g. 12 | **0** |
| num_tokens | 13 | 13 (unchanged) |
| queue | running | waiting |

V1 does **recompute**, not swap: nothing is copied to CPU. Freed blocks stay in the prefix cache
until reused, so a re-admitted victim may recompute less than all of it.

## Q5: refund (PRIORITY only)
If the victim was already planned this step, undo the plan. Each spend has a mirror:
| spent at | undone at |
|---|---|
| L975 add to scheduled_running_reqs | L926 remove |
| L979 num_scheduled_tokens[id] | L927 pop -> `restored` |
| L980 token_budget -= n | L928 += restored |
| L981 input_budget -= n + draft_slots | L929 += restored + draft_slots |
| L978 req_to_new_blocks[id] | L930 pop |
| L999 spec tokens, L1007 encoder inputs | L931, L932, L942 |
- Refund reads `restored` from the dict: that is the exact amount spent, already capped.
- `req_to_new_blocks` entry must go: it tells the worker which blocks A owns; they were just freed.
- `num_computed_tokens` is not rolled back here because it is only advanced later in
  `_update_after_schedule`.
- FCFS never refunds: the victim is `running[-1]`, which is never an already-scheduled request.
- `drafts_needed` and `prefill_scheduled` are never undone.

## FCFS vs PRIORITY
One policy per engine, set at startup (`policy = "fcfs"` is the default). In FCFS, `request.priority` is ignored.
Lower priority number = more important.

## Loops and breaks in the running pass
- OUTER `while req_index < len(running) and token_budget > 0` (L769).
  - `break`: L772, L972.
  - `continue`: skips (each does `req_index += 1` first); L970 is the PRIORITY self-preempt.
- INNER `while True` (L887). Its four breaks only escape the retry loop:
  L896 success; L902, L916, L958 failure.
- After the inner loop, L960 `if new_blocks is None` decides continue (L970) or break (L972).
- The waiting pass has ONE loop (L1059), nothing nested.
- A `break` or `continue` belongs to the nearest enclosing `while` or `for`.

## A vs B exits (running loop)
- A: this request gets nothing, try the next (`req_index += 1; continue`).
- B: stop the whole running loop; nobody after is considered.
- Both leave the request in `running` with its blocks. Neither is preemption.
- Exception: L958 under FCFS. The request preempted itself, then B-exit at L972.

## Traps
- Budget low means fewer tokens, not a victim.
- `_preempt_request` does not remove from `running`. The caller does.
- New requests never evict running ones.
- Prefix cache is not separate memory. It is the same block pool, reused by hash.

## Self-test (answer with the file closed)
1. What does `allocate_slots` return on failure?
2. Who is the FCFS victim? The PRIORITY victim?
3. Which of blocks, status, `num_computed_tokens` change on the victim?
4. Why does the PRIORITY branch decrement `req_index`?
5. When is the refund block skipped?
