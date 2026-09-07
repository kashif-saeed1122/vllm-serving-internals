# Week 3 — Full Execution Plan
### Expand the sweep, find the memory ceiling

This is the only file for Week 3. Nothing else has been added to the repo — you
build each script yourself, in order, from Part 4. Every flag and every line is
explained where it first appears, so the tooling is understood rather than
pasted.

---

## Contents

- **[Part 0 — Close out Week 2 first](#part-0)** ← *answers "what am I missing before Week 3?"*
- **[Part 1 — Concepts you need before touching the pod](#part-1)**
- **[Part 2 — Capacity arithmetic: predict the ceiling before measuring it](#part-2)**
- **[Part 3 — Experiment design](#part-3)**
- **[Part 4 — Build the tooling, step by step](#part-4)**
- **[Part 5 — Pod session runbook](#part-5)**
- **[Part 6 — Analysis](#part-6)**
- **[Part 7 — Weekend teardown](#part-7)**
- **[Part 8 — Deliverables, risks, and the one-line test](#part-8)**

**Locked serve config — do NOT change this week:** `--max-model-len 8192`,
`--gpu-memory-utilization 0.90`, `--quantization awq`. Varying
`gpu_memory_utilization` is explicitly **Week 10's** job. Changing it now would
make every Week 3 number non-comparable to the Week 2 baseline, and would
destroy the "ENVIRONMENT.md unchanged since Week 1 (locked stack)" requirement
from the end-of-quarter checklist.

---

<a name="part-0"></a>
# Part 0 — Close out Week 2 first

You asked what you are missing. Here it is, checked against the Q1 plan's own
Week 2 "To cover" and "To write" lists, item by item. **Four things are
genuinely incomplete.** Three of them cost zero GPU time and should be finished
before Week 3 starts; the fourth folds into the Week 3 pod session.

### Week 2 scorecard

| Q1 plan required | Status | Evidence |
|---|---|---|
| Weekly paper: PagedAttention, one page | 🟢 **Done** | Was incomplete (*"reading in progress"* + a placeholder listing three missing pieces). Now finished — see [W2-A](#w2-a). |
| Learn `vllm bench serve` — what it measures | 🟢 Done | Glossary section, TTFT/ITL/TPOT/E2EL definitions with the E2EL formula |
| Sweep across a real range (1, 2, 5, 10, 20) | 🟢 Done | Results table, 5 committed JSONs |
| Capture TTFT | 🟢 Done | mean/p50/p95/p99 all present |
| Capture ITL | 🟢 Done | mean ITL + TPOT |
| Capture throughput | 🟢 Done | req/s and output tok/s |
| Capture p50/p95/p99 | 🟡 **Weak** | Present, but from 100 requests per config — so "p99" is literally the 99th of 100 samples, i.e. the maximum. See [Part 1.6](#p16) |
| **First attempt at the memory ceiling** | 🔴 **NOT DONE** | Your own note: *"KV cache usage metric name found in `/metrics`: NOT CAPTURED this session — pane 3's step was missed/incomplete"* |
| Weekend teardown: how `vllm bench serve` times TTFT/ITL | 🟡 **PARTIAL** | You correctly found `calculate_metrics()` is an *aggregator*, not the stamping site — and wrote *"in a different function, the async request sender — not yet read"*. The question asked was where TTFT is **stamped**. That is still unread. |
| Benchmark script in repo | 🟢 Done | `week2-sweep.sh` |
| Results table | 🟢 Done | 5-row table with findings |
| SECURITY-NOTES entry: does concurrency change prefix-cache signal visibility? | 🔴 **NOT DONE** | Still an unchecked box in `week2-notes.md`; `Security-notes.md` has only the Week 1 entry |

**Verdict:** Week 2's *deliverable* (script + results table) is met. What is
missing is the paper note, the teardown's actual answer, the SECURITY-NOTES
line, and the memory-ceiling first attempt.

---

### W2-A — PagedAttention notes ✅ DONE

**Written into `week2-notes.md`** — the `*(reading in progress)*` marker and the
`*(more to be added...)*` placeholder are gone, replaced by four subsections:

1. **Block table mechanics** — block_size 16, the logical→physical indirection
   array, an ASCII diagram of the indirection layer, and the fragmentation
   arithmetic (≤ 0.2% internal waste for a 7,168-token sequence vs 12.5% for a
   contiguous 8,192-token buffer; uniform block size means no external
   fragmentation at all).
2. **Memory sharing across sequences** — shared physical blocks, reference
   counting, copy-on-write, plus a diagram of two sequences sharing a prompt.
3. **Tie-back to the Week 1 number** — the 34.4% → 53.9% → 58.6% hit-rate trend
   explained by reference-counted shared blocks, and why that makes the
   `/metrics` gauge a confirmed side-channel shape rather than a theoretical one.
4. **What PagedAttention does not solve** — sharing removes memory *waste* but
   does not decide *which requests run*; admission and preemption are the
   scheduler's job, and a full block pool still forces eviction.

Point 4 is the load-bearing one. It is the bridge to Week 3's paper (Orca:
iteration-level scheduling, and a KV reservation scheme that is the very problem
PagedAttention was written to fix) and to Month 3, where a full block pool gets
triggered on purpose.

### W2-B — Finish the teardown: find where TTFT is actually stamped (local, ~60 min, free)

Your Week 2 note is honest and correct: `calculate_metrics()` reads
`outputs[i].ttft` — values *already recorded*. You wrote that the stamping site
is "in a different function, the async request sender — not yet read."

**Why this matters for Week 3, not just for tidiness:** every number in your
Week 3 results table is a client-side measurement. If you do not know the exact
two clock reads that TTFT is the difference of, you cannot say what your TTFT
*excludes* — network round-trip, HTTP overhead, tokenizer time, the
`asyncio` event loop's own scheduling delay. Week 3 compares a prefill-bound
axis against a decode-bound axis and attributes differences to the engine. That
attribution is only valid if you know what the client-side timer wraps.

**How to do it.** First, clone the source. This costs nothing and the Q1 plan
schedules it for Week 5 — pulling it forward one week is free and unblocks both
this and Part 7:

```bash
git clone --depth 1 --branch v0.28.0 \
  https://github.com/vllm-project/vllm.git ~/src/vllm-0.28.0
cd ~/src/vllm-0.28.0
```

Then find the stamping site. You already know the file:

```bash
# 1. Find the async sender for the backend you actually used (openai-chat).
grep -rn "async def async_request_openai_chat" vllm/benchmarks/

# 2. Find every place ttft is assigned, not read.
grep -rn "ttft" vllm/benchmarks/lib/endpoint_request_func.py

# 3. Find the clock. This is the key line.
grep -rn "perf_counter\|monotonic\|time.time" vllm/benchmarks/lib/endpoint_request_func.py
```

*(File layout shifts between versions — if `endpoint_request_func.py` is not
there, `grep -rln "ttft" vllm/benchmarks/` will find it.)*

**Answer these five questions in `week2-notes.md`, replacing your placeholder:**

1. Which clock is used? (`time.perf_counter()` is monotonic and high-resolution;
   `time.time()` is wall-clock and can jump. Which one, and does it matter?)
2. Where is `st` / the start timestamp taken — **before** or **after** the HTTP
   request is dispatched? Everything after that point is inside your TTFT.
3. What triggers the "first token has arrived" stamp? Look for the streaming
   loop: it reads SSE chunks, and there is usually a guard like
   `if ttft == 0.0:` that fires once. **Note whether an empty first chunk or a
   role-only delta chunk can trip it** — if the OpenAI-compatible stream sends a
   `{"role":"assistant"}` delta before any content, TTFT may be stamping on a
   chunk containing zero text.
4. How is `itl` appended — one entry per SSE chunk, or one per token? If a chunk
   can carry more than one token, your "inter-token latency" is really
   "inter-chunk latency". Write down which it is.
5. What does `latency` (used for E2EL) span?

**Write:** replace the "not yet read" sentence in `week2-notes.md` with the
answer. Keep your existing `calculate_metrics()` finding — the correction you
already made (aggregator, not stamping site) is good work; you are now
completing it.

> **Bound your depth:** if question 3 or 4 does not resolve in one sitting,
> write down exactly what is unclear and move on. That is the plan's own rule.

---

### W2-C — Write the SECURITY-NOTES line you owe (local, ~15 min, free)

Week 2's "To write" list included: *"log anything noticed about whether
concurrency level changes the visibility of the prefix-cache signal —
opportunistic, not a dedicated task."*

You have no prefix-cache data from the Week 2 sweep, because you never captured
`/metrics` during it. So the honest entry is a **negative result plus a
method note**, which is a legitimate log entry:

```markdown
- [Week 2] [Prefix caching — NOT measured, method gap identified] — The Week 2
  concurrency sweep (1/2/5/10/20, 512-in/256-out, 100 prompts each) produced no
  prefix-cache data, because `/metrics` was never sampled during the runs. The
  sweep used `vllm bench serve --dataset-name random`, which generates unique
  random token IDs per prompt, so cross-request prefix sharing should be near
  zero except for the shared chat-template wrapper (~20-30 tokens) prepended to
  every request by `--backend openai-chat`. That wrapper alone is a shared
  prefix, so a small non-zero hit rate is expected even with fully random
  prompts. Untested. Method gap for Week 3: the engine log's 10-second rolling
  window cannot resolve per-run cache state, so continuous `/metrics` sampling
  is required. Still observation-only.
```

Also rename the file to match the plan and the end-of-quarter checklist:

```bash
git mv Security-notes.md SECURITY-NOTES.md
```

*(Case-only renames need `git mv` on Windows — a plain rename will not be seen
by git.)*

---

### W2-D — The memory ceiling: fold into Week 3 (pod time)

This is the one incomplete Week 2 item that genuinely needs a GPU. Do **not**
rent a pod just for it. It is designed into Week 3's axis C and axis D, and
Part 5 Step 4 is written specifically to close it.

The `week2-notes.md` "Carried forward to Week 3 pod session" block already says
this. Part 5 is the executable version of that block.

---

### W2-E — Repo housekeeping (local, ~20 min, free)

These are not conceptual, but they will bite at the Week 4 Month-1 checkpoint,
whose requirement is *"independently reproducible by a stranger."*

**1. Layout is split across two directories.** Right now:

```
D:\dev\                          <- git root
├─ Readme.md
├─ week2-sweep.log               <- stray, at root
├─ results/week2/*.json          <- data at root
└─ q1-vllm-serving/              <- everything else
   ├─ ENVIRONMENT.md
   ├─ week1-notes.md, week2-notes.md
   ├─ serve.sh, week2-sweep.sh
   └─ Security-notes.md
```

The split happened because `week2-sweep.sh` writes to a *relative* path
(`results/week2`) and you ran it from the repo root. Decide now, before Week 3
doubles the data volume. **Recommended:** consolidate under `q1-vllm-serving/`
so the project is one self-contained directory:

```bash
cd /d/dev
git mv results q1-vllm-serving/results
git mv week2-sweep.log q1-vllm-serving/results/week2/
```

Then in Week 3, always `cd q1-vllm-serving` before running the sweep, so
`results/week3` lands in the right place. Part 5 Step 1 assumes this.

**2. `Readme.md` status line is two weeks stale.** It currently reads:

> 🚧 Week 1 — environment verified, benchmarking harness not yet built.

The harness exists and has produced a full sweep. Update it, and add a
`## Results` pointer to `week2-notes.md`. The README is the first thing a
stranger reads; a stale status line makes the whole repo look abandoned.

**3. `week2-notes.md` is uncommitted** (`M` in `git status`). Commit it before
starting Week 3 so the Week 3 diff is clean and reviewable.

**4. Naming inconsistency.** The Q1 plan says `NOTES.md` and `SECURITY-NOTES.md`;
you have `week1-notes.md` / `week2-notes.md` / `Security-notes.md`. Per-week
files are arguably *better* than one growing `NOTES.md` — but decide and be
consistent. **Recommended:** keep per-week note files (they are easier to read
cold), uppercase the two standing documents (`SECURITY-NOTES.md`,
`ENVIRONMENT.md` ✓), and add a `NOTES.md` that is just an index linking to each
week's file. One line in the README explaining the convention removes the
ambiguity for a stranger.

**5. `.gitignore` has a UTF-8 BOM** on line 1 before `.env`. Harmless, but if
you ever add a pattern on the first line it will silently not match. Worth
rewriting the file cleanly while you are in there.

---

### Part 0 checklist — do all of this before the Week 3 pod session

Zero GPU cost, roughly one evening:

- [x] **W2-A** PagedAttention notes finished: block table mechanics, memory
      sharing + reference counting + copy-on-write, the tie-back to the Week 1
      hit-rate number, and one sentence on what it does not solve
- [ ] **W2-B** vLLM source cloned at `v0.28.0`; TTFT stamping site found; the
      five questions answered in `week2-notes.md`; the "not yet read"
      placeholder replaced
- [ ] **W2-C** SECURITY-NOTES entry written; file renamed with `git mv`
- [ ] **W2-E** layout consolidated, README status updated, `week2-notes.md`
      committed, naming convention decided
- [ ] **Flag verification from source** — see [Part 4.7](#p47). Confirm
      `--save-detailed`, `--temperature`, `--random-range-ratio`, and
      `--random-prefix-len` exist in *your* pinned version, by grepping the
      clone. Free, and it removes the main way a paid pod session gets wasted.
- [ ] **Pod readiness** — confirm the corrected custom RunPod template from
      Week 1 still exists, and that the 25 GB network volume `q1-serving` still
      has the model weights cached (otherwise budget an extra ~10 min of paid
      time for the download)
- [ ] **Predictions pre-registered** — [Part 8.2](#p82) filled in *before* boot

Only then start Part 1.

---

<a name="part-1"></a>
# Part 1 — Concepts you need before touching the pod

Week 3's deliverable is a *number with an explanation*. These seven concepts are
what the explanation is made of. Read this part before Part 2, because Part 2's
arithmetic assumes all of it.

### 1.1 — Why VRAM is the wrong signal for the memory ceiling

This is the most important idea in Week 3, and it resolves an apparent
contradiction sitting in your Week 1 notes right now.

Week 1 recorded two things that look incompatible:

- *"KV cache pre-allocation dominates idle VRAM (~87% reserved regardless of load)"*
- *"GPU KV cache usage stayed under 1% even at 5 concurrent long-generation requests"*

Both are true, and they are measuring different things:

```
+------------------------- 24 GiB physical VRAM ------------------------+
|                                                                       |
|  [ AWQ weights ~5.5-6 GiB ][ activations/CUDA graphs ~1.5-2 GiB ]     |
|                                                                       |
|  +----------------- pre-allocated KV block pool ~14 GiB ------------+ |
|  |  [used][used][free][free][free][free][free][free][free][free]   | |
|  |   ^^^^^^^^^^                                                     | |
|  |   this fraction is what `gpu_cache_usage_perc` reports           | |
|  +------------------------------------------------------------------+ |
|                                                                       |
+-----------------------------------------------------------------------+
 ^
 `nvidia-smi memory.used` reports THIS WHOLE THING -- ~90% of 24 GiB,
 constant from the moment the server boots, whether idle or saturated.
```

vLLM asks CUDA for `gpu_memory_utilization` × total VRAM **once, at startup**,
and carves the KV pool out of it. That allocation never shrinks. So:

- **`nvidia-smi memory.used` is a flat line.** It tells you what vLLM reserved.
  It carries no information about load. It is useless as a ceiling signal.
- **`gpu_cache_usage_perc` from `/metrics` is occupancy *inside* the pool.**
  This is the real signal, and it is the one Week 2 failed to capture (W2-D).

State this explicitly in `week3-notes.md`. It is a genuine finding, it makes
your two Week 1 numbers consistent instead of contradictory, and it is the kind
of thing that distinguishes someone who has actually run a serving stack from
someone who has read about one.

### 1.2 — KV cache size, and why GQA changes the arithmetic

The KV cache stores, for every token, the Key and Value vectors at every layer,
so later tokens can attend to it without recomputation.

```
KV bytes per token = 2                (K and V)
                   x num_layers
                   x num_key_value_heads     <-- NOT num_attention_heads
                   x head_dim
                   x bytes_per_element
```

The trap is `num_key_value_heads`. Qwen2.5-7B uses **Grouped-Query Attention**:
28 query heads but only **4** KV heads. Seven query heads share each KV head. Had
you used `num_attention_heads` (28), you would have overestimated the KV cache
by **7×** and predicted a ceiling seven times too low.

Two more things that catch people:

- **AWQ does not quantize the KV cache.** AWQ is *weight-only* 4-bit
  quantization. KV stays fp16 → 2 bytes. (There is a separate
  `--kv-cache-dtype fp8` flag which would halve this. You are not using it. Not
  a Week 3 variable.)
- **`bytes_per_element` = 2**, not 1.

Verify the config numbers rather than trusting this document:

```bash
# after the model is cached locally, or from the HF model page's config.json
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct-AWQ')
for k in ['num_hidden_layers','num_attention_heads','num_key_value_heads',
          'hidden_size','max_position_embeddings','vocab_size',
          'tie_word_embeddings']:
    print(k, getattr(c, k, None))
print('head_dim', c.hidden_size // c.num_attention_heads)
"
```

### 1.3 — Blocks, and why 16 matters

The KV pool is not allocated per token. It is allocated in **blocks** of
`block_size` tokens (vLLM default: **16**).

```
KV per block = block_size x KV bytes per token
             = 16 x 56 KiB = 896 KiB = 0.875 MiB

A 7,168-token sequence needs ceil(7168 / 16) = 448 blocks.
Rounding waste: at most 15 token-slots per sequence -- negligible.
```

Two consequences that matter this week:

1. **The pool is counted in blocks.** vLLM's startup line reports the pool in
   tokens, but internally it is `num_gpu_blocks`. Preemption happens when
   `get_new_blocks()` cannot satisfy a request — a **block** shortage, not a
   byte shortage.
2. **Sharing is block-granular.** Two sequences can only share a prefix in
   whole 16-token blocks. A 20-token shared prefix shares exactly one block
   (16 tokens); the remaining 4 tokens sit in a partially-filled block that
   cannot be shared until it fills. This is why prefix-cache hit rates are
   quantized and why very short shared prefixes yield nothing.

### 1.4 — The three distinct queuing triggers

**Do not report "queuing starts at concurrency N."** Report *which limit bound
first*. There are at least three, they fire under different conditions, and
telling them apart is the actual intellectual content of Week 3.

| # | Trigger | What it is | Signature in your sampler CSV |
|---|---|---|---|
| **T1** | `max_num_seqs` | Hard cap on sequences in the running batch | `running` plateaus at an exact round number (256? 1024?) while `kv_cache_usage` stays low |
| **T2** | `max_num_batched_tokens` | Token budget per scheduler step, under chunked prefill | `waiting > 0` with **low** `kv_cache_usage`, appearing during the prefill burst at the start of a run |
| **T3** | KV block exhaustion | The actual memory ceiling | `kv_cache_usage` → ~1.0 **and** `preemptions` starts incrementing |

**T2 will probably fire first on axis D, and that is a finding, not a failure.**
With 6,144-token prompts and a default 8,192-token step budget, roughly one
prompt can prefill per scheduler step. Admitting 64 requests therefore takes
~64 steps, during which the rest sit in `waiting` — with an almost-empty KV
pool. If you saw `waiting > 0` and reported "the memory ceiling is at
concurrency 16", you would be wrong, and the sampler CSV is what proves it.

**Only T3 is the memory ceiling.** Report all three, labelled.

Read the resolved values of `max_num_seqs` and `max_num_batched_tokens` from the
startup log (Part 5 Step 1). Do not guess them — defaults change between vLLM
versions and are derived from other settings.

### 1.5 — Chunked prefill, and the prefill/decode distinction

Your Week 2 glossary already has the two phases right: prefill is compute-bound,
decode is memory-bandwidth-bound. One addition Week 3 depends on:

**Chunked prefill** means a long prompt's prefill is split across multiple
scheduler steps rather than monopolising one. A 6,144-token prompt with an
8,192-token budget does not need chunking on its own — but two of them in the
same step do. This is why the token budget, not KV memory, is the first thing to
bind on axis D.

It also means **prefill and decode share steps**: a step can carry one
sequence's prefill chunk alongside many sequences' decode tokens. That is why
axis B (prefill-heavy) will show elevated ITL for *other* requests — a decode
step that has to also carry a big prefill chunk takes longer. Watch for it;
it is the mechanism Sarathi-Serve (Week 5's paper) is about.

### 1.6 — Why your Week 2 p99 was not really a p99 {#p16}

<a name="p16"></a>

Week 2 ran `--num-prompts 100` per configuration. TTFT percentiles are computed
over **one sample per request** — so N = 100.

- p50 of 100 samples: fine.
- p95 of 100 samples: the 95th value. Shaky but meaningful.
- **p99 of 100 samples: the 99th of 100 values — effectively the maximum.** It
  is a single observation. It has no confidence interval. It will move a lot
  between runs.

ITL percentiles are fine, because they pool *every inter-token gap from every
request* — 100 requests × 255 gaps ≈ 25,500 samples. That is why your ITL
numbers look so stable (std_tpot 0.065 ms) while TTFT p99 does not.

**Week 3's fix:** 3 repeats per configuration, pooled → 300 TTFT samples. Report
the run-to-run spread alongside the mean, so the reader can see the variance
Week 2's single runs could not show. This also gives you a real reproducibility
check (see prediction P14).

### 1.7 — Prefix caching is ON by default, and it will distort your ceiling

vLLM v1 enables automatic prefix caching by default. This has two consequences
for the ceiling hunt that will mislead you if you do not know them in advance:

**(a) `--dataset-name random` prompts are not fully unique.** Random token IDs
differ, but `--backend openai-chat` wraps every prompt in the same chat template
(`<|im_start|>system…<|im_start|>user`). Those first ~20–30 tokens are
**identical across every request in the sweep**, so at least one block is shared
by all of them. Expect a small non-zero hit rate even with random prompts. This
is the prediction to put in `SECURITY-NOTES.md` (W2-C).

**(b) Freed blocks are not immediately free.** When a sequence finishes, its
blocks are not returned straight to the free list — they are kept in the prefix
cache, evictable on demand (typically LRU). So `gpu_cache_usage_perc` can read
**high without any actual pressure**, because much of that occupancy is cached
blocks that would be evicted the instant someone needed them.

**Therefore: high cache usage alone does not prove the ceiling.** The
unambiguous ceiling evidence is:

```
kv_cache_usage ~= 1.0   AND   waiting > 0   AND   preemptions increasing
```

`preemptions` is the honest signal, because a cached-but-unreferenced block is
evicted silently, whereas preempting a *running* sequence only happens under
genuine pressure. Make this explicit in the writeup — it is the difference
between "the gauge hit 100%" and "the stack ran out of memory."

*(There is a `--no-enable-prefix-caching` flag. Do **not** use it this week: it
changes the locked config. Note it as a Week 10 knob instead.)*

---

<a name="part-2"></a>
# Part 2 — Capacity arithmetic: predict the ceiling before measuring it

Do this **before** renting the pod. A measured number with a prediction attached
is a finding; a measured number alone is just a number.

### 2.1 — The estimate

Using Part 1.2 with Qwen2.5-7B-Instruct's real config (28 layers, 4 KV heads,
head_dim 128, fp16):

```
KV bytes per token = 2 x 28 x 4 x 128 x 2
                   = 57,344 bytes
                   = 56 KiB / token
```

```
VRAM budget       24 GiB x 0.90                              ~= 21.6 GiB
- AWQ weights     4-bit body + fp16 embed & lm_head          ~=  5.5-6.0 GiB
                  (Qwen2.5-7B does NOT tie embeddings, so
                   vocab 152064 x 3584 x 2 bytes is paid
                   TWICE ~= 2.2 GiB of the total)
- activations, CUDA graph capture, misc                      ~=  1.5-2.0 GiB
-------------------------------------------------------------------------
= KV block pool                                              ~= 13.5-14 GiB
= pool in tokens        14 GiB / 56 KiB                      ~= 250,000-270,000
= pool in blocks        that / 16                            ~= 15,600-16,900
```

### 2.2 — What that implies

| Sequence size | Predicted max concurrent sequences before KV exhaustion |
|---|---|
| **768 tok** (Week 2: 512 in + 256 out) | **~330** |
| **1,152 tok** (Week 3 axis A) | **~220** |
| **4,128 tok** (Week 3 axis B) | **~62** |
| **7,168 tok** (Week 3 axis D) | **~36** |
| **8,192 tok** (full `max-model-len`) | **~31** |

**This table is the single most useful thing in Part 2**, because it explains
Weeks 1 and 2 retroactively and redirects Week 3's strategy:

- Week 1 ran 5 concurrent requests. 5 out of ~330. Of course cache usage was
  <1%.
- Week 2 ran 20 concurrent × 768 tokens = 15,360 tokens in flight. That is
  **~6% of the pool**. The ceiling was never remotely approached.
- **Therefore: long context, not high concurrency, is the fast route to the
  ceiling.** Getting there by concurrency alone would need ~330 concurrent
  requests. Getting there with 7,168-token sequences needs ~36. That is why
  axis D exists and why it uses long prompts.

### 2.3 — Free confirmation from the startup log

vLLM prints the real numbers when it boots. You do not have to trust the
estimate above:

```
GPU KV cache size: <N> tokens
Maximum concurrency for 8,192 tokens per request: <M>x
```

That second line is doing exactly the division in the table above, for you, with
the true pool size. **Capture both lines** (Part 5 Step 1) and paste them into
`week3-notes.md`. They convert an estimate into a hard, citable baseline number
for this exact GPU + model + config — which is literally what the Q1 plan asks
Week 3 to produce.

---

<a name="part-3"></a>
# Part 3 — Experiment design

### 3.1 — Why Week 2's design was not enough

Week 2 varied concurrency alone, at a fixed 512 in / 256 out. That conflates two
independent effects: a request's TTFT depends on prefill work (input length) and
its E2EL depends on decode work (output length). With both pinned, you cannot
tell which one drives what.

Week 3 separates them by holding one long and the other short.

### 3.2 — The axes

| Axis | Input | Output | Total/seq | Concurrency | Prompts/run | What it isolates |
|---|---|---|---|---|---|---|
| **A** | 128 | 1024 | 1,152 | 1, 4, 16, 32 | 24 / 48 / 96 / 128 | **Decode-bound.** Negligible prefill. ITL/TPOT dominate E2EL. |
| **B** | 4096 | 32 | 4,128 | 1, 4, 16, 32 | 100 | **Prefill-bound.** TTFT dominates E2EL and should scale with input length. |
| **C** | 512 | 256 | 768 | 1, 2, 5, 10, 20, 40 | 40 / 100×5 | **Week 2 anchor.** Identical levels → reproducibility check, plus one new point at 40. |
| **D** | 6144 | 1024 | 7,168 | 8, 16, 32, 48, 64 | 2× conc (min 32) | **Ceiling hunt.** Predicted T3 onset ~36 concurrent. |
| **E** | 512 | 256 | 768 | 20 only, range-ratio 0 vs 0.5 | 100 | **Tests the Week 2 `max_concurrent_requests` prediction.** |

**Repeats:** 3 for axes A/B/C (→ 300 pooled TTFT samples, per Part 1.6). Axis D:
one pass, then repeat only the level where something interesting happens. Axis E:
one pass each — the effect is either there or it is not.

**Why axis D uses 6144 + 1024 = 7,168 and not 8,192:** the chat template adds
~20–30 tokens of wrapper per request. Requesting exactly `max-model-len` would
overflow and fail the run. Leaving ~1,000 tokens of headroom is cheap insurance.

**Why axis C repeats Week 2's exact levels:** it is the only reproducibility
test you will get before the Month 1 checkpoint. If concurrency-20 throughput
comes back within ±5% of 1621 tok/s, every Week 2 number is validated. If it
does not, that is a finding about run-to-run variance that retroactively
qualifies the whole Week 2 table — and you need to know that *before* writing
the Month 1 report in Week 4.

**Priority if the session overruns: D > C > B > A > E.** Axis D is the week's
actual deliverable. Axis A is the most expendable and the most expensive per
data point.

### 3.3 — Axis E: testing your own Week 2 prediction

Your Week 2 teardown resolved the `max_concurrent_requests = 2 × configured`
mystery from source: `calculate_metrics()` buckets each request's start and end
into whole integer seconds and increments *inclusively*, so when all requests
are identical length, a departing wave and an arriving wave land in the same
bucket and it counts 2N.

You then wrote: *"Testable prediction for Week 3: once prompt/output lengths
vary, waves should desync, and this exact 2x pattern should weaken or
disappear."*

`--random-range-ratio` is exactly the knob. It samples input and output lengths
from a range around the target instead of pinning them. So:

- `--random-range-ratio 0` → all requests identical → waves synchronise → expect
  ratio ≈ 2.0
- `--random-range-ratio 0.5` → lengths vary ±50% → waves desync → expect ratio
  well below 2.0

Two runs, ~2 minutes, and your source-derived prediction is either confirmed or
refuted. **Confirm the flag's exact semantics from your clone before relying on
it** (Part 4.7) — in some versions the range is `[len × (1−r), len × (1+r)]` and
in others it is defined differently.

### 3.4 — One intentional config change from Week 2

Week 2's log carried this warning:

```
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests
by default. The default will be determined on the server side and can be
model/API specific. For the old behavior, include --temperature=0.
```

Week 3 adds `--temperature 0` to every run, so sampling is deterministic and the
runs are reproducible. Decode cost per token is essentially independent of
temperature when token counts are pinned by `--ignore-eos`, so this should not
move the numbers — but **it is a real config delta and must be disclosed** when
comparing axis C against Week 2. Treat the axis-C rerun as the new anchor, and
say so in the notes.

---

<a name="part-4"></a>
# Part 4 — Build the tooling, step by step

Four scripts. Build them in this order, in `q1-vllm-serving/`. Each subsection
explains what the code does and why, so you type it rather than paste it.

**Files you will end up with:**

| File | Purpose | Closes |
|---|---|---|
| `week3-sweep.sh` | The length × length × concurrency sweep | The week's stated deliverable |
| `metrics-sampler.sh` | 4 Hz `/metrics` → CSV | W2-D, the memory ceiling |
| `gpu-sampler.sh` | Unambiguous `nvidia-smi` sampling | The Week 2 column ambiguity |
| `parse_results.py` | JSONs → CSV + markdown tables | The results table |

### 4.1 — Start from one single benchmark invocation

Before writing any loop, understand one run. This is `week2-sweep.sh`'s body
with the Week 3 additions marked:

```bash
vllm bench serve \
  --backend openai-chat \                  # which client shape to use
  --base-url http://localhost:8000 \       # NEW: explicit, don't rely on defaults
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \                  # synthetic prompts, no dataset download
  --random-input-len 512 \                 # target prompt length in tokens
  --random-output-len 256 \                # target generation length in tokens
  --random-range-ratio 0 \                 # NEW: length variance. 0 = all identical
  --ignore-eos \                           # generate exactly N tokens, never stop early
  --temperature 0 \                        # NEW: greedy, deterministic (see 3.4)
  --num-prompts 100 \                      # total requests in this run
  --max-concurrency 20 \                   # cap on in-flight requests
  --request-rate inf \                     # fire as fast as concurrency allows
  --percentile-metrics ttft,tpot,itl,e2el \# NEW: e2el added
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename example.json \
  --save-detailed                          # NEW: per-request arrays (see 4.5)
```

**The three flags that define the shape of a run, and why:**

- **`--ignore-eos`** is what makes this a controlled experiment. Without it the
  model stops at `<EOS>` and every request generates a different number of
  tokens, so throughput would depend on the model's chattiness rather than on
  your variable. With it, every request generates *exactly*
  `--random-output-len` tokens. This is why your Week 2 ITL numbers are so
  clean — and, per your own teardown, it is also the direct cause of the
  `max_concurrent_requests` 2× artifact, because equal-length requests
  synchronise into waves.
- **`--request-rate inf` + `--max-concurrency N`** together mean *closed-loop*
  load: keep exactly N requests in flight at all times, launching a replacement
  the moment one finishes. This is a saturation test. The alternative
  (`--request-rate 5` for 5 req/s) is *open-loop* — an arrival-rate test, which
  is what you would use to find a latency SLO. **Week 3 stays closed-loop.**
  Note the distinction in your notes; it is a real fork in benchmarking
  methodology and interviewers ask about it.
- **`--seed`** makes the random prompts reproducible. Week 3 uses `42 + repeat`
  so each repeat gets *different but reproducible* prompts — repeats should
  sample run-to-run variance, not re-run the identical workload.

Run this once by hand on the pod before scripting anything. Read the output
block. Then script it.

### 4.2 — `week3-sweep.sh`, layer by layer

**Layer 1 — a function, so the flag list appears once.**

```bash
#!/usr/bin/env bash
set -euo pipefail
# -e  exit on any command failure
# -u  error on unset variable -- catches typos in $VARNAMES
# -o pipefail  a failure anywhere in a pipeline fails the whole pipeline

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
OUTDIR="${OUTDIR:-results/week3}"
REPEATS="${REPEATS:-3}"
AXES="${AXES:-A B C D E}"
SEED="${SEED:-42}"
DETAILED="${DETAILED:-1}"
SETTLE="${SETTLE:-5}"

mkdir -p "$OUTDIR"
```

`${VAR:-default}` means "use `$VAR` if set, otherwise the default". This is what
lets you override anything from the command line without editing the file:
`REPEATS=1 AXES="C" ./week3-sweep.sh`. Worth the habit — on a metered pod you
will want to re-run a subset without opening an editor.

**Layer 2 — the `bench` function.**

```bash
bench () {
  local tag=$1 inlen=$2 outlen=$3 conc=$4 prompts=$5 rep=$6 rrr=${7:-0}
  local name="${tag}_in${inlen}_out${outlen}_c${conc}_rrr${rrr}_r${rep}"
  local extra=()
  [[ "$DETAILED" == "1" ]] && extra+=(--save-detailed)

  echo ""
  echo "=================================================================="
  echo "RUN ${name}   $(date -u +%FT%TZ)"
  echo "=================================================================="
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len "$inlen" \
    --random-output-len "$outlen" \
    --random-range-ratio "$rrr" \
    --ignore-eos \
    --temperature 0 \
    --num-prompts "$prompts" \
    --max-concurrency "$conc" \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --seed "$(( SEED + rep ))" \
    --save-result \
    --result-dir "$OUTDIR" \
    --result-filename "${name}.json" \
    "${extra[@]}"
  sleep "$SETTLE"
}
```

Three details that matter more than they look:

- **The filename encodes the full experiment coordinates**
  (`A_in128_out1024_c16_rrr0_r2.json`). Week 2's `week2_conc5.json` only encoded
  concurrency, because that was the only variable. Week 3 has five. Encoding
  them in the filename means the parser reconstructs the whole design from
  `ls`, and you never have to maintain a separate index that can drift out of
  sync with the data.
- **`echo "RUN ... $(date -u +%FT%TZ)"` before every run.** This is what lets
  you slice the continuous sampler CSV into per-run windows afterwards. Without
  it you have one long undifferentiated timeline and no way to say which run a
  `waiting > 0` event belongs to. **This one line is what makes the samplers
  useful.**
- **`sleep "$SETTLE"` after every run.** Lets the engine finish draining and the
  metrics settle, so run N's tail does not contaminate run N+1's opening
  samples.

**Layer 3 — the warm-up. This closes Week 2's G1.**

Week 2's concurrency-1 run had the **highest** TTFT in the sweep (98 ms),
breaking an otherwise clean monotonic trend, with a tight distribution
(std ≈ 13 ms, all percentiles clustered 100–104 ms). A tight distribution means
the whole run was slow — not one or two outliers. Leading hypothesis: one-time
cost paid by whichever run went first — CUDA graph capture for an unseen batch
shape, or GPU clock ramp-up from an idle power state.

```bash
echo "### WARM-UP (not recorded) ###"
for c in 1 4 16 32; do
  vllm bench serve \
    --backend openai-chat --base-url "$BASE_URL" --model "$MODEL" \
    --endpoint /v1/chat/completions --dataset-name random \
    --random-input-len 512 --random-output-len 128 --ignore-eos \
    --temperature 0 --num-prompts $(( c * 2 )) --max-concurrency "$c" \
    --request-rate inf --seed 7 >/dev/null
done
sleep 10
echo "### WARM-UP DONE ###"
```

Note there is **no `--save-result`** — these runs are deliberately discarded.
The loop touches the batch shapes used later so the one-time cost is paid before
any measured run.

**Layer 4 — the axes.**

```bash
for rep in $(seq 0 $(( REPEATS - 1 ))); do

  if [[ " $AXES " == *" A "* ]]; then          # decode-bound
    bench A 128 1024 1   24  "$rep"
    bench A 128 1024 4   48  "$rep"
    bench A 128 1024 16  96  "$rep"
    bench A 128 1024 32 128  "$rep"
  fi

  if [[ " $AXES " == *" B "* ]]; then          # prefill-bound
    bench B 4096 32 1  100 "$rep"
    bench B 4096 32 4  100 "$rep"
    bench B 4096 32 16 100 "$rep"
    bench B 4096 32 32 100 "$rep"
  fi

  if [[ " $AXES " == *" C "* ]]; then          # Week 2 anchor
    bench C 512 256 1   40 "$rep"
    bench C 512 256 2  100 "$rep"
    bench C 512 256 5  100 "$rep"
    bench C 512 256 10 100 "$rep"
    bench C 512 256 20 100 "$rep"
    bench C 512 256 40 100 "$rep"
  fi
done
```

The `[[ " $AXES " == *" A "* ]]` pattern is a substring test with spaces around
both sides, so `AXES="A"` matches `A` but `AXES="AB"` does not match `A`. Lets
you run `AXES="D" ./week3-sweep.sh` to hit only the ceiling hunt.

**Why `--num-prompts` varies with concurrency.** At concurrency 1, axis A's
1,024-token generations take ~9.5 s each; 100 of them would be ~16 minutes of
paid time for a configuration whose percentile spread you already know is tiny.
24 prompts is enough for a single-stream baseline. At concurrency 32 you want
enough requests that the steady state dominates the ramp-up, hence 128.

**Layer 5 — axis D, the ceiling hunt.**

```bash
if [[ " $AXES " == *" D "* ]]; then
  for c in 8 16 32 48 64; do
    p=$(( c * 2 )); (( p < 32 )) && p=32
    bench D 6144 1024 "$c" "$p" 0
  done
fi
```

`p = 2 × concurrency` means every run processes exactly two "waves", which is
enough to reach steady state without paying for 100 long-context requests at
every level. `min 32` keeps the low end statistically non-trivial.

**Layer 6 — axis E and the G1 re-run.**

```bash
if [[ " $AXES " == *" E "* ]]; then
  bench E 512 256 20 100 0 0.0     # identical lengths -> waves sync
  bench E 512 256 20 100 0 0.5     # varied lengths    -> waves desync
fi

# G1: concurrency=1 again, LAST, after the GPU has been hot for an hour.
bench G1RERUN 512 256 1 40 0

echo ""
echo "SWEEP COMPLETE $(date -u +%FT%TZ)"
```

The G1 re-run placement is the whole experiment: **identical config to axis C's
concurrency-1 run, but executed last instead of first.** If Week 2's 98 ms was a
cold-start artifact, this run comes back near ~25 ms and the anomaly is
explained. If it comes back at ~98 ms again, the warm-up hypothesis is wrong and
something structural about concurrency-1 is slow — which would be a more
interesting finding, and one you would carry into Month 2's scheduler tracing.

**Layer 7 — a smoke-test escape hatch.** Put this right after the config block,
before the warm-up:

```bash
if [[ "${SMOKE:-0}" == "1" ]]; then
  bench SMOKE 128 32 4 8 0 0
  echo "SMOKE OK - all flags accepted. Re-run without SMOKE=1."
  exit 0
fi
```

`SMOKE=1 ./week3-sweep.sh` runs one ~10-second benchmark whose only purpose is
to prove every flag is accepted by this build. If `--save-detailed` or
`--temperature` does not exist, you find out for 10 seconds of pod time instead
of discovering it 40 minutes into the sweep. Combined with the source check in
Part 4.7, this makes a wasted session essentially impossible.

### 4.3 — `metrics-sampler.sh`: the most important new artifact

**Why this exists.** Weeks 1 and 2 both hit the same wall, and you documented it
both times:

> Week 1: *"the log's rolling window missed the live `Running: 5 reqs` moment on
> the first attempt"*
> Week 1: *"engine log stats are 10-second rolling-window snapshots, not
> instantaneous rates"*
> Week 2: *"pane 3's `/metrics | grep cache` step was missed/incomplete"*

The engine log cannot resolve short-lived scheduler states. Week 3's entire
deliverable is a short-lived scheduler state: *the moment `waiting` first goes
above 0*. So the log is structurally incapable of producing Week 3's answer.
This script replaces it with 4 Hz sampling into a timestamped CSV.

**Layer 1 — metric names are version-dependent, so discover them.**

This is precisely where Week 2 got stuck: the gauge name was never found. Do not
hardcode a guess.

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUT="${1:?usage: metrics-sampler.sh <outfile.csv>}"
INTERVAL="${INTERVAL:-0.25}"

M_RUN="${M_RUN:-vllm:num_requests_running}"
M_WAIT="${M_WAIT:-vllm:num_requests_waiting}"
M_CACHE="${M_CACHE:-auto}"
M_PREEMPT="${M_PREEMPT:-vllm:num_preemptions_total}"
M_PFX_HIT="${M_PFX_HIT:-vllm:prefix_cache_hits_total}"
M_PFX_Q="${M_PFX_Q:-vllm:prefix_cache_queries_total}"

if [[ "$M_CACHE" == "auto" ]]; then
  probe=$(curl -s "$BASE_URL/metrics")
  for cand in vllm:gpu_cache_usage_perc vllm:kv_cache_usage_perc \
              vllm:gpu_cache_usage_percent; do
    if printf '%s\n' "$probe" | grep -q "^${cand}"; then M_CACHE="$cand"; break; fi
  done
  if [[ "$M_CACHE" == "auto" ]]; then
    echo "FATAL: no KV cache gauge matched. Check metric-names.txt, then" >&2
    echo "re-run with M_CACHE=<name> ./metrics-sampler.sh <out.csv>"      >&2
    exit 1
  fi
  echo "kv cache metric = $M_CACHE" >&2
fi
```

`${1:?usage: ...}` errors out with that message if no argument is given —
cheaper than writing an if-statement, and it fails loudly rather than writing to
an empty filename.

The `for cand in ...` loop tries the known name variants across vLLM v0/v1 and
tells you which one it found. If it exits, Part 5 Step 4 has dumped the real
list and you override with `M_CACHE=<name>`. **This is the design that closes
W2-D**: it cannot silently succeed while recording nothing.

**Layer 2 — parsing Prometheus text format.**

`/metrics` returns lines like:

```
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 20.0
```

You want the last whitespace-separated field of the line starting with your
metric name, skipping `#` comment lines:

```bash
echo "ts,running,waiting,kv_cache_usage,preemptions,prefix_hits,prefix_queries" > "$OUT"
trap 'echo "sampler stopped -> $OUT" >&2; exit 0' INT TERM

while true; do
  snap=$(curl -s --max-time 2 "$BASE_URL/metrics" || true)
  ts=$(date +%s.%N)
  get () {
    printf '%s\n' "$snap" \
      | awk -v k="$1" 'index($0,k)==1 && substr($0,1,1)!="#" {print $NF; exit}'
  }
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$ts" \
    "$(get "$M_RUN")"     "$(get "$M_WAIT")" \
    "$(get "$M_CACHE")"   "$(get "$M_PREEMPT")" \
    "$(get "$M_PFX_HIT")" "$(get "$M_PFX_Q")" >> "$OUT"
  sleep "$INTERVAL"
done
```

Line by line:

- **`snap=$(curl -s ...)` fetched once per iteration**, then parsed six times.
  Six separate `curl` calls would give you six *different* moments in one CSV
  row — the values would not be mutually consistent, which defeats the point of
  correlating `waiting` against `kv_cache_usage`.
- **`|| true`** so a transient curl failure records a row of empty fields
  instead of killing the sampler (remember `set -e` is on). You want a gap in
  the data, not a dead sampler discovered an hour later.
- **`--max-time 2`** so a hung request cannot stall the loop indefinitely.
- **`date +%s.%N`** = Unix seconds with nanoseconds. Same clock base as
  `date -u +%FT%TZ` in the sweep log, so the two files are joinable.
- **`index($0,k)==1`** means "the line *starts with* this key" — safer than a
  bare grep, which would also match `vllm:num_requests_running_total` or a
  substring inside a HELP line.
- **`exit` inside the awk block** stops after the first match. There is one
  series per metric here, but with multiple engines or models there could be
  several, and you would want to notice rather than silently take one.
- **`trap ... INT TERM`** so Ctrl-C prints where the file went instead of dying
  silently.

**Why 4 Hz (`INTERVAL=0.25`)?** A scheduler step at concurrency 20 is ~12 ms
(your measured ITL), so 250 ms is ~20 steps — far too coarse to see individual
steps, but far *finer* than the 10-second log window, and fine enough to catch a
queuing episode lasting a fraction of a second. Cost: ~21,600 rows over 90
minutes, about 1.5 MB. Cheap enough to commit.

### 4.4 — `gpu-sampler.sh`: resolving the Week 2 column ambiguity

Week 2's note reads: *"saw ~35% baseline, spiking to 37-38%, with brief spikes
toward 100% — TODO confirm which column (`memory.used` % vs `utilization.gpu` %)
these numbers came from."* The fix is to name every column explicitly:

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: gpu-sampler.sh <outfile.csv>}"
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
  --format=csv,nounits \
  -lms 500 > "$OUT"
```

`--query-gpu=` with `--format=csv` writes a header row naming each field, so the
ambiguity cannot recur. `-lms 500` loops every 500 ms (if your `nvidia-smi`
rejects `-lms`, fall back to `-l 1` for 1-second intervals — coarser, but the
`/metrics` sampler is the one that matters for the ceiling).

**The three fields, and the trap:**

| Field | What it actually is |
|---|---|
| `memory.used` | MiB of VRAM allocated. Pinned near 90% of 24 GB from server start, barely moves (Part 1.1). **Useless as a load signal.** |
| `utilization.gpu` | **Percent of time** a kernel was running. Not memory. |
| `utilization.memory` | **Percent of time** the memory bus was being read or written. **NOT the fraction of VRAM in use.** The single most commonly misread `nvidia-smi` field. |

**Near-certain resolution of the Week 2 ambiguity, which you can reason out now
without a GPU:** ~35% *cannot* have been `memory.used` as a percentage, because
with `gpu_memory_utilization=0.90` that value is a flat ~90%. So it was
`utilization.gpu`. Which means the observation **does** confirm the
prefill/decode split from your PagedAttention notes: GPU compute mostly idle at
~35% during decode (memory-bandwidth-bound, waiting on weight reads), spiking
toward 100% during prefill (compute-bound, big GEMMs).

Confirm it with a labelled column this week rather than asserting it — but note
in `week3-notes.md` that the inference was available from the config alone, and
what that says about recording units alongside numbers.

Also add `power.draw` and `clocks.sm` to the query: if the concurrency-1 anomaly
is clock ramp-up (G1's hypothesis), `clocks.sm` is where you would see it. The
G1 re-run plus this column is a direct test.

### 4.5 — `parse_results.py`: from JSONs to a table

Each result JSON is a flat dict of ~40 scalars (you have five of them from Week
2 to develop against). The parser needs to:

1. Reconstruct experiment coordinates from the filename.
2. Pull the scalar metrics.
3. Group repeats and aggregate.
4. Emit a flat CSV (for Week 4's charts) and markdown tables (for the notes).

**The filename regex is the core of it:**

```python
import re
W3 = re.compile(
    r"^(?P<tag>[A-Za-z0-9]+)_in(?P<inlen>\d+)_out(?P<outlen>\d+)"
    r"_c(?P<conc>\d+)_rrr(?P<rrr>[\d.]+)_r(?P<rep>\d+)$"
)
W2 = re.compile(r"^week2_conc(?P<conc>\d+)$")     # so Week 2 data still parses
```

Keeping the Week 2 pattern means the same tool reads both weeks' data — which is
how you validate it (Part 4.6) and how Week 4 builds one continuous dataset.

**Scalars worth extracting:**

```python
SCALARS = [
    "duration", "completed", "failed",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "max_output_tokens_per_s", "max_concurrent_requests",
    "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms",
    "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p50_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "p50_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
]
```

Use `raw.get(key)` rather than `raw[key]` throughout — Week 2's JSONs have no
`e2el` fields (you did not request them), and the parser must handle both weeks.

**Three derived columns that do real analytical work:**

```python
# 1. The Week 2 G4 artifact ratio, computed rather than eyeballed.
mc, cfg = raw.get("max_concurrent_requests"), raw.get("max_concurrency")
row["mcr_ratio"] = round(mc / cfg, 3) if mc and cfg else None

# 2. Run-to-run spread across repeats -- the variance Week 2 could not show.
def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    m = statistics.fmean(vals)
    if len(vals) < 2 or m == 0: return m, 0.0
    return m, (max(vals) - min(vals)) / m * 100.0

# 3. Pooled percentiles from --save-detailed, when available.
pooled_ttft = [v for r in group for v in r.get("ttfts", [])]
```

That third one is why `--save-detailed` is in the sweep. Without it you can only
average each run's *own* p99 — the mean of three near-maxima of 100 samples,
which is not a p99 of 300 samples. With it you pool all 300 raw TTFT values and
compute a real percentile. Have the parser report which method it used, so the
notes state it honestly.

*(Note `ttfts` are in **seconds** in the raw arrays while the summary fields are
in **milliseconds** — multiply by 1000 when pooling, or your table will silently
mix units by a factor of 1000.)*

### 4.6 — Validate the parser against Week 2 data (free, and do it first)

You have five committed Week 2 JSONs. Point the parser at them **before** the pod
session:

```bash
cd /d/dev/q1-vllm-serving
python parse_results.py results/week2 --out /tmp/w2check
cat /tmp/w2check.md
```

It must reproduce the table already in `week2-notes.md` exactly — TTFT mean
98.41 / 24.36 / 32.10 / 44.21 / 71.39, ITL mean 8.826 → 12.033, throughput
108.55 → 1621.29. If any cell disagrees, the parser is wrong, and you fix it
now for free rather than on a metered pod with fresh data you cannot yet
sanity-check.

**This also gives you a free confirmation of your own Week 2 teardown.** The
`mcr_ratio` column should come out as exactly `2.0` at all five concurrency
levels — independent arithmetic confirming the source-derived explanation you
wrote. That is a nice thing to be able to say in the notes: the artifact was
predicted from source, then confirmed numerically across every data point.

### 4.7 — Verify every flag from the source clone (free) {#p47}

<a name="p47"></a>

The main way a paid pod session gets wasted is a flag that does not exist in
your pinned version. You will have the source cloned from W2-B, so check locally
instead of guessing:

```bash
cd ~/src/vllm-0.28.0
grep -rn "save-detailed\|save_detailed"       vllm/benchmarks/
grep -rn "random-range-ratio\|random_range_ratio" vllm/benchmarks/
grep -rn "random-prefix-len\|random_prefix_len"   vllm/benchmarks/
grep -rn '"--temperature"'                     vllm/benchmarks/
grep -rn '"--base-url"'                        vllm/benchmarks/
```

For each: confirm it exists, and **read its `help=` string and `default=`.** Two
that matter specifically:

- **`--random-range-ratio`** — read the code that consumes it, not just the help
  text. Confirm whether the sampled range is `[len × (1−r), len × (1+r)]` or
  something else, and what the default is. Axis E's entire interpretation
  depends on this.
- **`--random-prefix-len`** — a *deliberately shared* prefix prepended to every
  random prompt. You are not using it in the sweep, but note it in
  `SECURITY-NOTES.md`: it is the exact knob for measuring prefix-cache hit rate
  as a function of shared-prefix length, which is the natural next step for the
  side-channel thread you have been tracking since Week 1. A candidate
  opportunistic experiment if the pod session runs short: three runs at
  `--random-prefix-len 0 / 256 / 1024`, same everything else, and watch
  `prefix_hits / prefix_queries` in the sampler CSV.

While you are in the source, also confirm the metric names for Part 4.3 so the
`auto` probe is a formality rather than a gamble:

```bash
grep -rn "cache_usage\|num_preemptions\|prefix_cache" vllm/v1/metrics/
```

---

<a name="part-5"></a>
# Part 5 — Pod session runbook

Every step in order. Three tmux panes. Budget ~2 hours of paid time, ~90 minutes
of it actual benchmarking.

**Estimated breakdown** (from Week 2's measured throughput, extrapolated):

| Phase | Time |
|---|---|
| Boot + model load + verification | 5–8 min |
| Metric-name capture + smoke test | 5 min |
| Warm-up | 3 min |
| Axis A × 3 repeats | ~32 min |
| Axis B × 3 repeats | ~13 min |
| Axis C × 3 repeats | ~8 min |
| Axis D single pass | ~12 min |
| Axis E | ~2 min |
| G1 re-run | ~4 min |
| Pull artifacts off pod, verify locally | 5 min |
| **Total** | **~90 min + boot** |

### Step 1 — Boot, and capture the startup log

```bash
cd /workspace/vllm-serving-internals/q1-vllm-serving   # adjust to your pod path
mkdir -p results/week3

./serve.sh 2>&1 | tee results/week3/server-startup.log
```

The `| tee` is not optional. The startup log contains the capacity numbers
Week 3 is supposed to report, and they scroll past in seconds. While it loads,
in another pane:

```bash
grep -Ei "KV cache size|Maximum concurrency|max_num_seqs|max_num_batched_tokens|Graph capturing|chunked prefill|block_size" \
  results/week3/server-startup.log | tee results/week3/capacity-config.txt
```

**Paste `capacity-config.txt` straight into `week3-notes.md`.** Compare against
Part 2's estimate of ~250–270k tokens. These lines give you:

- the true KV pool size → validates or corrects Part 2.1
- `Maximum concurrency for 8,192 tokens per request: Mx` → the ceiling, computed
  by vLLM itself
- resolved `max_num_seqs` and `max_num_batched_tokens` → **T1 and T2 from
  Part 1.4**, which you need to interpret axis D at all
- `block_size` → confirms the 16 assumed in Part 1.3

### Step 2 — Verify the model (Week 1's lesson: never trust the logs)

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool \
  | tee results/week3/models.json
```

Week 1's whole incident was a template silently serving Qwen3-0.6B while the env
var said otherwise. Confirm `Qwen/Qwen2.5-7B-Instruct-AWQ`.

**Also re-verify the revision hash against `ENVIRONMENT.md`**
(`b25037543e9394b818fdfca67ab2a00ecc7dd641`). If Hugging Face has published a
new revision and your volume pulled it, your "locked stack" is no longer locked
and every Week 3 number is measured against a different model than Week 2's.
Check the startup log or the model cache path for the resolved commit. The
end-of-quarter checklist requires `ENVIRONMENT.md` be *"accurate, unchanged
since Week 1"* — this is the step that enforces it.

### Step 3 — Start both samplers (panes 2 and 3)

```bash
# pane 2
./gpu-sampler.sh results/week3/gpu_full-session.csv

# pane 3
./metrics-sampler.sh results/week3/metrics_full-session.csv
```

**Leave both running for the whole session.** One continuous timeline is easier
to work with than restarting samplers between runs, because the sweep script
prints a UTC timestamp before each run (Part 4.2, Layer 2) and you slice
afterwards. Restarting per-run risks missing exactly the transition you care
about.

Confirm pane 3 printed `kv cache metric = ...`. If it exited fatally, do Step 4
first, then restart it with the right name.

### Step 4 — Capture the metric names. DO NOT SKIP. This is W2-D.

This is the step Week 2 missed.

```bash
curl -s localhost:8000/metrics | grep -E '^# HELP vllm:' | sort \
  > results/week3/metric-names.txt
curl -s localhost:8000/metrics > results/week3/metrics-idle-snapshot.txt
wc -l results/week3/metric-names.txt
```

Two files, deliberately:

- `metric-names.txt` — the catalogue. Commit it. It answers "what is observable
  on this stack" permanently, and it is a reference you will use in Month 2's
  source tracing and Month 3's perturbation work.
- `metrics-idle-snapshot.txt` — every metric's *value at idle*, before any load.
  This is your zero point. Cache usage at idle, prefix-cache counters at idle,
  preemptions at idle (should be 0). Without a zero point, a counter reading
  mid-sweep tells you nothing, because counters are cumulative since boot.

Check that these exist and note their exact spellings:
`num_requests_running`, `num_requests_waiting`, a KV/GPU cache-usage gauge,
`num_preemptions_total`, `prefix_cache_hits_total` / `prefix_cache_queries_total`,
`request_queue_time_seconds`, `request_prefill_time_seconds`,
`request_decode_time_seconds`, `iteration_tokens_total`.

Those last four are histograms, and they are worth noting for Month 2 even if
you do not sample them: `request_queue_time_seconds` is the server's *own*
measurement of queuing delay, which is a direct cross-check on your client-side
TTFT. If TTFT rises and `queue_time` rises with it, the delay is scheduler
queuing; if TTFT rises and `queue_time` stays flat, it is prefill compute. That
distinction is exactly what axis B is trying to establish, and the server will
just tell you.

### Step 5 — Smoke test (10 seconds of pod time, saves the session)

```bash
SMOKE=1 ./week3-sweep.sh
```

One tiny benchmark, purely to prove every flag is accepted. Part 4.7 should have
made this a formality, but a formality that costs 10 seconds and eliminates a
90-minute failure mode is worth keeping. On failure: re-run with `DETAILED=0`,
or delete the offending flag.

### Step 6 — Run the sweep

```bash
./week3-sweep.sh 2>&1 | tee results/week3/week3-sweep.log
```

**Watch pane 3 during axis D.** You are looking for two separate moments, and
distinguishing them is the week's deliverable:

1. The first time `waiting` goes above 0 — **note the `kv_cache_usage` value at
   that same instant.** Low → T2 (token budget). High → T3 (memory). This single
   correlation is the finding.
2. The first time `preemptions` increments — that is T3, unambiguously.

Note the wall-clock times so you can locate them in the CSV afterwards.

### Step 7 — If axis D at concurrency 64 shows no queuing

Escalate cheaply, in this order, and **record which one finally triggered it**:

```bash
# (a) more concurrent long sequences
vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 6144 --random-output-len 1024 \
  --ignore-eos --temperature 0 --num-prompts 192 --max-concurrency 96 \
  --request-rate inf --save-result --result-dir results/week3 \
  --result-filename D_in6144_out1024_c96_rrr0_r0.json

# (b) longer sequences: 7000 in / 1000 out = 8000, leaving ~190 tokens
#     of headroom for the chat template under max-model-len 8192
```

**Do not raise `--gpu-memory-utilization`. Do not raise `--max-model-len`.**
Those are Week 10 variables, and touching them breaks the locked stack.

If the ceiling genuinely cannot be reached inside the locked config, **that is
the reportable finding**, stated plainly: *"On this GPU/model/config the KV pool
is large enough that scheduler token-budget limits bind before memory does, up
to concurrency N. The memory ceiling was not reached and requires either longer
context than `max-model-len` permits or a reduced KV pool — deferred to
Week 10."* An honest negative result with a mechanism is a legitimate
deliverable; the Q1 plan says so explicitly.

### Step 8 — Get everything off the pod, then terminate

```bash
ls -la results/week3/ | tee results/week3/MANIFEST.txt
tar czf week3-results.tar.gz results/week3/
# scp / runpodctl / rsync to the ThinkPad
```

Verify the tarball opens locally **and** that the JSON count matches what you
expect (3 repeats × 14 configs + 5 axis-D + 2 axis-E + 1 G1 = 50) **before**
terminating. Then **terminate** the pod, not stop — per the cost rule.

Do not analyse on the pod. Analysis is free on the ThinkPad.

---

<a name="part-6"></a>
# Part 6 — Analysis

### 6.1 — Build the tables

```bash
cd /d/dev/q1-vllm-serving
python parse_results.py results/week3 --out results/week3/summary
```

### 6.2 — Slice the sampler CSV per run

The sweep log has a `RUN <name> <UTC timestamp>` line before each run; the
sampler CSV has Unix-epoch timestamps. Join them to get per-run windows. The
questions to answer from each window:

| Question | Column to read |
|---|---|
| Peak KV cache usage in this run | max of `kv_cache_usage` |
| Did anything queue? | max of `waiting` |
| **At the first `waiting > 0`, what was `kv_cache_usage`?** | the correlation — T2 vs T3 |
| Did `running` plateau at a round number? | max of `running` vs `max_num_seqs` — T1 |
| Any preemptions? | `preemptions` delta across the window |
| Prefix-cache hit rate | `(prefix_hits delta) / (prefix_queries delta)` |

**Counters are cumulative since boot** — `preemptions`, `prefix_hits`, and
`prefix_queries` must be differenced across the window, not read directly. This
is what `metrics-idle-snapshot.txt` from Step 4 is the zero point for.

### 6.3 — The finding to write

Not "queuing started at concurrency N". Write, with evidence:

> At concurrency C, `waiting` first exceeded 0 while `kv_cache_usage` was only
> X% — so the binding constraint was **T2, the per-step token budget of
> `max_num_batched_tokens` = B**, not memory. KV cache usage did not exceed 90%
> until concurrency D, and the first preemption occurred at concurrency E. The
> memory ceiling for this GPU + model + config is therefore ~P tokens of KV
> pool, which at 7,168 tokens per sequence is ~S concurrent sequences —
> matching / diverging from the startup log's stated `Maximum concurrency` of M.

Then cross-reference each claim against the Part 7 source trace.

### 6.4 — Also settle these

- **Axis A vs B at matched concurrency** — does TTFT track input length? Does ITL
  stay roughly independent of it? If axis B's ITL is elevated, that is chunked
  prefill stealing decode-step time (Part 1.5), and it previews Sarathi-Serve.
- **Axis C vs Week 2** — within ±5%? (Prediction P14.) If not, that is a
  variance finding that qualifies the entire Week 2 table, and Week 4's Month 1
  report must say so.
- **G1** — did the concurrency-1 TTFT anomaly disappear? Check `clocks.sm` in the
  GPU CSV for the clock-ramp hypothesis specifically.
- **Axis E** — did `mcr_ratio` drop below 2.0 at range-ratio 0.5? Your
  source-derived prediction, confirmed or refuted.
- **G5 / SECURITY-NOTES** — prefix-cache hit rate as a function of input length
  and concurrency, now that you have continuous sampling instead of Week 1's
  three spot checks.

---

<a name="part-7"></a>
# Part 7 — Weekend teardown: what triggers queuing

You now have both halves: an observed threshold from Part 6, and the source
cloned back in W2-B. Trace the mechanism.

Find the files (paths shift between versions — grep, do not assume):

```bash
cd ~/src/vllm-0.28.0
find vllm/v1/core -name '*.py'
grep -rn "class Scheduler"     vllm/v1/core/sched/
grep -rn "def schedule"        vllm/v1/core/sched/scheduler.py
grep -rn "def allocate_slots"  vllm/v1/core/
grep -rni "preempt"            vllm/v1/core/sched/scheduler.py
grep -rn "token_budget\|max_num_scheduled_tokens" vllm/v1/core/sched/scheduler.py
```

**Four questions, each anchored to a number from your own sweep:**

1. **What is the waiting pool, concretely?** Find the data structure (a deque? a
   priority queue?) and the loop that drains it in `Scheduler.schedule()`.
2. **What exactly stops a waiting request from being admitted this step?** Look
   for the token-budget variable, the running-request cap, and the call into the
   KV cache manager that can fail to return blocks. There should be **more than
   one `break` / `continue`** — and each one is a different trigger from
   Part 1.4. Map them: this `break` is T1, that one is T2, that one is T3.
3. **Which one fired in your axis-D run?** Use the sampler CSV: was
   `kv_cache_usage` near 1.0 when `waiting` first went positive, or near 0? This
   single cross-reference — one CSV column against one line of source — is the
   whole teardown, and it is the sentence Part 8.4 asks for.
4. **What happens on preemption?** Find where a running request is pushed back.
   Note whether it is recompute or swap, and what happens to its blocks. Do not
   go deep — this is Week 7's and Week 10's material. Note the entry point and
   move on.

**Write:** `week3-notes.md` → "Weekend Teardown: what triggers queuing".

> **Bound your depth.** If question 2 or 4 does not resolve in one sitting,
> write down exactly what is unclear and stop. That is the plan's own rule, and
> Week 4 is a consolidation week — do not let this eat the Month 1 checkpoint.

---

<a name="part-8"></a>
# Part 8 — Deliverables, risks, and the one-line test

### 8.1 — Checklist

**Part 0 (before anything else — see the [Part 0 checklist](#part-0))**
- [x] W2-A PagedAttention notes finished
- [ ] W2-B TTFT stamping site found, five questions answered
- [ ] W2-C SECURITY-NOTES entry written, file renamed
- [ ] W2-E layout consolidated, README updated, Week 2 notes committed
- [ ] Flags verified from source (Part 4.7)
- [ ] Predictions pre-registered (8.2)

**Code (you write these, Part 4)**
- [ ] `week3-sweep.sh` — length × length × concurrency, warm-up, repeats, structured filenames, smoke mode
- [ ] `metrics-sampler.sh` — 4 Hz `/metrics` → CSV with name auto-discovery
- [ ] `gpu-sampler.sh` — explicitly named `nvidia-smi` columns
- [ ] `parse_results.py` — validated against Week 2 data first (Part 4.6)

**Data (committed)**
- [ ] `results/week3/*.json` — every run
- [ ] `results/week3/metric-names.txt` — **closes W2-D**
- [ ] `results/week3/metrics-idle-snapshot.txt` — the zero point for counters
- [ ] `results/week3/capacity-config.txt` — KV pool size, max concurrency, T1/T2 values
- [ ] `results/week3/metrics_full-session.csv`
- [ ] `results/week3/gpu_full-session.csv`
- [ ] `results/week3/server-startup.log`, `week3-sweep.log`, `MANIFEST.txt`

**Writing**
- [ ] `week3-notes.md` → Orca paper notes, one page, connected to your own numbers
- [ ] `week3-notes.md` → weekend teardown: what triggers queuing
- [ ] `week3-notes.md` → per-axis results tables with repeat spread shown
- [ ] `week3-notes.md` → **the memory-ceiling finding**: which limit binds first,
      at what concurrency, with the code-level reason (Part 6.3)
- [ ] `week3-notes.md` → "VRAM is the wrong ceiling signal" (Part 1.1), which
      reconciles Week 1's two apparently contradictory numbers
- [ ] `week3-notes.md` → pre-registration table with Measured/Verdict filled in
- [ ] `week3-notes.md` → G1–G5 each marked resolved or explicitly still open
- [ ] `week2-notes.md` → clear the "Carried forward to Week 3" block; fill in
      "Memory ceiling observations" with real numbers, as that block instructs
- [ ] `SECURITY-NOTES.md` → prefix-cache hit rate vs input length and concurrency
- [ ] `Readme.md` → status line updated to Week 3

**Carried into Week 4 (the Month 1 checkpoint)**
- [ ] Charts: TTFT / ITL / throughput vs concurrency, per axis
- [ ] Open questions, written down honestly

### 8.2 — Pre-register these predictions BEFORE the pod session {#p82}

<a name="p82"></a>

| # | Prediction | Predicted | Measured | Verdict |
|---|---|---|---|---|
| P1 | KV pool size from startup log | ~250k–270k tokens | | |
| P2 | `Maximum concurrency for 8,192 tokens` line | ~31× | | |
| P3 | Axis C concurrency-20 peak KV cache usage | ~6% | | |
| P4 | **G1:** concurrency-1 TTFT after warm-up | ~25 ms (down from 98 ms) | | |
| P5 | Axis D concurrency where `waiting > 0` first appears | 16 or below | | |
| P6 | KV cache usage at that first `waiting > 0` | LOW (<30%) → **T2 binds first, not T3** | | |
| P7 | Axis D concurrency where KV usage first exceeds 90% | ~36–48 | | |
| P8 | Any preemptions at all? | Yes, at concurrency 48 or 64 | | |
| P9 | **Axis E:** `mcr_ratio` at range-ratio 0 | ~2.0× | | |
| P10 | **Axis E:** `mcr_ratio` at range-ratio 0.5 | Well under 2.0× | | |
| P11 | Axis B vs axis A TTFT at concurrency 16 | B much larger; roughly tracks the 32× input-length ratio | | |
| P12 | Axis B ITL vs axis A ITL | B *elevated* — chunked prefill steals decode-step time (Part 1.5) | | |
| P13 | The ~35% column from Week 2 | `utilization.gpu`, not memory | | |
| P14 | Axis C concurrency-20 throughput vs Week 2's 1621 tok/s | Within ±5% | | |
| P15 | Prefix-cache hit rate on random prompts | Small but non-zero — the shared chat-template wrapper (Part 1.7) | | |

A wrong prediction with a recorded explanation is a stronger result than a right
one with no prediction. **P6 is the one that matters most**: it is the difference
between correctly identifying the token budget as the first binding constraint
and mistakenly reporting it as the memory ceiling.

### 8.3 — Risks and fallbacks

| Risk | Fallback |
|---|---|
| A flag is rejected by this build | Part 4.7 checks the source for free; Part 5 Step 5 catches it in 10 s. Set `DETAILED=0` or drop the flag. |
| Axis A concurrency-1 is slow (~4 min/run × 3) | Already cut to 24 prompts. Drop to `REPEATS=2` for axis A if time is tight. |
| Ceiling not reached at concurrency 64 | Part 5 Step 7 escalation. If still unreached, report *that*, with the mechanism — do not touch `gpu_memory_utilization`. |
| Cache usage reads high but nothing queues | Expected — that is prefix-cache retention (Part 1.7). Use `preemptions`, not the gauge, as the ceiling evidence. |
| Pod session overruns | Priority: **D > C > B > A > E**. |
| Sampler CSV size | 4 Hz × 90 min ≈ 21,600 rows ≈ 1.5 MB. Fine to commit. |
| Run-to-run variance swamps the repeats | That is a result. Report the spread; it retroactively bounds every Week 2 number, and Week 4's report must say so. |
| `nvidia-smi` rejects `-lms` | Use `-l 1`. The `/metrics` sampler is the one that matters. |
| Model revision hash changed since Week 1 | Stop. Note it in `ENVIRONMENT.md`, and decide explicitly whether to pin the old revision or re-baseline. Do not silently mix. |

### 8.4 — The one-line test

> **Week 3 succeeds if `week3-notes.md` can state, in one sentence backed by a
> CSV timestamp and a line of source code:**
>
> *"On this GPU with this model and this config, requests begin queuing at
> concurrency N because X binds first — and the KV cache does not run out until
> concurrency M."*

Everything in this document exists to make that one sentence true and
defensible.

### 8.5 — Why this feeds Week 4 directly

Week 4 is the Month 1 checkpoint: a reproducible benchmark script plus an
800–1500 word baseline report covering **five** metrics — TTFT, ITL, throughput,
p50/p95/p99, and **memory ceiling**.

Four of the five already exist from Week 2. **The memory ceiling is the one
missing metric, and Week 3 is the last data-gathering week before the
checkpoint.** If Week 3 does not produce a ceiling number — or a documented,
mechanism-backed reason why the ceiling was unreachable inside the locked
config — the Month 1 checkpoint cannot state all five, and it is the checkpoint,
not the understanding, that counts.

That is why axis D is the priority, and why the token-budget-versus-memory
distinction in Part 1.4 is the intellectual core of the week rather than a
technicality.
