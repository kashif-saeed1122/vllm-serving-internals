# Week 3 — Day-by-Day Execution Guide

**Companion to [`week3-plan.md`](./week3-plan.md).**

The two files do different jobs, and you need both:

| File | Answers |
|---|---|
| `week3-plan.md` | **Why.** The concepts (GQA, blocks, the three queuing triggers), the capacity arithmetic, the experiment design, the risk table. |
| **`week3-day-by-day.md`** (this file) | **What, where, and when.** Which day, which directory, which command, which URL to read, which line to write down. Plus a from-scratch explanation of every line of bash. |

When this file says *"see Plan Part 1.4"* it means section **Part 1.4** of
`week3-plan.md`. Read the concept there once; the daily steps here assume it.

---

## Contents

- [§A — How to use this document](#a)
- [§B — The file map: where everything goes](#b)
- [§C — Master source list](#c)
- **[Day 1 — Setup, housekeeping, clone the source](#day1)** (local, ~2h)
- **[Day 2 — Read Orca, write the paper note](#day2)** (local, ~2.5h)
- **[Day 3 — Finish the Week 2 teardown + security note](#day3)** (local, ~2h)
- **[Day 4 — Write `week3-sweep.sh`](#day4)** (local, ~3h)
- **[Day 5 — Write the samplers + parser, pre-register predictions](#day5)** (local, ~3h)
- **[Day 6 — POD SESSION](#day6)** (**paid**, ~2h)
- **[Day 7 — Analysis, teardown, write-up, commit](#day7)** (local, ~4h)
- [Appendix A — The sweep script, built from scratch in 8 versions](#appA)
- [Appendix B — How to read `/metrics` output](#appB)
- [Appendix C — Pod session quick-reference card](#appC)

---

<a name="a"></a>
## §A — How to use this document

**Days 1–5 and 7 cost nothing.** Only Day 6 spends money. That is the entire
point of the ordering: everything that can be built, read, or written without a
GPU is done first, so the paid session is 90 minutes of pure data collection
with zero thinking, zero debugging, and zero surprises.

**Do not reorder Days 1–5.** Day 4 needs the source clone from Day 1. Day 5's
parser validation needs Week 2's data. Day 6 needs Day 5's scripts. Day 2's Orca
reading needs the finished PagedAttention note (already done — see the
[W2-A section](./week3-plan.md#w2-a) of the plan).

**If a day runs long, cut scope inside that day, not by skipping a day.** The
one exception: Day 3 (W2-B, the TTFT stamping trace) can move to Day 7 if you
are short on time — it is the only prerequisite that is not load-bearing for the
pod session. Everything else is.

**Each day ends with a `Done when` checklist.** If you cannot tick every box, do
not start the next day — note what is unfinished and finish it first. The one
thing that will actually waste money is arriving at Day 6 with an untested
script.

---

<a name="b"></a>
## §B — The file map: where everything goes

This is the answer to *"where do I put what."* Memorise the two rules at the
bottom.

### Current state (verified, committed at `6ca8f3d`)

```
D:\dev\                                 <- GIT REPO ROOT
├─ .git\
├─ Readme.md                            <- status line still says "Week 1", fix on Day 1
├─ week2-sweep.log                      <- Week 2 raw console log
├─ week-wise-plan\
│  ├─ week3-plan.md                     <- the design doc (why)
│  └─ week3-day-by-day.md               <- THIS FILE (what/where/when)
├─ q1-vllm-serving\                     <- all code and notes live here
│  ├─ .gitignore                        <- ignores .env and week3-plan.md
│  ├─ .env                              <- pod URL / secrets. NEVER commit.
│  ├─ ENVIRONMENT.md                    <- locked stack fingerprint
│  ├─ Security-notes.md                 <- side-channel observation log
│  ├─ serve.sh                          <- launches vLLM on the pod
│  ├─ week1-notes.md
│  ├─ week2-notes.md
│  └─ week2-sweep.sh                    <- Week 2's benchmark script
└─ results\                             <- ALL DATA lives here, at repo root
   └─ week2\
      ├─ week2_conc1.json
      ├─ week2_conc2.json
      ├─ week2_conc5.json
      ├─ week2_conc10.json
      └─ week2_conc20.json
```

### What Week 3 adds, and exactly where

| New file | Path | Created on |
|---|---|---|
| Week 3 notes | `q1-vllm-serving\week3-notes.md` | **Day 2** |
| The sweep script | `q1-vllm-serving\week3-sweep.sh` | **Day 4** |
| Metrics sampler | `q1-vllm-serving\metrics-sampler.sh` | **Day 5** |
| GPU sampler | `q1-vllm-serving\gpu-sampler.sh` | **Day 5** |
| Results parser | `q1-vllm-serving\parse_results.py` | **Day 5** |
| All Week 3 data | `results\week3\` | **Day 6** (created on the pod) |
| vLLM source clone | **outside the repo** — e.g. `D:\src\vllm-0.28.0\` | **Day 1** |

### The two rules

> **Rule 1 — Code and notes go in `q1-vllm-serving\`. Data goes in
> `results\week3\` at the repo root.**
>
> This is not arbitrary tidiness: it is what Week 2 already did.
> `week2-sweep.sh` contains `--result-dir results/week2`, a *relative* path, and
> you ran it from `D:\dev`, so the data landed at `D:\dev\results\week2\`.
> Week 3 keeps the identical convention so one parser reads both weeks and Week
> 4's charts read one continuous dataset.

> **Rule 2 — Always run scripts from the repo root, with the folder in the path.**
>
> ```bash
> cd /d/dev                          # or  cd /workspace/<repo>  on the pod
> ./q1-vllm-serving/week3-sweep.sh
> ```
>
> **Not** `cd q1-vllm-serving && ./week3-sweep.sh` — that would write to
> `q1-vllm-serving\results\week3\`, splitting your data across two locations.
> If you ever see a `results` folder appear inside `q1-vllm-serving`, you broke
> Rule 2. Delete it and re-run from the root.

### The vLLM source clone goes OUTSIDE the repo

```bash
mkdir -p /d/src
git clone --depth 1 --branch v0.28.0 \
  https://github.com/vllm-project/vllm.git /d/src/vllm-0.28.0
```

Why outside: it is ~50 MB and tens of thousands of files that are not yours.
Cloning it inside `D:\dev` would either pollute `git status` with thousands of
untracked files or force a `.gitignore` entry and a nested-repo mess. Keep it at
`D:\src\vllm-0.28.0`. You read from it; you never commit it.

---

<a name="c"></a>
## §C — Master source list

Everything you need to read this week, in one place. Day sections link back here.

> Links were accurate as of writing. If a `docs.vllm.ai` URL 404s, the page was
> moved — search from **https://docs.vllm.ai/** rather than guessing a new path.
> The GitHub links are pinned to tag `v0.28.0`, so they will not drift.

### Papers

| Paper | Where | Use |
|---|---|---|
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., OSDI '22) | Landing page: https://www.usenix.org/conference/osdi22/presentation/yu · PDF: https://www.usenix.org/system/files/osdi22-yu.pdf | **Day 2 — this week's paper** |
| **Efficient Memory Management for LLM Serving with PagedAttention** (Kwon et al., SOSP '23) | https://arxiv.org/abs/2309.06180 | Already read (Week 2). Re-open §4 for the block table when writing Day 2's comparison. |
| **Sarathi-Serve** (chunked prefill) | https://arxiv.org/abs/2403.02310 | Week 5's paper. **Do not read this week.** Mentioned only because Day 7 may observe chunked-prefill behaviour in axis B. |

*Note: Orca is a Samsung/Seoul National University system and is **not** open
source. There is no repo to read — the paper is the only artifact. This matters
for Day 2: you cannot check Orca's claims against code, only against your own
measurements and against vLLM's implementation of the same ideas.*

### vLLM documentation

| Topic | URL |
|---|---|
| Docs root (start here if any link below breaks) | https://docs.vllm.ai/ |
| Architecture overview | https://docs.vllm.ai/en/latest/design/arch_overview.html |
| **Metrics design** — what each Prometheus metric means | https://docs.vllm.ai/en/latest/design/metrics.html |
| Optimization and tuning — `max_num_seqs`, `max_num_batched_tokens`, chunked prefill | https://docs.vllm.ai/en/latest/configuration/optimization.html |
| Engine args reference (`--max-model-len`, `--gpu-memory-utilization`, …) | https://docs.vllm.ai/en/latest/configuration/engine_args.html |
| Benchmark suite README (the `vllm bench serve` flags) | https://github.com/vllm-project/vllm/tree/v0.28.0/benchmarks |

### vLLM source paths (in your `D:\src\vllm-0.28.0\` clone)

Paths shift between versions. Each row gives the expected path **and** a `grep`
that finds it if the path is wrong. **Always trust the grep over the path.**

| What | Expected path | Find it with |
|---|---|---|
| Benchmark driver, `calculate_metrics()` | `vllm/benchmarks/serve.py` | `grep -rn "def calculate_metrics" vllm/` |
| **Where TTFT is stamped** (Day 3) | `vllm/benchmarks/lib/endpoint_request_func.py` | `grep -rln "ttft" vllm/benchmarks/` |
| **The scheduler** (Day 7) | `vllm/v1/core/sched/scheduler.py` | `grep -rn "class Scheduler" vllm/v1/` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `grep -rn "def allocate_slots" vllm/` |
| Block pool / free list | `vllm/v1/core/block_pool.py` | `grep -rn "def get_new_blocks" vllm/` |
| Metric definitions | `vllm/v1/metrics/loggers.py` | `grep -rn "cache_usage\|num_preemptions" vllm/v1/metrics/` |
| Scheduler config defaults | `vllm/config/scheduler.py` | `grep -rn "max_num_batched_tokens" vllm/config/` |

### Background reading (optional, high value)

| Topic | URL |
|---|---|
| Aleksa Gordić, *"Inside vLLM: Anatomy of a High-Throughput LLM Inference System"* | https://www.aleksagordic.com/blog/vllm |
| Anyscale, *"Continuous batching for LLM inference"* — the clearest plain-English explanation of Orca's core idea | https://www.anyscale.com/blog/continuous-batching-llm-inference |
| Model config (verify the GQA numbers yourself) | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/config.json |

### Reference material for the tooling

| Topic | Source |
|---|---|
| Prometheus text exposition format (what `/metrics` returns) | https://prometheus.io/docs/instrumenting/exposition_formats/ |
| `nvidia-smi` query field list — **authoritative, run it locally** | `nvidia-smi --help-query-gpu` |
| Bash: shell functions | https://www.gnu.org/software/bash/manual/bash.html#Shell-Functions |
| Bash: arrays | https://www.gnu.org/software/bash/manual/bash.html#Arrays |
| Bash: parameter expansion (`${VAR:-default}`) | https://www.gnu.org/software/bash/manual/bash.html#Shell-Parameter-Expansion |
| **ShellCheck** — paste a script in, it finds your quoting bugs | https://www.shellcheck.net/ |
| Bash Pitfalls — the 40 mistakes everyone makes | https://mywiki.wooledge.org/BashPitfalls |

---

<a name="day1"></a>
# Day 1 — Setup, housekeeping, clone the source

**Time:** ~2 hours · **Cost:** free · **Prerequisite:** none

**Goal:** end the day with a clean repo, the vLLM source readable locally, and
the model's real config numbers confirmed. Nothing conceptual — this is the day
that removes friction from every later day.

---

### Step 1.1 — Confirm your starting point (5 min)

```bash
cd /d/dev
git status
git log --oneline -3
```

You should see a clean tree at `6ca8f3d`. If there are uncommitted changes,
commit or stash them now — you want Week 3's diff to be legible.

---

### Step 1.2 — Clone the vLLM source (10 min)

```bash
mkdir -p /d/src
git clone --depth 1 --branch v0.28.0 \
  https://github.com/vllm-project/vllm.git /d/src/vllm-0.28.0
```

`--depth 1` fetches only the latest commit, not the full history — ~50 MB
instead of ~500 MB. `--branch v0.28.0` pins the **exact tag your pod runs**
(from `ENVIRONMENT.md`). This matters: reading `main` would show you code that
is not what produced your numbers.

Verify:

```bash
cd /d/src/vllm-0.28.0
git describe --tags          # should print v0.28.0
ls vllm/v1/core/             # should list block_pool.py, kv_cache_manager.py, sched/
```

**Why today and not Week 5** (where the Q1 plan schedules it): Day 3 needs it to
finish the Week 2 teardown, Day 4 needs it to verify benchmark flags without
spending pod time, and Day 7 needs it for the scheduler trace. It costs nothing
and unblocks three days.

---

### Step 1.3 — Verify the model config yourself (15 min)

Plan Part 2's entire capacity estimate rests on four numbers. Confirm them
rather than trusting the document.

**Option A — read the JSON directly in a browser** (no install needed):
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/config.json

**Option B — if you have `transformers` locally:**

```bash
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct-AWQ')
for k in ['num_hidden_layers','num_attention_heads','num_key_value_heads',
          'hidden_size','max_position_embeddings','vocab_size',
          'tie_word_embeddings']:
    print(f'{k:28} {getattr(c, k, None)}')
print(f'{\"head_dim (derived)\":28} {c.hidden_size // c.num_attention_heads}')
"
```

**Write these into `week3-notes.md` on Day 2.** Expected:

| Field | Expected | Why it matters |
|---|---|---|
| `num_hidden_layers` | 28 | multiplier in the KV formula |
| `num_attention_heads` | 28 | **NOT the one you want** |
| `num_key_value_heads` | **4** | GQA. Using 28 here overestimates KV by **7×** |
| `hidden_size` | 3584 | → `head_dim` = 3584 / 28 = 128 |
| `tie_word_embeddings` | **false** | embed + lm_head are separate → ~2.2 GiB of fp16 weights paid twice |

If `num_key_value_heads` is not 4, **stop and redo Plan Part 2's arithmetic** —
every prediction in Part 8.2 depends on it.

---

### Step 1.4 — Housekeeping: the three things that will bite at Week 4 (30 min)

Week 4 is the Month 1 checkpoint, whose requirement is *"independently
reproducible by a stranger."* Fix these now while they are cheap.

**(a) `Readme.md` status line is two weeks stale.** It currently says:

> 🚧 Week 1 — environment verified, benchmarking harness not yet built.

The harness exists and has produced a full 5-point sweep. Replace with something
like:

```markdown
## Status
🚧 Week 3 (Month 1 — Baseline). Week 1 environment verified; Week 2 benchmark
harness built and first concurrency sweep captured (TTFT, ITL, throughput,
p50/p95/p99 at concurrency 1/2/5/10/20). Week 3 in progress: prompt-length ×
output-length × concurrency sweep and the KV-cache memory ceiling.

## Results
- [Week 1 notes](./q1-vllm-serving/week1-notes.md) — environment, wrong-model incident
- [Week 2 notes](./q1-vllm-serving/week2-notes.md) — PagedAttention notes, first real sweep
- [Raw data](./results/) — `vllm bench serve` JSON output, one file per configuration
```

The README is the first thing a stranger reads. A stale status line makes an
active repo look abandoned.

**(b) Decide the naming convention and write it down.** The Q1 plan says
`NOTES.md` and `SECURITY-NOTES.md`; you have `week1-notes.md`,
`week2-notes.md`, `Security-notes.md`. Per-week files are **better** — easier to
read cold, and they show progression. So keep them, but make the two standing
documents uppercase so they stand out from the weekly ones:

```bash
cd /d/dev/q1-vllm-serving
git mv Security-notes.md SECURITY-NOTES.md
```

**A case-only rename needs `git mv`.** Windows filesystems are
case-insensitive, so renaming in Explorer or with `mv` changes nothing that git
can see. `git mv` updates the index explicitly.

Then add one line to the README so the convention is not a mystery:

> Notes are per-week (`week1-notes.md`, `week2-notes.md`, …).
> `ENVIRONMENT.md` and `SECURITY-NOTES.md` are standing documents maintained
> across the whole quarter.

**(c) Leave `results/` where it is.** Plan Part 0 W2-E suggested consolidating
it under `q1-vllm-serving/`. **Skip that.** Week 2's data is already committed at
`results/week2/`, and moving it now buys tidiness at the cost of breaking the
one convention Week 3 depends on (Rule 2 in §B). Consistency beats tidiness.
Just follow Rule 2 and always run from the repo root.

---

### Step 1.5 — Commit (5 min)

```bash
cd /d/dev
git add -A
git commit -m "week3 day 1: README status, SECURITY-NOTES rename, model config verified"
```

---

### Done when

- [ ] `git status` clean at start of day, and again after the Day 1 commit
- [ ] `/d/src/vllm-0.28.0` exists; `git describe --tags` prints `v0.28.0`
- [ ] `ls vllm/v1/core/` shows `block_pool.py`, `kv_cache_manager.py`, `sched/`
- [ ] The five model config numbers written down, `num_key_value_heads` confirmed as 4
- [ ] `Readme.md` status line says Week 3, with a Results section
- [ ] `SECURITY-NOTES.md` renamed via `git mv`
- [ ] Naming convention documented in the README in one line

---

<a name="day2"></a>
# Day 2 — Read Orca, write the paper note

**Time:** ~2.5 hours · **Cost:** free · **Prerequisite:** Day 1

**Goal:** one page of Orca notes in a new `week3-notes.md`, with every claim
connected to a number you measured yourself.

---

### Step 2.1 — Create `week3-notes.md` with its skeleton (10 min)

**Path:** `q1-vllm-serving\week3-notes.md`

Create it now, with headings for the whole week, so every later day has a
labelled place to write and you never face a blank file:

```markdown
# Week 3 Notes

Goal: separate prefill-bound from decode-bound effects by sweeping prompt
length and output length independently, and find the KV-cache memory ceiling
for this GPU + model + config.

Plan: ../week-wise-plan/week3-plan.md
Daily guide: ../week-wise-plan/week3-day-by-day.md

---

## Model config (verified Day 1)
<!-- Day 1 Step 1.3 numbers go here -->

## Paper Notes: Orca
<!-- Day 2 -->

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
```

Paste your Day 1 config numbers under "Model config" now.

---

### Step 2.2 — Read Orca (75 min)

**Source:** https://www.usenix.org/system/files/osdi22-yu.pdf
(landing page: https://www.usenix.org/conference/osdi22/presentation/yu)

**Reading order — do not read front to back.** Read §1 (Introduction), then §3
(the two core mechanisms), then §2 (background) only if §3 leaves you stuck,
then skim §4–5 (distributed architecture — skip, single GPU) and §6 (evaluation
— skim the graph shapes only).

Read for these five things. **Nothing else this week.**

---

**① Iteration-level scheduling — the paper's headline contribution**

Find where the paper explains the problem with request-level batching. The
shape of the argument: if you batch 4 requests and one finishes at token 10
while another runs to token 500, the finished one's slot is idle for 490 steps,
and a newly arrived request waits for the whole batch to drain.

Orca's fix: make the scheduling decision **every iteration** (every forward
pass) instead of every request. Finished sequences leave the batch immediately;
waiting ones join immediately.

> **Connect it to your own number.** This is the mechanism behind Week 2's
> Finding 3: mean ITL rose only 8.826 ms → 12.033 ms (**+36%**) while output
> throughput rose 108.55 → 1621.29 tok/s (**~15×**) across a 20× concurrency
> increase. Write that sentence into the note. The paper describes a mechanism;
> you have the number it produces on your own hardware.

---

**② Selective batching — the part most people skip, and the one that matters**

This is *why* iteration-level scheduling is even possible.

The problem: when you batch sequences of different lengths, most operations are
fine — a Linear layer treats the batch as `[total_tokens, hidden]` and does one
big matrix multiply, and it does not care which token belongs to which sequence.
**Attention is different.** Each token must attend only to its *own* sequence's
KV, so you cannot flatten Attention the way you flatten a GEMM.

Orca's answer: **split the batch treatment by operation.** Batch the Linear /
LayerNorm ops across everything; process Attention per-sequence.

Find the figure that shows this split and copy its structure into your notes as
ASCII. Something like:

```
Batched (one big GEMM, sequence identity irrelevant):
  Linear / LayerNorm / activation
        [tok tok tok tok tok tok tok]  <- all sequences flattened together

NOT batched (must respect sequence boundaries):
  Attention
        [tok tok tok] [tok tok] [tok tok]   <- seq A    seq B    seq C
         each attends only to its own KV
```

> **Why this is the important one:** it is the reason ragged batches work at all,
> and therefore the reason your ITL barely moved while throughput went up 15×.
> Without selective batching you would have to pad every sequence to the longest
> one, and your concurrency-20 run would have been dramatically slower.

---

**③ The scheduler's admission decision**

Find where the paper describes how the scheduler picks the next batch, and what
makes a request wait. Note the knob Orca calls `max_bs` (max batch size).

> **This is Day 7's teardown target.** vLLM implements the same idea with more
> constraints — Plan Part 1.4's three triggers (`max_num_seqs`,
> `max_num_batched_tokens`, KV block exhaustion). Orca has essentially one.
> Note the difference; it is the gap the next four years of systems work filled.

---

**④ Orca's KV reservation scheme — and why it is the weak point**

Find where the paper explains how memory is reserved for the KV cache. Orca
reserves slots, and each slot must hold the *worst case*: `max_bs × max_tokens`.

Work out the waste on your own hardware and put the number in your notes:

```
Orca-style reservation, with your config:
  a slot must hold max-model-len = 8,192 tokens
  KV per token (Plan Part 1.2)  = 56 KiB
  -> a slot reserves 8,192 x 56 KiB = 448 MiB

Your Week 2 requests used 512 in + 256 out = 768 tokens
  -> actually used  768 x 56 KiB = 42 MiB
  -> WASTED         406 MiB per slot  = 90.6% of the reservation

With a ~14 GiB pool: ~32 Orca-style slots, versus ~330 concurrent
768-token sequences under PagedAttention (Plan Part 2.2).
```

> **That ~10× gap is the entire reason PagedAttention exists.** You now have both
> papers' central claims expressed as one number on your own hardware. This is
> the single most valuable paragraph you will write this week — it is exactly the
> kind of thing that reads as first-hand understanding rather than summary.

---

**⑤ What Orca does not solve**

One sentence, as the bridge forward. Candidate: *Orca schedules at iteration
granularity but still reserves memory at request granularity — it fixed the
time axis and left the memory axis for PagedAttention.*

---

### Step 2.3 — Write the note (45 min)

Into the `## Paper Notes: Orca` section of `week3-notes.md`. Structure:

```markdown
## Paper Notes: Orca

**Paper:** Orca: A Distributed Serving System for Transformer-Based
Generative Models — Yu et al., OSDI '22
https://www.usenix.org/conference/osdi22/presentation/yu
(Not open source — the paper is the only artifact.)

### The problem: request-level batching wastes the time axis
### Contribution 1 — Iteration-level scheduling
  (+ my Week 2 number: ITL +36% while throughput +15x)
### Contribution 2 — Selective batching
  (+ the ASCII split diagram)
### How Orca's scheduler admits requests
  (+ contrast: Orca has ~1 constraint, vLLM has >=3 -- Plan Part 1.4)
### Orca's KV reservation, and its 90.6% waste on my config
  (+ the arithmetic above)
### What Orca does not solve -> the bridge to PagedAttention
### Where this shows up in Week 3
  (Prediction: axis D's queuing onset is a scheduler decision, not a
   memory event -- Plan Part 8.2, P6)
```

**Stop at one page.** The plan's rule is explicit: *"one paper, one page of
notes, then stop. Mastery is not required."* If you are still reading at the
2.5-hour mark, write down what you did not get to and stop.

---

### Done when

- [ ] `q1-vllm-serving\week3-notes.md` exists with the full skeleton
- [ ] Day 1's model config numbers pasted in
- [ ] Orca note written, one page, covering all five items
- [ ] The ITL/throughput number from Week 2 quoted inside the note
- [ ] The 448 MiB / 42 MiB / 90.6% waste arithmetic written out
- [ ] Committed

---

<a name="day3"></a>
# Day 3 — Finish the Week 2 teardown + the security note

**Time:** ~2 hours · **Cost:** free · **Prerequisite:** Day 1 (the clone)

**Goal:** close the two Week 2 items that are still open — W2-B and W2-C from
[Plan Part 0](./week3-plan.md#part-0).

---

### Step 3.1 — Find where TTFT is actually stamped (60 min)

**Why this is not optional.** Your Week 2 note is honest and correct:
`calculate_metrics()` is an *aggregator* — it reads `outputs[i].ttft`, values
already recorded. You wrote *"in a different function, the async request sender
— not yet read."*

The question Week 2 asked was *where TTFT is stamped*. That is still unread.
And it matters for Week 3 specifically: **Day 7 compares a prefill-bound axis
against a decode-bound axis and attributes the difference to the engine.** That
attribution is only valid if you know what the client-side timer wraps —
network round-trip, HTTP overhead, tokenizer time, `asyncio` event-loop delay.

```bash
cd /d/src/vllm-0.28.0

# 1. Which file records ttft?  (trust this over any path I give you)
grep -rln "ttft" vllm/benchmarks/

# 2. Find the sender for the backend you used: --backend openai-chat
grep -rn "openai_chat" vllm/benchmarks/

# 3. In that file: every place ttft is ASSIGNED, not read
grep -n "ttft" vllm/benchmarks/lib/endpoint_request_func.py

# 4. The clock. This is the key line.
grep -n "perf_counter\|monotonic\|time.time" vllm/benchmarks/lib/endpoint_request_func.py
```

Then open the function and read it top to bottom. Answer these five questions in
`week2-notes.md`, replacing the "not yet read" sentence:

| # | Question | What to look for |
|---|---|---|
| 1 | **Which clock?** | `time.perf_counter()` is monotonic and high-resolution; `time.time()` is wall-clock and can jump if NTP adjusts. Which is used, and does it matter for a 15-second run? |
| 2 | **Where is the start timestamp taken** — before or after the HTTP request is dispatched? | Everything after that point is inside your TTFT. If it is before `session.post(...)`, your TTFT includes connection setup. |
| 3 | **What trips the "first token arrived" stamp?** | Find the streaming loop and a guard like `if ttft == 0.0:`. **Critically: can an empty or role-only delta chunk trip it?** OpenAI-compatible streams often send `{"role":"assistant"}` with no content first. If so, your TTFT is time-to-first-*chunk*, not time-to-first-*token*. |
| 4 | **Is `itl` appended once per chunk or once per token?** | If a chunk can carry >1 token, your "inter-token latency" is really *inter-chunk* latency. Write down which. |
| 5 | **What does `latency` (used for E2EL) span?** | Start stamp to last chunk? Or to stream close? |

**Question 3 is the one that could change how you report your numbers.** If TTFT
stamps on a contentless chunk, then TTFT slightly *understates* true
time-to-first-token, uniformly across all runs — which is fine for comparing
axes but must be stated.

**Write it as an update, not a rewrite.** Keep your existing
`calculate_metrics()` finding and the `max_concurrent_requests` resolution —
both are good work. You are completing them.

> **Bound your depth.** If Q3 or Q4 does not resolve in one sitting, write down
> exactly what is unclear and stop. That is the plan's own rule.

---

### Step 3.2 — Write the SECURITY-NOTES entry (20 min)

Week 2's "To write" list included: *"log anything noticed about whether
concurrency level changes the visibility of the prefix-cache signal."*

You have no data, because `/metrics` was never sampled during the sweep. The
honest entry is a **negative result plus a method note** — which is a legitimate
log entry, and it sets up Week 3's measurement.

Append to `q1-vllm-serving\SECURITY-NOTES.md`:

```markdown
- [Week 2] [Prefix caching — NOT measured, method gap identified] — The Week 2
  concurrency sweep (1/2/5/10/20 at 512-in/256-out, 100 prompts each) produced
  no prefix-cache data: `/metrics` was never sampled during the runs. The sweep
  used `vllm bench serve --dataset-name random`, which generates unique random
  token IDs per prompt, so cross-request prefix sharing should be near zero —
  EXCEPT that `--backend openai-chat` wraps every prompt in the same chat
  template, making the first ~20-30 tokens identical across every request in the
  sweep. Since PagedAttention shares in whole blocks of 16 tokens, that wrapper
  alone should share 1-2 blocks across all 100 requests, giving a small non-zero
  hit rate even with fully random prompts. Untested prediction.
  Method gap: the engine log's 10-second rolling window cannot resolve per-run
  cache state (same limitation hit in Week 1's concurrency test). Continuous
  `/metrics` sampling is required — built in Week 3 as `metrics-sampler.sh`.
  Still observation-only; no timing-based inference attempted.
```

Note the reasoning chain here — random prompts, *but* a shared chat template,
*but* block-granular sharing so it must exceed 16 tokens to count. That is the
Week 2 PagedAttention note doing work. That is what a paper note is for.

---

### Step 3.3 — Commit (5 min)

```bash
cd /d/dev
git add -A
git commit -m "week3 day 3: complete W2 teardown (TTFT stamping site), W2 prefix-cache security note"
```

---

### Done when

- [ ] The "not yet read" placeholder in `week2-notes.md` is gone
- [ ] All five TTFT questions answered (or explicitly marked unresolved)
- [ ] The exact clock function and the two stamp sites named, with file and line
- [ ] `SECURITY-NOTES.md` has the Week 2 entry
- [ ] Committed

---

<a name="day4"></a>
# Day 4 — Write `week3-sweep.sh`

**Time:** ~3 hours · **Cost:** free · **Prerequisite:** Day 1

**Goal:** a working, dry-run-verified sweep script — and an actual understanding
of every line in it.

> ### Read [Appendix A](#appA) first — all of it.
>
> Appendix A builds this script in **8 versions**, starting from
> `week2-sweep.sh` (which you already understand) and adding one idea at a time,
> explaining every piece of bash syntax where it first appears: what a function
> is, what `$1` means, what `local` does, what arrays are for, what
> `${VAR:-default}` does, why quoting matters.
>
> **Do not skip to the final script.** The whole reason Day 4 exists as its own
> day is so the tooling is understood rather than pasted. If the `bench`
> function did not make sense before, Appendix A is the fix.

---

### Step 4.1 — Work through Appendix A (90 min)

Type each version out. Do not copy-paste. Typing V4 → V5 → V6 is what makes the
function click.

---

### Step 4.2 — Save the final script (15 min)

**Path:** `q1-vllm-serving\week3-sweep.sh` (Appendix A, Version 8)

```bash
cd /d/dev/q1-vllm-serving
# create the file, then:
chmod +x week3-sweep.sh
bash -n week3-sweep.sh      # syntax check only, runs nothing
```

`bash -n` parses the file and reports syntax errors without executing anything.
A clean `bash -n` means the shell can *read* your script; it says nothing about
whether the logic is right. That is what Step 4.3 is for.

---

### Step 4.3 — DRY RUN: read all 50 commands without a GPU (45 min)

**This is the most valuable half-hour of the week.** Version 8 has a `DRYRUN`
mode that prints each `vllm bench serve` command it *would* run, instead of
running it. It works on the ThinkPad with no vLLM installed and no pod running.

```bash
cd /d/dev
DRYRUN=1 ./q1-vllm-serving/week3-sweep.sh | tee /tmp/week3-dryrun.txt
wc -l /tmp/week3-dryrun.txt
```

Now actually audit the output:

```bash
# How many real (non-warmup) runs?  Expect 50.
grep -c "save-result" /tmp/week3-dryrun.txt

# Every distinct input/output length pair -- should be exactly 4 pairs
grep -o "random-input-len [0-9]* --random-output-len [0-9]*" /tmp/week3-dryrun.txt \
  | sort | uniq -c

# Every filename, in order
grep -o "result-filename [^ ]*" /tmp/week3-dryrun.txt

# Any duplicate filenames?  MUST be empty -- a duplicate silently overwrites data.
grep -o "result-filename [^ ]*" /tmp/week3-dryrun.txt | sort | uniq -d
```

**Expected run count:**

| Axis | Configs | Repeats | Runs |
|---|---|---|---|
| A | 4 | 3 | 12 |
| B | 4 | 3 | 12 |
| C | 6 | 3 | 18 |
| D | 5 | 1 | 5 |
| E | 2 | 1 | 2 |
| G1 re-run | 1 | 1 | 1 |
| **Total** | | | **50** |

**Then read the dry-run output line by line.** Pick three lines at random and
check each against Plan Part 3.2's axis table by hand. If line 1 of axis A does
not read `--random-input-len 128 --random-output-len 1024 --max-concurrency 1
--num-prompts 24`, you have a bug — and you have found it for free, on Day 4,
instead of discovering it at minute 40 of a paid session.

**The duplicate-filename check is the important one.** If two runs write the same
filename, the second silently overwrites the first, and you lose data you paid
for and will not notice until Day 7.

---

### Step 4.4 — Run it through ShellCheck (15 min)

Paste the script into https://www.shellcheck.net/ (or `shellcheck week3-sweep.sh`
if installed). It will catch unquoted variables and other classic bash bugs.

Fix everything it flags except deliberate choices — and for each one you keep,
add a comment saying why. ShellCheck error codes are documented at
https://www.shellcheck.net/wiki/ — read the wiki page for anything you do not
understand rather than blindly silencing it.

---

### Step 4.5 — Verify the flags exist in YOUR version (15 min)

The single most likely way to waste a paid session is a flag that does not exist
in v0.28.0. You have the source. Check.

```bash
cd /d/src/vllm-0.28.0
grep -rn "save-detailed\|save_detailed"           vllm/benchmarks/
grep -rn "random-range-ratio\|random_range_ratio" vllm/benchmarks/
grep -rn "random-prefix-len\|random_prefix_len"   vllm/benchmarks/
grep -rn '"--temperature"'                        vllm/benchmarks/
grep -rn '"--base-url"'                           vllm/benchmarks/
grep -rn '"--percentile-metrics"'                 vllm/benchmarks/
```

For each: confirm it exists, and **read its `help=` string and `default=`.**

**Two need more than existence-checking:**

**`--random-range-ratio`** — do not trust the help text. Find the code that
*consumes* it and confirm the sampled range:

```bash
grep -rn "range_ratio" vllm/benchmarks/datasets.py
```

You are looking for whether the range is `[len × (1−r), len × (1+r)]` or
something else, and what the default is. **Axis E's entire interpretation
depends on this** — if the default is already non-zero, then Week 2's runs were
*not* fixed-length, and your `max_concurrent_requests` explanation needs
revisiting.

**`--random-prefix-len`** — a *deliberately shared* prefix prepended to every
random prompt. You are not using it in the sweep, but note it in
`SECURITY-NOTES.md`: it is the exact knob for measuring prefix-cache hit rate as
a function of shared-prefix length — the natural next step for the side-channel
thread you have tracked since Week 1. Keep it as a spare experiment for Day 6 if
the session runs short (three runs at `--random-prefix-len 0 / 256 / 1024`).

While you are in the source, confirm the metric names Day 5's sampler needs:

```bash
grep -rn "cache_usage\|num_preemptions\|prefix_cache" vllm/v1/metrics/
```

Write the exact strings down. This is what makes Day 5's auto-detection a
formality rather than a gamble — and it is what closes Week 2's gap G2.

---

### Done when

- [ ] Appendix A worked through, versions typed not pasted
- [ ] `q1-vllm-serving\week3-sweep.sh` exists, `chmod +x`, `bash -n` clean
- [ ] `DRYRUN=1` produces **50** `--save-result` lines
- [ ] **Zero duplicate `--result-filename` values**
- [ ] Three dry-run lines hand-checked against Plan Part 3.2
- [ ] ShellCheck clean, or every remaining warning has a comment explaining why
- [ ] All six flags confirmed present in v0.28.0
- [ ] `--random-range-ratio` semantics and default read **from the consuming code**
- [ ] Exact `/metrics` names written down for Day 5
- [ ] Committed

---

<a name="day5"></a>
# Day 5 — Write the samplers + parser, pre-register predictions

**Time:** ~3 hours · **Cost:** free · **Prerequisite:** Day 4

**Goal:** all four scripts finished and tested, and every prediction written
down *before* any data exists.

---

### Step 5.1 — `metrics-sampler.sh` (45 min)

**Path:** `q1-vllm-serving\metrics-sampler.sh`

**Why this script is the week's most important new artifact.** You have hit the
same wall three times and documented it each time:

> *Week 1:* "the log's rolling window missed the live `Running: 5 reqs` moment on the first attempt"
> *Week 1:* "engine log stats are 10-second rolling-window snapshots, not instantaneous rates"
> *Week 2:* "pane 3's `/metrics | grep cache` step was missed/incomplete"

The engine log emits a snapshot every ~10 seconds. **Week 3's entire deliverable
is a sub-second event** — the moment `waiting` first goes above 0, and what
`kv_cache_usage` reads at that same instant. The log is structurally incapable of
answering it. This replaces the log with 4 Hz sampling into a timestamped CSV.

Full script with per-line explanation is in Plan Part 4.3. The parts to
understand:

**What `/metrics` actually returns** — see [Appendix B](#appB). Prometheus text
format: comment lines starting `#`, then one line per metric with optional
labels in `{}` and the value last.

**Metric-name auto-detection.** Names differ between vLLM v0 and v1, and Week 2
got stuck precisely because the name was never found. So: probe a list of
candidates, print which one matched, and **exit loudly if none match**:

```bash
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

**The design rule: it must not be able to silently succeed while recording
nothing.** That is exactly what happened in Week 2.

**One fetch per row, parsed six times** — not six fetches:

```bash
snap=$(curl -s --max-time 2 "$BASE_URL/metrics" || true)
ts=$(date +%s.%N)
get () {
  printf '%s\n' "$snap" \
    | awk -v k="$1" 'index($0,k)==1 && substr($0,1,1)!="#" {print $NF; exit}'
}
```

- **Six separate `curl` calls would give six different moments in one CSV row.**
  The values would not be mutually consistent — which destroys the whole point,
  because the finding *is* the correlation between `waiting` and
  `kv_cache_usage` at one instant.
- `|| true` — a transient curl failure records a row of empty fields instead of
  killing the sampler (`set -e` is on). You want a gap in the data, not a dead
  sampler discovered an hour later.
- `--max-time 2` — a hung request cannot stall the loop.
- `date +%s.%N` — Unix seconds plus nanoseconds. Same clock base as the sweep
  log's timestamps, so the two files join.
- `index($0,k)==1` — "the line *starts with* this key". Safer than `grep`, which
  would also match `..._total` variants or a substring inside a `# HELP` line.
- `$NF` — awk's "last field". The metric value is always last.
- `exit` — stop at the first match.

**Why 4 Hz?** A scheduler step at concurrency 20 is ~12 ms (your measured ITL),
so 250 ms is ~20 steps — too coarse to see individual steps, but 40× finer than
the log, and fine enough to catch a queuing episode lasting a fraction of a
second. Cost: ~21,600 rows over 90 minutes ≈ 1.5 MB. Cheap to commit.

**Test it locally without a server:**

```bash
bash -n q1-vllm-serving/metrics-sampler.sh
# then confirm the FATAL path fires (nothing is listening on :8000):
./q1-vllm-serving/metrics-sampler.sh /tmp/x.csv ; echo "exit=$?"
```

You want it to print the FATAL message and exit non-zero. A script that fails
loudly when it cannot do its job is the whole design goal.

---

### Step 5.2 — `gpu-sampler.sh` (20 min)

**Path:** `q1-vllm-serving\gpu-sampler.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: gpu-sampler.sh <outfile.csv>}"
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
  --format=csv,nounits \
  -lms 500 > "$OUT"
```

**This exists to fix one specific Week 2 failure.** Your note reads: *"saw ~35%
baseline, spiking to 37-38%, with brief spikes toward 100% — TODO confirm which
column."* Naming every field in `--query-gpu=` makes the ambiguity impossible to
repeat: `--format=csv` writes a header row.

**The three fields, and the trap:**

| Field | What it actually is |
|---|---|
| `memory.used` | MiB of VRAM allocated. Pinned near 90% of 24 GB from server start; **barely moves**. Useless as a load signal (Plan Part 1.1). |
| `utilization.gpu` | **Percent of time** a kernel was running. Not memory. |
| `utilization.memory` | **Percent of time** the memory bus was read/written. **NOT the fraction of VRAM in use.** The most commonly misread `nvidia-smi` field. |

**Authoritative reference:** `nvidia-smi --help-query-gpu` — run it on the pod
and save the output. It documents every field, and it is the source that settles
arguments.

**You can resolve the Week 2 ambiguity right now, without a GPU:** ~35% cannot
have been `memory.used` as a percentage, because with
`gpu_memory_utilization=0.90` that value is a flat ~90%. So it was
`utilization.gpu` — which means the observation **does** confirm the
prefill/decode split: compute mostly idle (~35%) during decode
(memory-bandwidth-bound, waiting on weight reads), spiking toward 100% during
prefill (compute-bound GEMMs). Confirm it with a labelled column on Day 6, but
write the inference down now.

**Why `power.draw` and `clocks.sm` are in the query:** if the Week 2
concurrency-1 anomaly was GPU clock ramp-up from an idle power state (the G1
hypothesis), `clocks.sm` is where you would see it. The G1 re-run on Day 6 plus
this column is a direct test of that hypothesis.

If `nvidia-smi` rejects `-lms`, fall back to `-l 1` (1-second). Coarser, but the
`/metrics` sampler is the one that matters for the ceiling.

---

### Step 5.3 — `parse_results.py` (45 min)

**Path:** `q1-vllm-serving\parse_results.py`

Structure and code in Plan Part 4.5. Four jobs:

1. Reconstruct experiment coordinates from the filename.
2. Pull the ~40 scalar metrics out of each JSON.
3. Group repeats and aggregate.
4. Emit a flat CSV (for Week 4's charts) plus markdown tables (for the notes).

**The filename regex is the core**, and it is why Day 4's filenames encode all
five coordinates:

```python
W3 = re.compile(
    r"^(?P<tag>[A-Za-z0-9]+)_in(?P<inlen>\d+)_out(?P<outlen>\d+)"
    r"_c(?P<conc>\d+)_rrr(?P<rrr>[\d.]+)_r(?P<rep>\d+)$"
)
W2 = re.compile(r"^week2_conc(?P<conc>\d+)$")   # so Week 2 data still parses
```

Keeping the Week 2 pattern is what lets you validate the parser (Step 5.4) and
what lets Week 4 build one continuous dataset across both weeks.

**Use `raw.get(key)`, never `raw[key]`.** Week 2's JSONs have no `e2el` fields —
you did not request them — so `raw["mean_e2el_ms"]` would crash on exactly the
data you are validating against.

**Two unit traps that will silently corrupt your table:**

- Summary fields (`mean_ttft_ms`) are in **milliseconds**. The per-request arrays
  from `--save-detailed` (`ttfts`) are in **seconds**. Multiply by 1000 when
  pooling, or your table mixes units by 1000×.
- Counters in `/metrics` are **cumulative since server boot**. `preemptions`,
  `prefix_hits`, `prefix_queries` must be *differenced* across a run window, not
  read directly. That is what Day 6's idle snapshot is the zero point for.

**Three derived columns that do real analytical work:**

```python
# 1. The Week 2 artifact ratio, computed rather than eyeballed.
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
pooled_ttft = [v * 1000 for r in group for v in r.get("ttfts", [])]
```

That third one is why `--save-detailed` is in the sweep at all. Without it you
can only average each run's *own* p99 — the mean of three near-maxima of 100
samples, which is not a p99 of 300 samples (Plan Part 1.6). Have the parser
report **which method it used**, so the notes can state it honestly.

---

### Step 5.4 — Validate the parser against Week 2's data (20 min)

**Do this before Day 6. It is free and it is the only chance to check the parser
against data whose right answer you already know.**

```bash
cd /d/dev
python q1-vllm-serving/parse_results.py results/week2 --out /tmp/w2check
cat /tmp/w2check.md
```

It must reproduce the table already in `week2-notes.md`, cell for cell:

| conc | TTFT mean | ITL mean | req/s | out tok/s |
|---|---|---|---|---|
| 1 | 98.41 | 8.826 | 0.424 | 108.55 |
| 2 | 24.36 | 8.846 | 0.874 | 223.65 |
| 5 | 32.10 | 9.092 | 2.117 | 542.04 |
| 10 | 44.21 | 10.017 | 3.826 | 979.55 |
| 20 | 71.39 | 12.033 | 6.333 | 1621.29 |

If any cell disagrees, the parser is wrong. Fix it now, for free, rather than on
a metered pod with fresh data you cannot sanity-check.

**Bonus: this also independently confirms your own Week 2 teardown.** The
`mcr_ratio` column should come out as exactly `2.0` at all five concurrency
levels. You predicted that from reading the source; now arithmetic over the data
confirms it at every point. Worth a sentence in the notes — a source-derived
prediction, then confirmed numerically.

---

### Step 5.5 — Pre-register the predictions (30 min)

**Copy Plan Part 8.2's table into `week3-notes.md` under "Predicted capacity",
and fill in the Predicted column before any data exists.**

Why this matters and is not ceremony: a measured number alone is just a number.
A measured number with a prediction attached is a *finding* — and a wrong
prediction with a recorded explanation is a stronger result than a right one
with no prediction, because it proves you had a model and updated it.

**P6 is the one that matters most:**

> *At the moment `waiting` first exceeds 0 on axis D, `kv_cache_usage` will be
> LOW (<30%) — proving the per-step token budget (T2) bound first, not memory
> (T3).*

That is the difference between correctly identifying the token budget as the
first binding constraint and mistakenly reporting it as the memory ceiling. If
you have not written that prediction down beforehand, you will very likely see
`waiting > 0` on Day 6 and write "memory ceiling reached at concurrency 16."

---

### Step 5.6 — Commit (5 min)

```bash
cd /d/dev
git add -A
git commit -m "week3 day 5: metrics/gpu samplers, results parser (validated vs week2), predictions pre-registered"
```

---

### Done when

- [ ] Four scripts exist in `q1-vllm-serving\`, all `chmod +x`, all `bash -n` clean
- [ ] `metrics-sampler.sh` exits with the FATAL message when no server is up
- [ ] Parser reproduces the Week 2 table **exactly**
- [ ] `mcr_ratio` comes out as 2.0 at all five Week 2 levels
- [ ] Predictions table in `week3-notes.md`, **Predicted column filled**
- [ ] Pod readiness confirmed: the corrected Week 1 custom template still exists,
      and the 25 GB volume `q1-serving` still has the model cached
- [ ] Committed and **pushed** (you will pull this on the pod)

---

<a name="day6"></a>
# Day 6 — POD SESSION

**Time:** ~2 hours · **Cost: THE ONLY PAID DAY** · **Prerequisite:** Days 4 and 5 complete

**Goal:** collect data. Nothing else. No thinking, no debugging, no analysis.

> **Print or open [Appendix C](#appC) before you boot.** It is the one-page
> command sequence. This section explains *why* each step exists; Appendix C is
> what you actually follow while the meter runs.

**Golden rules:**
1. **Do not analyse on the pod.** Analysis is free on the ThinkPad. Every minute
   spent reading numbers is a minute paid for nothing.
2. **Do not edit scripts on the pod.** If something needs editing, you skipped
   Day 4 or 5. Fix, push, pull.
3. **Terminate, do not stop, when finished.** A stopped pod may still bill.

---

### Time budget

| Phase | Time |
|---|---|
| Boot, pull repo, model load | 5–8 min |
| Verify model + revision hash | 2 min |
| Start both samplers | 1 min |
| **Capture metric names (Step 6.4)** | 3 min |
| Smoke test | 1 min |
| Warm-up (inside the sweep) | 3 min |
| Axis A × 3 | ~32 min |
| Axis B × 3 | ~13 min |
| Axis C × 3 | ~8 min |
| Axis D × 1 | ~12 min |
| Axis E + G1 re-run | ~6 min |
| Pull artifacts off, verify | 5 min |
| **Total** | **~95 min** |

---

### Step 6.1 — Boot and get your code onto the pod (8 min)

Deploy from the **corrected custom template you saved in Week 1** — the one with
the model baked into the container start command. That template exists precisely
because Week 1's stock template silently served Qwen3-0.6B.

```bash
cd /workspace
git clone https://github.com/kashif-saeed1122/vllm-serving-internals.git
cd vllm-serving-internals
```

**This is the repo root on the pod.** Rule 2 from §B applies here exactly as on
the ThinkPad: stay in this directory for everything.

```bash
mkdir -p results/week3
chmod +x q1-vllm-serving/*.sh
```

---

### Step 6.2 — Launch the server and CAPTURE THE STARTUP LOG (5 min)

```bash
./q1-vllm-serving/serve.sh 2>&1 | tee results/week3/server-startup.log
```

**The `| tee` is not optional.** The startup log contains the capacity numbers
Week 3 is supposed to report, and they scroll past in seconds.

- `2>&1` merges stderr into stdout — vLLM logs to stderr, so without this the
  log file would be nearly empty.
- `tee` writes to the file **and** to your screen, so you can watch it load.

In a second pane, once it says the server is up:

```bash
grep -Ei "KV cache size|Maximum concurrency|max_num_seqs|max_num_batched_tokens|Graph capturing|chunked prefill|block_size" \
  results/week3/server-startup.log | tee results/week3/capacity-config.txt
```

**This one file is the week's headline deliverable.** It gives you:

| Line | What it settles |
|---|---|
| `GPU KV cache size: N tokens` | The true pool size → validates or corrects Plan Part 2.1's ~250–270k estimate |
| `Maximum concurrency for 8,192 tokens per request: Mx` | **The ceiling, computed by vLLM itself** |
| `max_num_seqs` | **T1** from Plan Part 1.4 |
| `max_num_batched_tokens` | **T2** — the value you need to interpret axis D at all |
| `block_size` | Confirms the 16 assumed in Plan Part 1.3 |

**Paste `capacity-config.txt` verbatim into `week3-notes.md`.** Do not
paraphrase it.

---

### Step 6.3 — Verify the model AND the revision hash (2 min)

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool \
  | tee results/week3/models.json
```

Confirm `Qwen/Qwen2.5-7B-Instruct-AWQ`. Week 1's entire incident was a template
serving a different model than the config claimed. **Never trust the logs; ask
the API.**

**Then check the revision hash against `ENVIRONMENT.md`:**
`b25037543e9394b818fdfca67ab2a00ecc7dd641`

```bash
grep -Ei "revision|commit|snapshot" results/week3/server-startup.log
ls /workspace/*/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/ 2>/dev/null
```

**If the hash differs, stop and decide before spending 90 minutes.** A new
Hugging Face revision means your "locked stack" is not locked and Week 3 would
be measured against a different model than Week 2 — making axis C's
reproducibility check meaningless. The end-of-quarter checklist requires
`ENVIRONMENT.md` be *"accurate, unchanged since Week 1"*. This step enforces it.

---

### Step 6.4 — Capture the metric names. DO NOT SKIP. (3 min)

**This is the step Week 2 missed. It is the reason you have no memory-ceiling
data.**

```bash
curl -s localhost:8000/metrics | grep -E '^# HELP vllm:' | sort \
  > results/week3/metric-names.txt
curl -s localhost:8000/metrics > results/week3/metrics-idle-snapshot.txt
nvidia-smi --help-query-gpu > results/week3/nvidia-smi-fields.txt
wc -l results/week3/metric-names.txt
```

**Three files, each for a distinct reason:**

- **`metric-names.txt`** — the catalogue of what is observable on this stack.
  Commit it. You will use it in Month 2's source tracing and Month 3's
  perturbation work.
- **`metrics-idle-snapshot.txt`** — every metric's value **at idle, before any
  load**. This is your zero point. Counters are cumulative since boot, so a
  mid-sweep counter reading is meaningless without it.
- **`nvidia-smi-fields.txt`** — the authoritative field documentation, saved so
  the Week 2 column ambiguity can never recur.

Confirm these exist and note their exact spellings:
`num_requests_running`, `num_requests_waiting`, a KV/GPU cache-usage gauge,
`num_preemptions_total`, `prefix_cache_hits_total` / `prefix_cache_queries_total`,
`request_queue_time_seconds`, `request_prefill_time_seconds`,
`request_decode_time_seconds`, `iteration_tokens_total`.

> **Note `request_queue_time_seconds` specifically.** It is the server's *own*
> measurement of queuing delay — a direct cross-check on your client-side TTFT.
> If TTFT rises and `queue_time` rises with it, the delay is scheduler queuing.
> If TTFT rises and `queue_time` stays flat, it is prefill compute. That is
> exactly the distinction axis B is trying to establish, and the server will just
> tell you.

---

### Step 6.5 — Start both samplers (1 min)

```bash
# pane 2
./q1-vllm-serving/gpu-sampler.sh results/week3/gpu_full-session.csv

# pane 3
./q1-vllm-serving/metrics-sampler.sh results/week3/metrics_full-session.csv
```

**Confirm pane 3 prints `kv cache metric = <name>`.** If it printed FATAL
instead, find the right name in `metric-names.txt` and restart:

```bash
M_CACHE=<the-real-name> ./q1-vllm-serving/metrics-sampler.sh results/week3/metrics_full-session.csv
```

**Leave both running for the whole session.** One continuous timeline is easier
to work with than restarting per run, because the sweep prints a UTC timestamp
before each run and you slice afterwards. Restarting per run risks missing
exactly the transition you care about.

---

### Step 6.6 — Smoke test (1 min)

```bash
SMOKE=1 ./q1-vllm-serving/week3-sweep.sh
```

One ~10-second benchmark whose only job is to prove every flag is accepted by
this build. Day 4 Step 4.5 should have made this a formality — but a formality
that costs 10 seconds and eliminates a 90-minute failure mode is worth keeping.

On failure: `DETAILED=0 ./q1-vllm-serving/week3-sweep.sh`, or delete the
offending flag.

---

### Step 6.7 — Run the sweep (~75 min)

```bash
./q1-vllm-serving/week3-sweep.sh 2>&1 | tee results/week3/week3-sweep.log
```

**What to watch, and what to write down.** Pane 3's CSV is being written
regardless — but noting these live makes Day 7's analysis far faster.

| When | Watch for | Write down |
|---|---|---|
| Axis A/B/C | Nothing dramatic expected | If `waiting` ever exceeds 0, note the run name — that would contradict prediction P5 |
| **Axis D, all levels** | **First moment `waiting > 0`** | The concurrency level, **and `kv_cache_usage` at that same instant** ← *this is the week's finding* |
| Axis D | `running` plateauing at a round number | The number, and whether it equals `max_num_seqs` → **T1** |
| Axis D | `kv_cache_usage` climbing past 0.90 | The concurrency level → approaching **T3** |
| **Axis D** | **`preemptions` incrementing above its idle value** | The concurrency level → **T3 confirmed, the real ceiling** |
| G1 re-run at the end | Its TTFT vs Week 2's 98 ms | The mean TTFT |

**The one correlation that is the deliverable:** at the first `waiting > 0`, is
`kv_cache_usage` **low** or **high**?

- **Low** → the per-step token budget (T2) bound first. Not the memory ceiling.
- **High + preemptions** → genuine KV exhaustion (T3). The memory ceiling.

Per Plan Part 1.7, high cache usage *alone* does not prove the ceiling, because
freed blocks are retained in the prefix cache and are evictable on demand.
**`preemptions` is the honest signal.**

---

### Step 6.8 — If axis D at concurrency 64 shows no queuing (10 min, only if needed)

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

**Do NOT raise `--gpu-memory-utilization`. Do NOT raise `--max-model-len`.**
Those are Week 10 variables; touching them breaks the locked stack and
invalidates axis C's comparison to Week 2.

**If the ceiling still cannot be reached, that is the reportable finding**,
stated plainly: *"On this GPU/model/config the KV pool is large enough that
scheduler token-budget limits bind before memory does, up to concurrency N. The
memory ceiling was not reached and requires either longer context than
`max-model-len` permits or a reduced KV pool — deferred to Week 10."* An honest
negative result with a mechanism is a legitimate deliverable; the Q1 plan says
so explicitly.

---

### Step 6.9 — Extract everything, verify, THEN terminate (5 min)

```bash
# stop the samplers: Ctrl-C in panes 2 and 3

ls -la results/week3/ | tee results/week3/MANIFEST.txt
find results/week3 -name '*.json' | wc -l        # expect 50 (+1 if you did 6.8)
wc -l results/week3/metrics_full-session.csv     # expect ~20,000+
wc -l results/week3/gpu_full-session.csv

tar czf week3-results.tar.gz results/week3/
ls -lh week3-results.tar.gz
```

Then copy it off — `scp`, `runpodctl send`, or `rsync`.

**Verify the tarball opens on the ThinkPad and the JSON count is right BEFORE
terminating.** This is the last moment the data exists in only one place.

```bash
# on the ThinkPad
tar tzf week3-results.tar.gz | wc -l
```

**Then TERMINATE the pod — not stop.** Per the cost rule.

---

### Done when

- [ ] `results/week3/` on the ThinkPad with **50 JSON files**
- [ ] `capacity-config.txt` non-empty, with the KV-cache-size and max-concurrency lines
- [ ] `metric-names.txt` non-empty ← **closes Week 2's G2**
- [ ] `metrics-idle-snapshot.txt` saved (the counter zero point)
- [ ] `metrics_full-session.csv` has ~20,000+ rows
- [ ] `gpu_full-session.csv` non-empty with a header row naming every column
- [ ] `server-startup.log`, `week3-sweep.log`, `models.json`, `nvidia-smi-fields.txt`
- [ ] Model revision hash confirmed matching `ENVIRONMENT.md`
- [ ] **Pod TERMINATED**

---

<a name="day7"></a>
# Day 7 — Analysis, teardown, write-up, commit

**Time:** ~4 hours · **Cost:** free · **Prerequisite:** Day 6

---

### Step 7.1 — Build the tables (20 min)

```bash
cd /d/dev
tar xzf week3-results.tar.gz          # into results/week3/
python q1-vllm-serving/parse_results.py results/week3 --out results/week3/summary
cat results/week3/summary.md
```

Paste the per-axis tables into the corresponding `## Results — Axis X` sections
of `week3-notes.md`.

---

### Step 7.2 — Slice the sampler CSV per run (60 min)

`week3-sweep.log` has a `RUN <name> <UTC timestamp>` line before each run;
`metrics_full-session.csv` has Unix-epoch timestamps. Join them to get per-run
windows.

```bash
# every run's start marker
grep "^RUN " results/week3/week3-sweep.log

# convert one to epoch, then slice the CSV between two markers
date -u -d "2026-09-12T14:33:07Z" +%s
awk -F, '$1 >= 1789... && $1 <= 1789...' results/week3/metrics_full-session.csv
```

For each axis-D window, answer:

| Question | How |
|---|---|
| Peak KV cache usage | `max` of column 4 |
| Did anything queue? | `max` of column 3 (`waiting`) |
| **At first `waiting > 0`, what was `kv_cache_usage`?** | first row where col 3 > 0 → read col 4 · **T2 vs T3** |
| Did `running` plateau at a round number? | `max` of column 2 vs `max_num_seqs` · **T1** |
| Any preemptions? | col 5 **delta** across the window · **T3** |
| Prefix-cache hit rate | (col 6 delta) / (col 7 delta) |

**Remember: columns 5, 6, 7 are cumulative counters.** Difference them across the
window. `metrics-idle-snapshot.txt` is the zero point.

---

### Step 7.3 — The weekend teardown: trace the queuing trigger (60 min)

You now have both halves: an observed threshold from 7.2, and the source cloned
on Day 1.

```bash
cd /d/src/vllm-0.28.0
grep -rn "class Scheduler"     vllm/v1/core/sched/
grep -rn "def schedule"        vllm/v1/core/sched/scheduler.py
grep -rn "token_budget\|max_num_scheduled_tokens" vllm/v1/core/sched/scheduler.py
grep -rn "def allocate_slots"  vllm/v1/core/
grep -rn "def get_new_blocks"  vllm/v1/core/block_pool.py
grep -rni "preempt"            vllm/v1/core/sched/scheduler.py
```

Answer four questions, each anchored to a number from your own sweep:

1. **What is the waiting pool, concretely?** Find the data structure (deque?
   priority queue?) and the loop that drains it in `Scheduler.schedule()`.
2. **What exactly stops a waiting request from being admitted this step?** Find
   the token-budget variable, the running-request cap, and the call into the KV
   cache manager that can fail to return blocks. **There should be more than one
   `break`/`continue`, and each is a different trigger from Plan Part 1.4.**
   Map them explicitly: *this* `break` is T1, *that* one is T2, *that* one is T3.
3. **Which one fired in your axis-D run?** Your Step 7.2 answer — was
   `kv_cache_usage` near 1.0 when `waiting` first went positive, or near 0?
   **This single cross-reference, one CSV column against one line of source, is
   the whole teardown.**
4. **What happens on preemption?** Find where a running request is pushed back.
   Recompute or swap? What happens to its blocks? **Do not go deep** — this is
   Week 7's and Week 10's material. Note the entry point and move on.

> **Bound your depth.** If Q2 or Q4 does not resolve in one sitting, write down
> exactly what is unclear and stop. Week 4 is the Month 1 checkpoint — do not let
> this eat it.

---

### Step 7.4 — Write THE finding (45 min)

Into `## THE MEMORY CEILING FINDING`. **Not** "queuing started at concurrency N."
Write, with evidence:

> At concurrency **C**, `waiting` first exceeded 0 while `kv_cache_usage` was
> only **X%** — so the binding constraint was **T2, the per-step token budget
> `max_num_batched_tokens` = B** (`capacity-config.txt`, and
> `scheduler.py:LINE`), not memory. KV cache usage did not exceed 90% until
> concurrency **D**, and the first preemption occurred at concurrency **E**
> (`metrics_full-session.csv`, row NNNN). The memory ceiling for this
> GPU + model + config is therefore **~P tokens** of KV pool, which at 7,168
> tokens per sequence is **~S concurrent sequences** — matching / diverging from
> the startup log's stated `Maximum concurrency` of **M**.

Then write the remaining sections:

- **`## Why VRAM is the wrong ceiling signal`** — Plan Part 1.1. This reconciles
  Week 1's two apparently contradictory numbers (~87% VRAM reserved vs <1% cache
  usage). Include the `gpu_full-session.csv` evidence that `memory.used` is a
  flat line all session.
- **Axis A vs B** — does TTFT track input length? Does ITL stay independent of
  it? If axis B's ITL is *elevated*, that is chunked prefill stealing decode-step
  time (Plan Part 1.5) — a preview of Sarathi-Serve.
- **Axis C vs Week 2** — within ±5%? (P14.) **If not, that is a variance finding
  that qualifies the entire Week 2 table, and Week 4's Month 1 report must say
  so.**
- **G1** — did the concurrency-1 anomaly disappear? Check `clocks.sm` in the GPU
  CSV for the clock-ramp hypothesis specifically.
- **Axis E** — did `mcr_ratio` drop below 2.0 at range-ratio 0.5? Your
  source-derived prediction, confirmed or refuted.
- **`SECURITY-NOTES.md`** — prefix-cache hit rate vs input length and
  concurrency, now from continuous sampling instead of Week 1's three spot
  checks. Was the shared chat-template prediction from Day 3 right?
- **`## Predictions: measured vs predicted`** — fill in the Measured and Verdict
  columns for all 15. **State the misses plainly.**
- **`## Open questions carried into Week 4`** — honest list.

---

### Step 7.5 — Update the standing documents and commit (30 min)

```bash
cd /d/dev
```

- **`week2-notes.md`** — clear the "⚠ Carried forward to Week 3 pod session"
  block at the top, and fill in the "Memory ceiling observations" entry with the
  real numbers, exactly as that block instructs. Note the resolved
  `nvidia-smi` column question there too.
- **`Readme.md`** — status line to Week 3 complete; add Week 3 to Results.
- **`ENVIRONMENT.md`** — only if the revision hash changed. Otherwise leave it
  untouched; "unchanged since Week 1" is the requirement.

```bash
git add -A
git commit -m "week3: length x concurrency sweep, memory ceiling finding, scheduler teardown, Orca notes"
git push
```

---

### Done when

- [ ] All five axis result tables in `week3-notes.md`
- [ ] `capacity-config.txt` pasted verbatim
- [ ] **THE MEMORY CEILING FINDING written, naming which of T1/T2/T3 bound
      first, with both a CSV row and a source line as evidence**
- [ ] "Why VRAM is the wrong ceiling signal" written, with GPU-CSV evidence
- [ ] Teardown written; the three `break`/`continue` sites mapped to T1/T2/T3
- [ ] All 15 predictions have Measured and Verdict filled in
- [ ] G1, axis E, axis C-vs-Week-2, and axis A-vs-B each answered
- [ ] `SECURITY-NOTES.md` updated with the prefix-cache measurement
- [ ] `week2-notes.md` carried-forward block cleared
- [ ] `Readme.md` updated
- [ ] Committed and pushed
- [ ] **`results/week3/` committed** — data is the deliverable

---

### The one-line test

> **Week 3 succeeded if `week3-notes.md` states, in one sentence backed by a CSV
> timestamp and a line of source code:**
>
> *"On this GPU with this model and this config, requests begin queuing at
> concurrency N because X binds first — and the KV cache does not run out until
> concurrency M."*

### Why this feeds Week 4

Week 4 is the Month 1 checkpoint: a reproducible benchmark script plus an
800–1500 word baseline report covering **five** metrics — TTFT, ITL, throughput,
p50/p95/p99, and **memory ceiling**.

Four of the five already exist from Week 2. **The memory ceiling is the one
missing metric, and Week 3 is the last data-gathering week before the
checkpoint.** If Week 3 does not produce a ceiling number — or a documented,
mechanism-backed reason why the ceiling was unreachable inside the locked
config — the Month 1 checkpoint cannot state all five. And it is the checkpoint,
not the understanding, that counts.

---

<a name="appA"></a>
# Appendix A — The sweep script, built from scratch in 8 versions

**This appendix exists because you said the `bench` function did not make
sense.** So we start from the script you already wrote and understand, and add
one idea at a time. Every piece of bash syntax is explained where it first
appears.

**Type each version. Do not paste.** Typing V4 → V5 is what makes it click.

---

## V1 — What you already have (`week2-sweep.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/week2

for c in 1 2 5 10 20; do
  echo "=== concurrency $c ==="
  vllm bench serve \
    --backend openai-chat \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --ignore-eos \
    --num-prompts 100 \
    --max-concurrency "$c" \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --seed 42 \
    --save-result \
    --result-dir results/week2 \
    --result-filename "week2_conc${c}.json"
done
```

### Everything in V1, explained

**`#!/usr/bin/env bash`** — the *shebang*. When you run `./script.sh`, the OS
reads the first line to learn which interpreter to use. `/usr/bin/env bash`
means "find `bash` on the `PATH`" rather than hardcoding `/bin/bash`, which is
more portable across Linux distros and macOS.

**`set -euo pipefail`** — four safety switches. Without them, bash's defaults
are dangerously forgiving:

| Switch | Without it | With it |
|---|---|---|
| `-e` | A failing command is ignored and the script continues | The script exits on the first failure |
| `-u` | A typo'd variable expands to an empty string, silently | Referencing an unset variable is an error |
| `-o pipefail` | `a \| b` reports only `b`'s exit status, so a failing `a` is invisible | A failure anywhere in the pipeline fails the whole pipeline |

Why `-u` matters concretely: if you typed `--result-dir $OUTDIRR` (two Rs), then
without `-u` bash expands it to nothing and `vllm` writes results somewhere
unexpected. With `-u` the script stops and tells you.

**`mkdir -p results/week2`** — `-p` means "create parent directories as needed,
and do not error if it already exists". Safe to run every time.

**`for c in 1 2 5 10 20; do ... done`** — a loop. The variable `c` takes each
listed value in turn, and the body runs once per value.

**`$c` vs `"$c"`** — `$c` inserts the variable's value. **The quotes matter.**
If a value ever contained a space, unquoted `$c` would split into two separate
arguments. Here the values are digits so it cannot happen — but quote by default
and you never have to think about it. This is the single most common bash bug
(see https://mywiki.wooledge.org/BashPitfalls).

**`"week2_conc${c}.json"`** — the braces separate the variable name from the
text around it. `"$c.json"` happens to work, but `"$cjson"` would look for a
variable named `cjson`. **Always use `${c}` when text follows the name.**

**The trailing `\`** — a line continuation. One logical command spread over
many lines for readability. **There must be no space or character after the
backslash**, or bash sees a literal backslash and breaks the command. This is
the #1 typo in multi-line shell commands.

---

## V2 — Why V1's shape does not work for Week 3

V1 varies **one** thing: concurrency. Everything else is hardcoded.

Week 3 needs **five** things to vary: a tag, input length, output length,
concurrency, and prompt count. And here is the crucial part:

> **They do not vary independently. They come in fixed *sets*.**

Look at Plan Part 3.2. Axis A is *always* 128 in / 1024 out. Axis B is *always*
4096 in / 32 out. You want:

```
A  128  1024   1   24
A  128  1024   4   48
B  4096   32   1  100
...
```

**You cannot express that with nested loops.** Nested loops give you the
*cross product* — every input length crossed with every output length crossed
with every concurrency. That would be 4 inputs × 4 outputs × 6 concurrencies ×
3 repeats = 288 runs, most of them combinations you never wanted (like 4096 in /
1024 out at concurrency 40, which would take forever and answer nothing).

So the question becomes: **how do you say "run this exact command 50 times with
50 specific sets of values"?**

---

## V3 — The obvious answer, and why it is wrong

Copy-paste the whole 22-flag command 50 times, changing the numbers:

```bash
vllm bench serve --backend openai-chat --model Qwen/... \
  --random-input-len 128 --random-output-len 1024 \
  --max-concurrency 1 --num-prompts 24 ...        # 22 lines

vllm bench serve --backend openai-chat --model Qwen/... \
  --random-input-len 128 --random-output-len 1024 \
  --max-concurrency 4 --num-prompts 48 ...        # 22 lines again

# ... 49 more times
```

Three reasons this fails:

1. **~1,100 lines.** Nobody can read it, and a stranger certainly cannot (which
   the Week 4 checkpoint explicitly requires).
2. **Changing one shared flag means 50 edits.** On Day 4 you add
   `--temperature 0` (Plan Part 3.4). That is 50 edits with 50 chances to miss
   one — and a single missed one silently makes that run non-comparable to the
   other 49. You would not notice until Day 7, if ever.
3. **Filenames drift from contents.** Nothing forces
   `--result-filename A_in128_out1024_c1...json` to match the `--random-input-len
   128` on the line above. One copy-paste slip and a file is labelled wrong.
   You then analyse it as the wrong configuration and reach a false conclusion.

**What you want:** the 22 flags written *once*, with the 5 varying values
supplied per call. That is exactly what a function is.

---

## V4 — Introducing the function

### What a shell function is

A **function** is a named block of commands. You define it once and then call it
by name, like any other command.

```bash
greet () {
  echo "hello"
}

greet          # prints: hello
greet          # prints: hello
```

`greet () { ... }` defines it. `greet` on its own line calls it. That is all.

### Passing values in: `$1`, `$2`, `$3`

Functions take arguments the same way commands do — by position. Inside the
function, the first argument is `$1`, the second `$2`, and so on.

```bash
greet () {
  echo "hello $1, you are $2"
}

greet Kashif 30       # prints: hello Kashif, you are 30
#     ^^^^^^ ^^
#       $1   $2
```

**There are no named parameters in bash.** No `greet(name, age)`. Just
positions. This is why order matters absolutely, and why the next trick exists.

### Why `local tag=$1 inlen=$2 ...`

`$1` and `$2` are unreadable in a 22-line body. Is `$4` the concurrency or the
prompt count? So the first line of the function **renames the positional
arguments to meaningful names**:

```bash
bench () {
  local tag=$1 inlen=$2 outlen=$3 conc=$4 prompts=$5 rep=$6
  # from here on:  $inlen  not  $2
}
```

That single line is just **six ordinary assignments written on one line.** It is
identical to:

```bash
local tag=$1
local inlen=$2
local outlen=$3
local conc=$4
local prompts=$5
local rep=$6
```

Nothing clever. It is a legend for the positional arguments.

### What `local` actually does

Without `local`, a variable assigned inside a function is **global** — it
outlives the call and is visible everywhere.

```bash
f () { x=inside; }
x=outside
f
echo "$x"        # prints: inside     <-- f overwrote your variable
```

With `local`, the variable exists only for the duration of the call:

```bash
f () { local x=inside; }
x=outside
f
echo "$x"        # prints: outside    <-- untouched
```

**Why this matters here concretely:** the sweep calls `bench` 50 times. If
`inlen` were global, then a call that somehow did not set it would silently
inherit the *previous* call's value — and you would get a run benchmarked at the
wrong input length, saved under a filename claiming the right one. You would
never catch it. `local` makes that impossible.

**Rule: always use `local` for every variable inside a function.** There is no
downside.

### V4 in full

```bash
bench () {
  local tag=$1 inlen=$2 outlen=$3 conc=$4 prompts=$5 rep=$6
  local name="${tag}_in${inlen}_out${outlen}_c${conc}_r${rep}"

  echo "=== RUN ${name} ==="
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len "$inlen" \
    --random-output-len "$outlen" \
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
    --result-filename "${name}.json"
}
```

**`local name="${tag}_in${inlen}_..."`** builds the filename **from the same
variables that build the command.** This is the fix for V3's problem #3: the
filename *cannot* disagree with the flags, because both come from one source. A
run named `A_in128_out1024_c16_r2.json` is *guaranteed* to have run at input
128, output 1024, concurrency 16, repeat 2.

**`"$(( SEED + rep ))"`** — `$(( ))` is **arithmetic expansion**: bash evaluates
the maths inside and substitutes the result. With `SEED=42` and `rep=2`, this
becomes `44`. Note that variables inside `$(( ))` do **not** need a `$` prefix —
that is a quirk of arithmetic context.

Why vary the seed per repeat: repeats should sample **run-to-run variance**, not
re-run the byte-identical workload. Same seed three times would give three
near-identical results and tell you nothing about variance — which is the whole
reason for repeating (Plan Part 1.6).

---

## V5 — The call table: this is where it clicks

Now the 50 runs become 50 short lines that read like a spreadsheet:

```bash
#      tag  in    out   conc  prompts  rep
bench  A    128   1024  1     24       0
bench  A    128   1024  4     48       0
bench  A    128   1024  16    96       0
bench  A    128   1024  32    128      0

bench  B    4096  32    1     100      0
bench  B    4096  32    4     100      0
bench  B    4096  32    16    100      0
bench  B    4096  32    32    100      0

bench  C    512   256   1     40       0
bench  C    512   256   2     100      0
bench  C    512   256   5     100      0
bench  C    512   256   10    100      0
bench  C    512   256   20    100      0
bench  C    512   256   40    100      0
```

**That comment line is the header row.** The columns line up with the `local`
line in V4:

```
local tag=$1  inlen=$2  outlen=$3  conc=$4  prompts=$5  rep=$6
bench  A       128       1024       1        24          0
```

**Read `bench A 128 1024 16 96 0` as: "axis A, 128 input tokens, 1024 output
tokens, concurrency 16, 96 prompts, repeat 0."**

### Trace one call by hand — do this, it is the whole point

`bench A 128 1024 16 96 0` with `SEED=42`, `BASE_URL=http://localhost:8000`,
`OUTDIR=results/week3` expands to exactly:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 1024 \
  --ignore-eos \
  --temperature 0 \
  --num-prompts 96 \
  --max-concurrency 16 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result \
  --result-dir results/week3 \
  --result-filename A_in128_out1024_c16_r0.json
```

Substitutions that happened:

| In the function | Became | Because |
|---|---|---|
| `"$inlen"` | `128` | `inlen=$2`, and the 2nd argument is `128` |
| `"$outlen"` | `1024` | `$3` |
| `"$conc"` | `16` | `$4` |
| `"$prompts"` | `96` | `$5` |
| `"$(( SEED + rep ))"` | `42` | `42 + 0` |
| `"${name}.json"` | `A_in128_out1024_c16_r0.json` | built from the same variables |

**Do this by hand for one more line before continuing.** `bench B 4096 32 32 100
1` — write out what it becomes. If you can do that, you understand the function,
and V6–V8 are small additions.

---

## V6 — Optional arguments: `${7:-0}`

Axis E needs a 7th value, `--random-range-ratio` (Plan Part 3.3). But the other
49 calls should not have to type a value they do not care about.

```bash
local rrr=${7:-0}
```

**`${7:-0}` means: "use `$7` if it was given; otherwise use `0`."**

The general form is `${VAR:-default}` — one of bash's *parameter expansions*
(https://www.gnu.org/software/bash/manual/bash.html#Shell-Parameter-Expansion).

```bash
bench A 128 1024 16 96 0            # 6 args -> $7 unset -> rrr = 0
bench E 512 256 20 100 0 0.5        # 7 args ->            rrr = 0.5
```

**This is the same trick as the config block at the top of the script:**

```bash
REPEATS="${REPEATS:-3}"
AXES="${AXES:-A B C D E}"
```

"Use the environment variable `REPEATS` if the caller set one; otherwise 3."
That is what makes this work without editing the file:

```bash
REPEATS=1 AXES="D" ./q1-vllm-serving/week3-sweep.sh
```

On a metered pod, being able to re-run a subset without opening an editor is
worth a lot.

---

## V7 — Arrays: making one flag conditional

`--save-detailed` should be togglable (in case v0.28.0 does not have it, or you
want smaller files). You cannot do this:

```bash
DETAIL_FLAG=""                        # WRONG
vllm bench serve ... $DETAIL_FLAG
```

Because when `DETAIL_FLAG` is empty, unquoted `$DETAIL_FLAG` may vanish
correctly *or* — if quoted — pass an **empty string as an argument**, which
`vllm` sees as a real (invalid) argument and rejects. Getting "present or
entirely absent" right requires an **array**.

### What a bash array is

```bash
fruits=(apple banana cherry)     # create, 3 elements
fruits+=(date)                   # append -> 4 elements
echo "${fruits[@]}"              # expand to all elements
```

**`"${fruits[@]}"` is the important form.** Quoted with `[@]`, it expands to each
element as a **separate word**, correctly, and to **nothing at all** when the
array is empty. Compare:

| Form | 3 elements | Empty array |
|---|---|---|
| `"${a[@]}"` | 3 separate arguments ✅ | zero arguments ✅ |
| `"${a[*]}"` | **1** argument, space-joined ❌ | 1 empty argument ❌ |
| `${a[@]}` | 3 arguments, but word-split on spaces ❌ | zero arguments |

**Always `"${array[@]}"`.** The other two forms are bugs waiting to happen.

### Using it — and building the whole flag list as an array

This is the version to actually write, because it unlocks the dry-run mode:

```bash
bench () {
  local tag=$1 inlen=$2 outlen=$3 conc=$4 prompts=$5 rep=$6 rrr=${7:-0}
  local name="${tag}_in${inlen}_out${outlen}_c${conc}_rrr${rrr}_r${rep}"

  local args=(
    --backend openai-chat
    --base-url "$BASE_URL"
    --model "$MODEL"
    --endpoint /v1/chat/completions
    --dataset-name random
    --random-input-len "$inlen"
    --random-output-len "$outlen"
    --random-range-ratio "$rrr"
    --ignore-eos
    --temperature 0
    --num-prompts "$prompts"
    --max-concurrency "$conc"
    --request-rate inf
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,95,99
    --seed "$(( SEED + rep ))"
    --save-result
    --result-dir "$OUTDIR"
    --result-filename "${name}.json"
  )

  if [[ "$DETAILED" == "1" ]]; then
    args+=(--save-detailed)
  fi

  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf 'vllm bench serve'
    printf ' %s' "${args[@]}"
    printf '\n'
    return 0
  fi

  echo ""
  echo "=================================================================="
  echo "RUN ${name}   $(date -u +%FT%TZ)"
  echo "=================================================================="
  vllm bench serve "${args[@]}"
  sleep "$SETTLE"
}
```

**Why build the flags as an array instead of writing them inline:** because now
the same list can be either *printed* or *executed*. That is what makes
`DRYRUN=1` possible — and `DRYRUN=1` is what lets you verify all 50 commands on
the ThinkPad, for free, with no vLLM installed and no pod running (Day 4
Step 4.3). **That single capability is worth more than the rest of the script.**

**`return 0`** exits the function early (like `return` in any language) with
success status. `exit` would kill the whole script; `return` just ends this call.

**`$(date -u +%FT%TZ)`** — `$( )` is **command substitution**: run the command
and substitute its output. `date -u` gives UTC; `+%FT%TZ` formats as
`2026-09-12T14:33:07Z`.

> **This timestamp line is what makes the samplers usable.** Without it, the
> sampler CSV is one long undifferentiated timeline and you cannot say which run
> a `waiting > 0` event belongs to. One `echo` converts a useless CSV into a
> sliceable dataset. It is the cheapest high-value line in the whole script.

**`sleep "$SETTLE"`** — pause 5 seconds between runs so the engine drains and
the metrics settle. Without it, run N's tail contaminates run N+1's opening
samples, and your per-run windows overlap.

---

## V8 — The final script

Adds: the config block, the warm-up, axis gating, and the smoke test.

```bash
#!/usr/bin/env bash
# week3-sweep.sh - Week 3: prompt-length x output-length x concurrency sweep
#
# RUN FROM THE REPO ROOT:
#     cd /d/dev  (or  cd /workspace/vllm-serving-internals  on the pod)
#     ./q1-vllm-serving/week3-sweep.sh
#
# Usage:
#   ./q1-vllm-serving/week3-sweep.sh                 # full sweep, 50 runs
#   DRYRUN=1 ./q1-vllm-serving/week3-sweep.sh        # print commands, run nothing
#   SMOKE=1  ./q1-vllm-serving/week3-sweep.sh        # 1 tiny run, flag check
#   AXES="D" REPEATS=1 ./q1-vllm-serving/week3-sweep.sh   # subset
#
# Design rationale: week-wise-plan/week3-plan.md Part 3
# Line-by-line explanation: week-wise-plan/week3-day-by-day.md Appendix A
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
OUTDIR="${OUTDIR:-results/week3}"
REPEATS="${REPEATS:-3}"
AXES="${AXES:-A B C D E}"
SEED="${SEED:-42}"
DETAILED="${DETAILED:-1}"
SETTLE="${SETTLE:-5}"

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------- bench ----
bench () {
  local tag=$1 inlen=$2 outlen=$3 conc=$4 prompts=$5 rep=$6 rrr=${7:-0}
  local name="${tag}_in${inlen}_out${outlen}_c${conc}_rrr${rrr}_r${rep}"

  local args=(
    --backend openai-chat
    --base-url "$BASE_URL"
    --model "$MODEL"
    --endpoint /v1/chat/completions
    --dataset-name random
    --random-input-len "$inlen"
    --random-output-len "$outlen"
    --random-range-ratio "$rrr"
    --ignore-eos
    --temperature 0
    --num-prompts "$prompts"
    --max-concurrency "$conc"
    --request-rate inf
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,95,99
    --seed "$(( SEED + rep ))"
    --save-result
    --result-dir "$OUTDIR"
    --result-filename "${name}.json"
  )
  if [[ "$DETAILED" == "1" ]]; then
    args+=(--save-detailed)
  fi

  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf 'vllm bench serve'; printf ' %s' "${args[@]}"; printf '\n'
    return 0
  fi

  echo ""
  echo "=================================================================="
  echo "RUN ${name}   $(date -u +%FT%TZ)"
  echo "=================================================================="
  vllm bench serve "${args[@]}"
  sleep "$SETTLE"
}

# ------------------------------------------------------------ smoke test ----
if [[ "${SMOKE:-0}" == "1" ]]; then
  bench SMOKE 128 32 4 8 0
  echo "SMOKE OK - all flags accepted. Re-run without SMOKE=1."
  exit 0
fi

# --------------------------------------------------------------- warm-up ----
# Closes Week 2's G1. Week 2's concurrency=1 run had the HIGHEST TTFT of the
# sweep (98ms), breaking an otherwise clean monotonic trend, with a TIGHT
# distribution (std ~13ms) -- so the whole run was slow, not a few outliers.
# Hypothesis: one-time cost paid by whichever run went first (CUDA graph
# capture for an unseen batch shape, or GPU clock ramp from an idle state).
# These runs have NO --save-result: they are deliberately discarded.
if [[ "${DRYRUN:-0}" != "1" ]]; then
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
fi

# ------------------------------------------------------------- the sweep ----
for rep in $(seq 0 $(( REPEATS - 1 ))); do

  # Axis A - decode-bound: tiny prefill, long generation
  if [[ " $AXES " == *" A "* ]]; then
    #      tag  in    out   conc  prompts  rep
    bench  A    128   1024  1     24       "$rep"
    bench  A    128   1024  4     48       "$rep"
    bench  A    128   1024  16    96       "$rep"
    bench  A    128   1024  32    128      "$rep"
  fi

  # Axis B - prefill-bound: long prompt, near-zero generation
  if [[ " $AXES " == *" B "* ]]; then
    bench  B    4096  32    1     100      "$rep"
    bench  B    4096  32    4     100      "$rep"
    bench  B    4096  32    16    100      "$rep"
    bench  B    4096  32    32    100      "$rep"
  fi

  # Axis C - Week 2 anchor: identical levels, plus one new point at 40.
  # NOTE: --temperature 0 is a deliberate delta from Week 2 (Plan Part 3.4).
  if [[ " $AXES " == *" C "* ]]; then
    bench  C    512   256   1     40       "$rep"
    bench  C    512   256   2     100      "$rep"
    bench  C    512   256   5     100      "$rep"
    bench  C    512   256   10    100      "$rep"
    bench  C    512   256   20    100      "$rep"
    bench  C    512   256   40    100      "$rep"
  fi
done

# Axis D - ceiling hunt. 6144+1024 = 7168 tok/seq, under max-model-len 8192
# (the chat template adds ~20-30 tokens; 8192 exactly would overflow).
# Predicted KV pool ~250-270k tokens -> KV exhaustion around ~36 concurrent.
# Single pass; repeat only the level where something interesting happens.
if [[ " $AXES " == *" D "* ]]; then
  for c in 8 16 32 48 64; do
    p=$(( c * 2 )); (( p < 32 )) && p=32
    #      tag  in    out   conc  prompts  rep  rrr
    bench  D    6144  1024  "$c"  "$p"     0
  done
fi

# Axis E - tests the Week 2 prediction that the max_concurrent_requests 2x
# artifact should weaken once request lengths vary and waves desync.
if [[ " $AXES " == *" E "* ]]; then
  bench  E    512   256   20    100      0    0.0
  bench  E    512   256   20    100      0    0.5
fi

# G1 - concurrency=1 again, LAST, after the GPU has been hot for an hour.
# Identical config to axis C's conc-1 run but executed last instead of first.
# If Week 2's 98ms was a cold-start artifact, this returns near ~25ms.
bench  G1RERUN  512  256  1  40  0

echo ""
echo "SWEEP COMPLETE $(date -u +%FT%TZ)"
```

### The last two pieces of syntax

**`for rep in $(seq 0 $(( REPEATS - 1 )))`** — `seq 0 2` prints `0 1 2`, so with
`REPEATS=3` the loop runs with `rep` = 0, 1, 2.

**`if [[ " $AXES " == *" A "* ]]`** — a glob (wildcard) match, not a string
equality test. Inside `[[ ]]`, an **unquoted** right-hand side is treated as a
pattern, where `*` matches anything.

The spaces are the trick. `AXES` is `"A B C D E"`, and `" $AXES "` becomes
`" A B C D E "` — with padding spaces at both ends. The pattern `*" A "*` looks
for `space-A-space` anywhere in it.

Why the padding matters:

| `AXES` value | Padded | Does `*" A "*` match? |
|---|---|---|
| `A B C` | `" A B C "` | ✅ yes — `" A "` is present |
| `D` | `" D "` | ❌ no |
| `AB` | `" AB "` | ❌ no — correctly rejects a substring match |

Without the padding spaces, `AXES="AB"` would match axis `A`, and you would
silently run the wrong experiments.

**`p=$(( c * 2 )); (( p < 32 )) && p=32`** — set `p` to twice the concurrency,
then bump it to 32 if that came out below 32. `(( ))` is arithmetic *evaluation*
(as a test), while `$(( ))` is arithmetic *expansion* (produces a value).
`A && B` runs `B` only if `A` succeeded — and in arithmetic context, "succeeded"
means the expression was non-zero, i.e. true.

`p = 2 × concurrency` means every axis-D run processes exactly two "waves" —
enough to reach steady state without paying for 100 long-context requests at
every level.

---

## Quick reference: every piece of bash in the script

| Syntax | Name | Means |
|---|---|---|
| `#!/usr/bin/env bash` | shebang | which interpreter to use |
| `set -euo pipefail` | shell options | fail fast, fail loud |
| `f () { ... }` | function definition | a named block of commands |
| `$1 $2 $3` | positional parameters | the 1st, 2nd, 3rd argument |
| `local x=$1` | local variable | scoped to this call only |
| `${VAR:-default}` | parameter expansion | `$VAR` if set, else `default` |
| `${7:-0}` | same, for an optional arg | `$7` if given, else `0` |
| `"${x}"` | braces + quotes | separates the name from adjacent text; prevents word-splitting |
| `$(( a + b ))` | arithmetic expansion | produces a computed number |
| `(( a < b ))` | arithmetic evaluation | used as a true/false test |
| `$( cmd )` | command substitution | run `cmd`, substitute its output |
| `arr=(a b c)` | array literal | create an array |
| `arr+=(d)` | array append | add an element |
| `"${arr[@]}"` | array expansion | each element as its own argument |
| `[[ x == *y* ]]` | pattern match | glob match, not string equality |
| `a && b` | AND list | run `b` only if `a` succeeded |
| `cmd > /dev/null` | redirect | discard stdout |
| `2>&1` | redirect | merge stderr into stdout |
| `\` at line end | line continuation | one command, many lines. **No trailing space.** |
| `return 0` | return | exit the function (not the script) |

**When something does not make sense:**
- ShellCheck: https://www.shellcheck.net/ — paste it in, it explains the bug
- Bash manual: https://www.gnu.org/software/bash/manual/bash.html
- Bash Pitfalls: https://mywiki.wooledge.org/BashPitfalls

---

<a name="appB"></a>
# Appendix B — How to read `/metrics` output

`curl -s localhost:8000/metrics` returns **Prometheus text exposition format**
(spec: https://prometheus.io/docs/instrumenting/exposition_formats/). It looks
like this:

```
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 20.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 0.0
```

**The structure:**

```
vllm:num_requests_running{engine="0",model_name="Qwen/..."} 20.0
^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^
metric name               labels (in braces, optional)      value (always last)
```

- Lines starting `#` are `HELP` (description) or `TYPE` (metric kind) — metadata,
  not data. **This is why the sampler's awk skips `substr($0,1,1)!="#"`.**
- The **value is always the last whitespace-separated field** — which is why the
  sampler uses awk's `$NF` ("number of fields" = the last one).
- Labels vary between deployments, so you cannot match on the full line. You
  match on the line **starting with** the metric name — awk's
  `index($0,k)==1`.

**Metric types, and why it matters for analysis:**

| Type | Behaviour | Examples | How to read it |
|---|---|---|---|
| **gauge** | current value, goes up and down | `num_requests_running`, `num_requests_waiting`, cache-usage | Read directly |
| **counter** | monotonically increasing since boot | `num_preemptions_total`, `prefix_cache_hits_total`, `..._queries_total` | **Difference across a window.** A raw reading is meaningless. |
| **histogram** | bucketed distribution | `request_queue_time_seconds`, `time_to_first_token_seconds` | Multiple `_bucket` lines plus `_sum` and `_count` |

**The counter rule is the one that will bite you.** `prefix_cache_hits_total`
reading `483,201` tells you nothing. Hits *during run X* =
`value at end of X − value at start of X`. That is why Day 6 Step 6.4 saves
`metrics-idle-snapshot.txt` — it is the zero point for the whole session.

**Useful one-liners on the pod:**

```bash
# just the metric names, no values
curl -s localhost:8000/metrics | grep -E '^# HELP vllm:' | sort

# anything cache-related (Week 2's missed step)
curl -s localhost:8000/metrics | grep -i cache | grep -v '^#'

# the four numbers that matter, right now
curl -s localhost:8000/metrics \
  | grep -E '^vllm:(num_requests_(running|waiting)|gpu_cache_usage_perc|num_preemptions_total)'
```

**Reference:** https://docs.vllm.ai/en/latest/design/metrics.html documents what
each vLLM metric means and how it is computed.

---

<a name="appC"></a>
# Appendix C — Pod session quick-reference card

**Have this open while the meter runs. Do not read anything else.**

```bash
# ---- 1. code onto the pod (repo root = stay here for everything) ----------
cd /workspace
git clone https://github.com/kashif-saeed1122/vllm-serving-internals.git
cd vllm-serving-internals
mkdir -p results/week3
chmod +x q1-vllm-serving/*.sh

# ---- 2. server, WITH the log captured -------------------------------------
./q1-vllm-serving/serve.sh 2>&1 | tee results/week3/server-startup.log

# ---- 3. capacity numbers (pane 2, once the server is up) -----------------
grep -Ei "KV cache size|Maximum concurrency|max_num_seqs|max_num_batched_tokens|Graph capturing|block_size" \
  results/week3/server-startup.log | tee results/week3/capacity-config.txt

# ---- 4. verify the model AND the revision hash ---------------------------
curl -s localhost:8000/v1/models | python3 -m json.tool | tee results/week3/models.json
grep -Ei "revision|commit|snapshot" results/week3/server-startup.log
#   expect Qwen/Qwen2.5-7B-Instruct-AWQ  +  b25037543e9394b818fdfca67ab2a00ecc7dd641

# ---- 5. METRIC NAMES -- DO NOT SKIP (this is Week 2's missed step) -------
curl -s localhost:8000/metrics | grep -E '^# HELP vllm:' | sort > results/week3/metric-names.txt
curl -s localhost:8000/metrics > results/week3/metrics-idle-snapshot.txt
nvidia-smi --help-query-gpu > results/week3/nvidia-smi-fields.txt
wc -l results/week3/metric-names.txt

# ---- 6. samplers, leave running all session ------------------------------
# pane 2:
./q1-vllm-serving/gpu-sampler.sh results/week3/gpu_full-session.csv
# pane 3:   (must print  kv cache metric = ...)
./q1-vllm-serving/metrics-sampler.sh results/week3/metrics_full-session.csv

# ---- 7. smoke test -------------------------------------------------------
SMOKE=1 ./q1-vllm-serving/week3-sweep.sh

# ---- 8. the sweep (~75 min) ----------------------------------------------
./q1-vllm-serving/week3-sweep.sh 2>&1 | tee results/week3/week3-sweep.log

# ---- 9. extract, VERIFY, then terminate ---------------------------------
# Ctrl-C both samplers first
ls -la results/week3/ | tee results/week3/MANIFEST.txt
find results/week3 -name '*.json' | wc -l          # expect 50
wc -l results/week3/metrics_full-session.csv       # expect 20,000+
tar czf week3-results.tar.gz results/week3/
# scp / runpodctl off the pod, verify it opens locally, THEN TERMINATE
```

### While axis D runs — the ONE thing to watch

In pane 3, the columns are:

```
ts , running , waiting , kv_cache_usage , preemptions , prefix_hits , prefix_queries
```

**At the first moment `waiting` goes above 0, read `kv_cache_usage` on that same
row and write both numbers down.**

| `kv_cache_usage` at that moment | Which trigger | What it means |
|---|---|---|
| **Low** (< 0.30) | **T2** — per-step token budget | **NOT the memory ceiling.** The scheduler is rate-limiting prefill. |
| **High** (≈ 1.0) **and** `preemptions` rising | **T3** — KV block exhaustion | **This is the memory ceiling.** |
| `running` flat at a round number, cache low | **T1** — `max_num_seqs` | Hard sequence cap, not memory. |

**`preemptions` is the honest ceiling signal.** High cache usage alone does not
prove exhaustion, because freed blocks are retained in the prefix cache and are
evictable on demand (Plan Part 1.7). Preempting a *running* sequence only
happens under genuine pressure.

### Emergency fallbacks

| Problem | Do this |
|---|---|
| Sampler prints FATAL, no cache gauge | `grep -i cache results/week3/metric-names.txt`, then `M_CACHE=<name> ./q1-vllm-serving/metrics-sampler.sh results/week3/metrics_full-session.csv` |
| Smoke test rejects a flag | `DETAILED=0 ./q1-vllm-serving/week3-sweep.sh` |
| Running out of time | Kill it. Re-run only what matters: `AXES="D" REPEATS=1 ./q1-vllm-serving/week3-sweep.sh`. Priority **D > C > B > A > E**. |
| No queuing even at conc 64 | Day 6 Step 6.8 escalation. **Never** touch `--gpu-memory-utilization` or `--max-model-len`. |
| `nvidia-smi` rejects `-lms` | Edit to `-l 1`. The `/metrics` sampler is the one that matters. |
| Revision hash differs from ENVIRONMENT.md | **Stop.** Decide before spending 90 min — Day 6 Step 6.3. |
