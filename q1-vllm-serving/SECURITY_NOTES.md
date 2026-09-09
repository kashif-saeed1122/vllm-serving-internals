- [Week 1] [Prefix caching — trend confirmed] — Measured prefix cache hit
  rate rising across three tests as concurrent identical-prompt requests
  increased: 34.4% (1 unique prompt) -> 53.9% (5 concurrent identical
  short prompts) -> 58.6% (5 concurrent identical long prompts). This is
  consistent, repeatable evidence that cache state is both observable
  (via /metrics, no auth) and directly shaped by prompt repetition/overlap
  across requests — the underlying mechanism a timing side-channel attack
  would exploit. Still observation-only; no timing-based inference attempted.

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

 - [Week 3] [Prefix caching — TIMING DIFFERENTIAL MEASURED, 13x, with hit-rate
  counters] — Continuous `/metrics` sampling (0.5 s, 4,892 samples over 2,598 s)
  closed the Week 2 gap. Two of that note's claims resolve in opposite
  directions.

  **The wrapper prediction was exactly right.** Both 512-in/256-out runs measured
  a 3.0% hit rate: anchor 54,100 queries / 1,600 hits over 100 prompts;
  cold-start 21,640 / 640 over 40 prompts. That is **exactly 16 tokens — one
  block — per request** (100 x 16 = 1,600; 40 x 16 = 640). The shared chat
  template is the entire hit rate for random prompts, and block granularity is
  why it rounds to 16 tokens rather than the ~29 the template occupies: a partial
  block is not cacheable.

  **The "near zero cross-request sharing" assumption was wrong, and the error is
  the finding.** `--dataset-name random` does produce unique prompts *within* a
  run, but `--seed 42` is passed to **every** run of both sweep scripts with
  identical `--random-input-len`, `--random-output-len` and `--num-prompts`. Every
  run in a sweep therefore sends the **same prompt set**, and with
  `enable_prefix_caching=True` (default; confirmed from
  `vllm:cache_config_info` — `block_size="16"`, `num_gpu_blocks="15709"`,
  `prefix_caching_hash_algo="sha256"`) every run after the first re-sends prompts
  the engine has already cached. Cross-request sharing was not near zero; it was
  near total.

  **The measurement.** Two runs, adjacent in one sweep, same server, same 32
  prompts (6144-in/1024-out, seed 42), differing only in prior cache state:

  | run | order | queries (tok) | hits (tok) | hit rate | TTFT mean | TTFT p50 |
  |---|---|---|---|---|---|---|
  | conc 8  | first  | 197,536 | 14,864  | **7.5%**  | **3332.7 ms** | 3135.2 ms |
  | conc 16 | second | 197,536 | 197,120 | **99.8%** | **255.7 ms**  | 236.1 ms  |

  Identical query counts, because identical prompts. **A 13.0x TTFT reduction at
  twice the concurrency** — against the direction concurrency alone would push it.
  32 x 6,144 = 196,608 tokens fits inside the 251,344-token pool, so the second
  run's prompts were still fully resident.

  Confirmed a third time by dilution: the next run sent 64 prompts of which only
  the first 32 had ever been cached, and measured **50.0%** (395,072 queries /
  197,408 hits — within 0.15% of the previous run's hit tokens). The cache
  returned exactly the same 32 prompts and nothing more. At 96 and 128 prompts,
  retention collapsed to the 0.3% template floor because the prompt set no longer
  fits the pool alongside the live requests.

  Corroborated at a different prompt length: the Week 3 anchor (512-in/256-out,
  conc 20, 3.0% hit rate) measured TTFT 1033.30 ms where Week 2's identical
  configuration — the fifth of five runs sharing one prompt set — measured
  71.39 ms. **14.5x.** Decode was unaffected in both cases (ITL p50 within 1.6%
  of Week 2's ITL mean), isolating the entire effect to prefill.

  **Why this matters for the security thread.** Week 1 established that cache
  state is observable via unauthenticated `/metrics`. This measures the more
  serious channel: **cache state is observable through response latency alone**,
  with no access to `/metrics` at all, at a signal-to-noise ratio needing no
  statistics — 3,332 ms versus 256 ms against a within-run TTFT standard
  deviation of 107 ms. An unprivileged client that can time its own requests can
  determine whether a given prefix has been processed before. On a shared
  endpoint that is a membership oracle over other tenants' prompt prefixes:
  submit a candidate prefix, time the first token, infer whether someone else
  recently sent something beginning the same way. Prefix granularity is the
  16-token block, so the oracle answers at 16-token resolution and can in
  principle be walked forward block by block.

  Two further details sharpen the picture:

  - **Eviction is also observable.** Retention collapsed from 99.8% to 0.3% once
    the working set exceeded the pool. A client that can drive traffic can
    therefore *clear* the channel as well as read it, which is what a controlled
    prime-and-probe experiment requires.
  - **The cache removes queueing, not just latency.** The 99.8% run recorded zero
    `num_requests_waiting` across 116 samples, where the cold run at *lower*
    concurrency peaked at 6 waiting. Cache state changes the server's admission
    behaviour, so the observable is not confined to one request's own latency.

  **What this is NOT.** All observations are self-inflicted: one client, one
  sweep, prompts it had itself submitted. This demonstrates the mechanism and
  quantifies the differential under controlled conditions with the engine's own
  counters as ground truth. It is not an executed cross-tenant attack, and
  nothing was extracted, because there was no other party.

  **Deferred deliberately.** A real timing-inference experiment — plant a prefix,
  evict it, time a probe, measure oracle accuracy and per-block resolution
  against known ground truth — is Month 11 work ("Red-team a tool-using setup",
  where the year's logged surfaces cash in). The Q4 eval harness is the right
  place to measure oracle accuracy properly. Logging it here and moving on, per
  the plan's rule that security never becomes a separate front.

  **Immediate non-security consequence, recorded because it bites first.** This is
  simultaneously a benchmarking defect: TTFT in `week2-notes.md` measures cache
  lookups rather than prefill for four of its five rows, and the Week 3 ceiling
  series is contaminated at conc 16 (99.8%) and conc 32 (50.0%), which forced the
  retraction of a throughput conclusion in `week3-notes.md`. Both sweep scripts
  need per-run seeds, or per-run hit-rate reporting alongside every TTFT, before
  another latency number is quoted. A performance benchmark and a side-channel
  turn out to be the same measurement read two ways — which is the clearest
  argument so far for the plan's decision to keep security logged inside the
  systems work rather than split off as its own front.

  **One open question.** The long-prompt cold run's floor was 7.5% (929 blocks
  over 32 requests, ~29 blocks each) rather than the single template block seen
  at short prompt lengths. Either the `random` dataset shares a longer prefix at
  6144 tokens, or chunked prefill re-queries the cache for portions it has already
  computed within the same request. The second would mean the hit-rate metric
  partly measures intra-request behaviour at long prompt lengths, which affects
  how the 99.8% figure should be read. One look at
  `vllm/benchmarks/datasets.py` and the chunked-prefill path.