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