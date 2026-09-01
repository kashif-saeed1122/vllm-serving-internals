- [Week 1] [Prefix caching — trend confirmed] — Measured prefix cache hit
  rate rising across three tests as concurrent identical-prompt requests
  increased: 34.4% (1 unique prompt) -> 53.9% (5 concurrent identical
  short prompts) -> 58.6% (5 concurrent identical long prompts). This is
  consistent, repeatable evidence that cache state is both observable
  (via /metrics, no auth) and directly shaped by prompt repetition/overlap
  across requests — the underlying mechanism a timing side-channel attack
  would exploit. Still observation-only; no timing-based inference attempted.