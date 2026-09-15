# My Observations

## 1. Two bounds, not one number:

- **Upper bound** (only request 1 misses): hits = 99 × 16 = 1,584 → rate = 1,584 / 54,100 ≈ 2.93%
- **Lower bound** (first 5 all miss): hits = 95 × 16 = 1,520 → rate = 1,520 / 54,100 ≈ 2.81%

## 2. E3 / E4 — Run 2 hit rate / hit tokens

Run 2, same seed → all 100 prompts identical to run 1, and (check E8 below) small enough to still be resident.

Split 541 by the block size: 541 ÷ 16 = 33.8125 → 33 complete blocks = 528 tokens, remainder = 13 tokens. Those 33 blocks were cached in run 1 and match exactly. The trailing 13-token remainder never formed a complete block on its own, so it has no cache entry and must be recomputed even on an exact repeat.

- Hits per request = 528 tokens
- Total hits = 100 × 528 = 52,800
- Queries = 54,100 (same as run 1 — same prompts, same token count)
- Rate = 52,800 / 54,100 ≈ 97.6%

## 3.

## 4.

ITL would be almost same or negligble higer in case of run 2. becuase decode is identical in either way

**Prediction:** within ±5%. If E6 moves more than that, something other than caching changed between your two runs, and that would undermine trusting E5.

## 5. E7 — relationship between the two query deltas

**Prediction:** queries(run1) = queries(run2) = 54,100, exactly. If these don't match exactly, something about the two runs wasn't actually identical (different prompt count, different length, a stray extra request) — that's a red flag on the whole experiment, not a rounding difference.

## 6.

## Experiment 1 — scorecard

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| E1 | hit rate 2.8–2.9% | 2.93% | hit |
| E2 | hit tokens 1,520–1,584 | 1,584 (exact) | hit — upper bound |
| E3 | hit rate 97.6% | 97.60% | hit — exact |
| E4 | hit tokens 52,800 | 52,800 (exact) | hit — exact |
| E5 | ratio << 42x | 7.3x (mean), 8.2x (p50) | hit |
| E6 | ITL p50 within ±5% | 0.23% | hit |
| E7 | queries equal, 54,100 | 54,100 both (verified via delta) | hit — exact |
| E8 | KV occupancy ~1.6% | not captured — revised reasoning puts Run 2 at ~22.6% (Run 1's ~3,300 resident blocks + Run 2's own 250), still far below 97.5% threshold | not captured, reasoning corrected |
| E9 | 0 preemptions | not captured, but consistent with E8's revised occupancy | not captured, consistent |
| E10 | text identical | identical, verified | hit |
| E11 | ~1.0–1.3x | 1.183x | hit |


The same seed produces the identical set of prompts every time. In Run 1, the only thing shared across the 100 requests is the chat-completion template wrapper, which forms one cacheable 16-token block. Request 1 pays for this block in full; requests 2–100 find it already cached. That gives 99 × 16 = 1,584 hit tokens out of 54,100 queried, a hit rate of 2.93%.

In Run 2, every prompt is byte-identical to its Run 1 counterpart, and Run 1's cache entries are still resident. Of each 541-token prompt, 528 tokens (33 complete 16-token blocks) match exactly and are served from cache; only the trailing 13-token remainder — too short to form a complete block — is recomputed. That gives a hit rate of 97.60%, roughly 33x higher than Run 1's rate.

ITL doesn't depend on prefix caching at all — it measures the per-token cost of decoding, which is fixed compute work the cache can't skip regardless of how much of the prompt was cached. That's why it stayed flat (0.23% change) between the two runs. TTFT, by contrast, is dominated by prefill compute plus network and scheduling overhead — cutting 528 of 541 tokens' worth of compute cuts TTFT substantially (7–8x), even though the fixed overhead keeps it from dropping the full 42x that the raw token-count ratio would suggest.

## Open question — warm-cache hit rate at n=500 far below prediction

Predicted (partial-LRU model): ~62% blended warm rate at n=500, conc 5.
Measured: 3.53% — barely above the cold baseline (3.15%).
500 x 50-block full footprint = 25,000 tokens > 15,709 pool, so overflow is
real, but simple recency-based eviction doesn't explain a rate this low.
Mechanism unresolved. Deferred — not chased further this week.

## Method note — units mismatch in detailed output

The `ttfts` array in --save-detailed JSON output is in SECONDS.
All summary fields (mean_ttft_ms, p50_ttft_ms, etc.) are in MILLISECONDS.
Multiply raw array values by 1000 before computing anything from them.
Verified: hand-computed percentiles (numpy linear interpolation) match the
tool's reported p50/p95/p99 exactly once units are corrected.

run        conc  n     mean     p50     p95     p99     min      max     std
c1_cold      1   100  110.97  107.07  110.08  116.67  103.90   487.79   37.91
c2_cold      2   500  152.21  148.78  197.14  203.29   70.65   269.60   43.21
c5_cold      5   500  381.74  450.84  468.12  474.17  108.84   482.14  134.82
c5_warm      5   500  382.32  450.80  466.47  472.44   69.23   476.00  135.82
c10_cold    10   500  642.87  787.94  882.42  906.42  120.17   929.51  239.66
c20_cold    20   500 1006.23 1096.91 1437.59 1739.04  121.13  1845.51  393.02