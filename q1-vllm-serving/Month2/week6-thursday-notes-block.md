# Week 6 · Thursday: the waiting loop and the way out

Source: vLLM v0.34.0rc0+728 (local checkout, commit f538463a2e). Line numbers are local; the v0.28.0 number is in [brackets] where I have it. Line numbers drift between versions.
Files: `vllm/v1/core/sched/scheduler.py` (`schedule()` waiting pass starts L1030, `update_from_output` stop path L2492), `kv_cache_manager.py`, `request_queue.py` (skimmed), `output.py`.

## One sentence
The waiting pass admits new requests with whatever token purse is left, charging only the tokens the prefix cache does not already cover. A request's blocks are freed when `check_stop` and `_handle_stopped_request` both say it is finished.

## Two things that get confused
Hit, chunk and remainder are three different slices of one prompt.

| | Hit | Chunk | Remainder |
|---|---|---|---|
| What | tokens found in the prefix cache (full 16-token blocks only) | tokens computed this step | tokens still to compute after this step |
| Charged to the purse? | no | yes | no, charged next step |
| Set at | lookup, L1121 [811] | `min(need, room)`, L1377 [966] | prompt - hit - chunk |
| Served by | nobody, already computed | this step's waiting pass | next step's running pass |

Example (illustrative): 541-token prompt, hit 16, room 425.
need = 541 - 16 = 525, chunk = min(525, 425) = **425**, `num_computed_tokens` after the step = 16 + 425 = 441, remainder = **100**.

Decide vs execute in the stop path:

| Function | Line | Job |
|---|---|---|
| `check_stop` | `utils.py:98` [94], called L2852 | should it end? (EOS, stop token, `max_tokens`, `max_model_len`) |
| `_handle_stopped_request` | L2812 | is it really finished? False for a resumable request |
| `_free_request` | L3123 [2388] | adds the id to `finished_req_ids` (L3147), then frees unless a connector delays |
| `_free_blocks` | L3157 [2417] | calls `_free_request_blocks`, then forgets the request |
| `_free_request_blocks` | L3177 [2429] | frees now, or defers when `defer_block_free` is on |
| `kv_cache_manager.free` | `kv_cache_manager.py:649` | returns blocks to the pool, tail first |

## Q1: when does the waiting loop not run?
**Answer:** if anything was preempted this step, or the scheduler is paused (`PAUSED_ALL`).
**Where:** `if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:` at L1030 [748].
- `preempted_reqs` is a local list, created empty each step (L706). Preemption appends to it (L955).
- A finish does not empty it. The next step starts with a new list.
- The victim goes to the front of `waiting` with `num_computed_tokens = 0` (L1843, L1863).

Why it is sensible: preemption means the blocks just ran out. Admitting someone now takes blocks again and causes another preemption next step.

Other ways the loop does nothing or stops:
| Reason | Effect |
|---|---|
| queues empty | nothing to admit |
| `token_budget <= 0` | purse empty |
| `num_running >= max_num_running_reqs` | no seats |
| `allocate_slots` returns None for the head (L1497-1504) | `break`; head stays; FCFS blocks everything behind |

**Hold this:** a preemption this step turns off admission for this step, and the victim waits at the front.

## Q2: how the count is cut, and where chunked prefill happens
**Answer:** `num_new_tokens = min(need, request_token_budget)` at L1377 [966]. Chunked prefill is that one line.
- need: `request.num_tokens - num_computed_tokens` (L1337 [932]).
- room: `min(token_budget, input_budget - draft_slots)` (L1326).
- Chunked prefill off and need > room: `break` at L1375. The request and everyone behind it stay in `waiting`.
- The purse refills every step, so it is retried next step.
- `num_computed_tokens` is set to the pre-chunk value at L1613 and advanced by `_update_after_schedule` (L1879).
- Chunk size comes from the purse only, never from free blocks. `allocate_slots` is all-or-nothing for the chunk; None means `break`.

Example: 700-token prompt, 300 left in the purse, chunked prefill on: `num_new_tokens` = **300**, the other **400** are served by the running loop next step.

**Hold this:** the purse sets the chunk, `allocate_slots` only says yes or no, and the running pass finishes the prompt.

## Q3: prefix-cache lookup, and does a hit save budget?
**Answer:** yes, a hit saves budget. The lookup runs in the waiting loop, only for the head request being examined, and only if `num_computed_tokens == 0` (L1121 [811]).
- Chain: `_get_local_prefix_cache_hit` (L656-668) -> `get_computed_blocks` (`kv_cache_manager.py:287-328`).
- Only full 16-token blocks hit. The cap `num_tokens - 1` (`request.py:352-356`) binds only when the length is a multiple of 16.
- The hit raises the starting `num_computed_tokens` (L1228-1230), so `num_new_tokens` at L1337 and the charge shrink by the hit.
- The lookup only finds blocks. `allocate_slots` attaches them (L1478-1495). If it returns None, nothing is attached.
- Shared blocks use `ref_cnt`: they save compute budget and new block allocation.
- Stats are recorded at admission (L1553), so queries = total input tokens.

Example: 100-token prompt, first 96 cached: hit = 96 tokens = 6 blocks, `num_new_tokens` = 4.

**Hold this:** a hit means already computed, so it is free in the purse. The charge is what comes after it.

## Q4: which event frees a request's blocks?
**Answer:** the stop event in `update_from_output`.
- `check_stop` returns True (L2852), then `if stopped:` (L2492), then `_handle_stopped_request` (L2812) returns True.
- Then `_free_request` -> `_free_blocks` -> `_free_request_blocks` -> `kv_cache_manager.free`. Tail blocks go first. Hashed blocks stay in the prefix cache until reused.
- The request is removed from `running` after the loop (L2566).
- Resumable (streaming) request: `check_stop` True but `_handle_stopped_request` returns False. Blocks are kept and the request is re-queued (L2733).
- 100th token with `max_tokens` 256: `check_stop` False, none of the chain runs.
- Deferral is not the normal case. `defer_block_free` is True only with a KV connector (consumer) and overlapping batches (L190-196). Plain `vllm serve` frees at once (L3182).
- Aborts go through `finish_requests` (L3053).

Example: 256th token on a 797-token request: 50 blocks freed.

**Hold this:** `check_stop` says it should end, `_handle_stopped_request` confirms it, and only then are the blocks freed.

## `output.py`: what `SchedulerOutput` holds
The one-step work order for the workers (`output.py:241`), built at the end of `schedule()` (L1747).
| Field | Meaning |
|---|---|
| `scheduled_new_reqs` | first-time requests: full prompt, `block_ids` |
| `scheduled_cached_reqs` | known requests: diff only |
| `num_scheduled_tokens`, total | tokens per request, total at most 2048 |
| `finished_req_ids` | how a free reaches the workers |
| `preempted_req_ids` | requests preempted since the last step |
- `finished_req_ids` is scheduler state, passed in (L1760) and replaced by a new `set()` (L1897). A finish after step N is reported in step N+1's output.
- v0.34: a preempted request that comes back is sent again as a **new** request (L1602). v0.28.0 used `resumed_req_ids` in `CachedRequestData`.
- Worker side not read.

**Hold this:** new requests come with everything, known requests come as diffs, and the finished and preempted lists carry scheduler decisions to the GPU side.

## Numbers for the Week 3 queuing finding
Facts to cite. The five sentences are still to be written from memory.
| Fact | Value | Source |
|---|---|---|
| Purse per step | 2048 | Tuesday, `arg_utils.py:3040-3049` |
| Seats | 256 (prediction, not yet measured), never the limit in the burst | Tuesday |
| Anchor c20 prompt | 541 = 512 + 29 template; 16 cached; charge 525 | week3-notes.md:150 |
| Anchor burst | 20 x 525 = 10,500 = about 5.1 purses = 6 steps | |
| Memory per request | 541 tokens = 34 blocks, 0.2% of 15,709 | arithmetic |
| c16 ceiling prompt | 6173 = 6144 + 29 (197,536 / 32 prompts) | week3-notes.md:316, 332 |
| c16 hit | 6160 = 385 blocks; charge 13 | |
| c16 total | 16 x 13 = 208, far below 2048, never queued | |
| c16 without the cache | 16 x 6173 = 98,768 = about 48 purses | |
| c8 | hit 7.5%, about 5,700 per request, a burst of 8 about 46,000 = about 22 purses | |
| c32 | 50.0% because 64 prompts, first 32 still cached | week3-notes.md:333 |

## Traps
- `preempted_reqs` is a per-step local list. A finish does not empty it.
- Chunked prefill does not fill free blocks first. The purse sets the chunk.
- Chunk vs remainder, twice swapped: 425 not 116; 300 not 400.
- The hit counts full blocks: 6160, not 6172.
- 6173 is one request (6144 + 29). It is not 16 x 512 + template: 16 is concurrency, 512 is the anchor run, and the charge is per request.
- `check_stop` True does not always mean free. `_handle_stopped_request` decides.
- A request mid-run (the 100th token) triggers none of the free chain.
- A finish after step N appears in step N+1's `SchedulerOutput`.
- A decoding request is in `scheduled_cached_reqs`, not `scheduled_new_reqs`.
- `_free_blocks` executes the free; it does not decide it.
- Chunking off: the `break` is in the waiting loop, and the purse refills every step.
- Block-free deferral needs a KV connector consumer plus overlapping batches.
- 256 seats is a prediction from the table, not a measurement. Seats are not what held the burst back.

## Self-test (answer with the file closed)
1. What condition skips the waiting loop, and where is it?
2. 700-token prompt, 300 left in the purse, chunked prefill on: `num_new_tokens`, and who serves the rest?
3. Same prompt with chunked prefill off: what happens to it and to the request behind it?
4. 100-token prompt, first 96 cached: hit and `num_new_tokens`?
5. Why did c16 never queue? Give the numbers.
6. Where does 6173 come from?
7. Which function decides whether a stopped request's blocks are freed?
8. D finishes after step N: which step's `SchedulerOutput` carries its id, and why?
9. A decoding request: new or cached?
10. When are blocks freed later than the stop event?

## Open at close
- Five-sentence "done when" (Week 3 queuing finding, two code locations) not yet written or marked.
- Four answers were marked wrong and not yet redone: the chunk numbers (self-test 2), which function decides the free (7), which step carries `finished_req_ids` (8), new vs cached (9).
- Not read: `allocate_new_computed_blocks`, `find_longest_cache_hit` internals, the worker side of `SchedulerOutput`, `_holds_kv_blocks`.
- Not verified: whether a request admitted earlier in the same step can be hit by the next one; whether re-running c16 after c64 would give 99.8% again.
