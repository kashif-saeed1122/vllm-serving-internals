# The lifecycle of one chat request in vLLM

One streaming request to `/v1/chat/completions`, followed from the moment it
reaches the server to the moment my benchmark client timestamps the reply.

**The request I am following:** the same one my Month 1 sweeps sent.
`vllm bench serve --backend openai-chat`, random dataset, 512 tokens in and
256 out, `stream=True`, `ignore_eos`. Model: Qwen2.5-7B-Instruct-AWQ on one GPU.

**Source:** vLLM `v0.28.0`. This is the version in
`ENVIRONMENT.md`, so the code I read is the code I benchmarked. Line numbers
will drift; the function names are the anchors.

**Marks:** `?` means I have not confirmed it. Those marks are Saturday's list,
and they are collected at the end.

Sections 1, 2, 3 and 9 are written this week. Sections 4 to 8 (scheduler,
KV blocks, model runner, sampling) come in Weeks 6 to 8.

---

## Diagram: the whole trip

```
 CLIENT  vllm/benchmarks/lib/endpoint_request_func.py
         async_request_openai_chat_completions
         st = perf_counter()
            |
            | POST /v1/chat/completions  (JSON, stream=True)
            v
+--------------------------- FRONTEND PROCESS (asyncio) ---------------------------+
| [1] api_router.create_chat_completion                                            |
|       -> OpenAIServingChat._create_chat_completion         chat_completion/serving.py
|       -> OnlineRenderer.render_chat -> preprocess_chat     renderers/online_renderer.py
|       -> BaseRenderer.render_chat_async                     renderers/base.py     |
|            arrival_time = time.time()           <-- server clock starts here    |
|            HfRenderer.render_messages_async      messages -> ONE STRING (template)
| [2]        tokenize_prompts_async                STRING -> prompt_token_ids      |
|            process_for_engine_async              -> EngineInput dict             |
|       -> AsyncLLM.generate -> add_request                   v1/engine/async_llm.py
|            InputProcessor.process_inputs         -> EngineCoreRequest            |
|            OutputProcessor.add_request           (frontend keeps text+detokenizer)
|            AsyncMPClient.add_request_async       msgpack -> ZMQ ROUTER           |
+-----------------------------|--------------------------------------^-------------+
          (ADD, frames)       |                                      | EngineCoreOutputs
          ROUTER -> DEALER    |                                      | PUSH -> PULL
+-----------------------------v------- ENGINE CORE PROCESS ----------|-------------+
| input IO thread:  process_input_sockets -> decode -> preprocess_add_request       |
|                   -> input_queue                                   |             |
| [3] main thread:  run_busy_loop                                    |             |
|                     _process_input_queue -> scheduler.add_request (WAITING)       |
|                     _process_engine_step -> step()                 |             |
|                        schedule -> execute_model -> update_from_output            |
|                     -> output_queue  (new_token_ids, no text)      |             |
| output IO thread: process_output_sockets -> msgpack -> PUSH -------+             |
+----------------------------------------------------------------------------------+
            |
            v  [9] back in the FRONTEND
   AsyncMPClient.process_outputs_socket -> outputs_queue
   AsyncLLM.output_handler -> OutputProcessor.process_outputs
        detokenizer.update        token ids -> delta text
        make_request_output       -> RequestOutputCollector.put
   AsyncLLM.generate yields RequestOutput
   chat_completion_stream_generator yields  "data: {json}\n\n"
   StreamingResponse writes it to the socket
            |
            v
 CLIENT  first chunk with "choices":  ttft = perf_counter() - st
```

---

## 1. Arrival and template

**One sentence:** the messages list is turned into a single prompt **string**
by the model's Jinja chat template, and that happens in the frontend before
anything is tokenised.

**Step by step:**

1. **Client sends.** `async_request_openai_chat_completions`
   (`vllm/benchmarks/lib/endpoint_request_func.py`) builds the payload
   (`messages`, `max_completion_tokens`, `stream: True`,
   `stream_options.include_usage: True`), sets `st = time.perf_counter()` and
   POSTs.
2. **FastAPI route.** `create_chat_completion`
   (`entrypoints/openai/chat_completion/api_router.py`, line 53) is registered
   for `POST /v1/chat/completions`. Before my code runs:
   - `validate_json_request` checks the content type.
   - FastAPI parses the body into `ChatCompletionRequest` (a pydantic model,
     `chat_completion/protocol.py`, line 213). Bad JSON is rejected here.
   - `@with_cancellation` and `@load_aware_call` wrap the handler. (?) My guess:
     the first aborts the request when the client disconnects, and the second
     counts in-flight requests for load metrics.
3. **Handler.** `chat(raw_request)` fetches the `OpenAIServingChat` instance
   from app state. The route calls `OpenAIServingChat.create_chat_completion`
   (`chat_completion/serving.py`, line 225), which wraps
   `_create_chat_completion` (line 241).
4. **Engine-aware checks.** `render_chat_request` (line 198) checks that the
   model name is served (`_check_model`) and that the engine is not dead. Then
   it hands off to `OnlineRenderer.render_chat`.
5. **Request-level checks.** `OnlineRenderer.render_chat`
   (`renderers/online_renderer.py`, line 117) validates `tool_choice` and the
   chat template, then calls `preprocess_chat` (line 379).
6. **Template settings.** `preprocess_chat` builds `chat_params` and
   `tok_params`. It sets `tokenize=False` unless the tokenizer is Mistral or
   prompt embeds are on. For Qwen this means the template returns **text**.
   It then calls `renderer.render_chat_async` (line 426).
7. **Server clock starts.** `BaseRenderer.render_chat_async`
   (`renderers/base.py`, line 1071) runs `arrival_time = time.time()` on its
   first line (line 1080), before templating and tokenising.
8. **Template applied.** `HfRenderer.render_messages_async`
   (`renderers/hf.py`, line 1060):
   - `parse_chat_messages_async` normalises the messages into a
     `conversation` list.
   - `_apply_chat_template_async` (line 1122) runs `safe_apply_chat_template`
     in the renderer's thread pool. That function finds the template
     (`resolve_chat_template`), converts `developer` messages to `system` if
     the template has no developer role, and calls the HF
     `tokenizer.apply_chat_template(..., tokenize=False)`.
   - `parse_dec_only_prompt` wraps the string as `{"prompt": "<text>"}`.

**What the template produces for Qwen2.5** (?) I expect ChatML, with a default
system prompt added when the request has none:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...my 512-token random prompt...<|im_end|>
<|im_start|>assistant
```

The last line comes from `add_generation_prompt`. It is why the model answers
as the assistant instead of continuing my text.

**Correction to my Wednesday notes:** I wrote that `arrival_time` is set in
`render_chat` (line 1044). That is the synchronous version. The online chat
path goes through `render_chat_async`, so the line that runs is 1080. The idea
is the same; the function is different. Trace the path that actually runs.

---

## 2. Tokenisation and handoff to the engine process

**One sentence:** the prompt string becomes token ids in the frontend
process; only the ids and the sampling settings cross to the engine, packed
in an `EngineCoreRequest`, serialised with msgpack, and sent over ZMQ.

**Tokenise (still inside `render_chat_async`, `renderers/base.py`):**

1. `tokenize_prompts_async` (line 641), then `tokenize_prompt_async`, then
   `_tokenize_singleton_prompt_async` (line 552).
2. `_tokenize_prompt_async` is `_tokenize_prompt` run in the renderer's
   `ThreadPoolExecutor` (set up at line 96), so the event loop is not blocked.
3. `_tokenize_prompt` (line 472) calls `tokenizer(prompt["prompt"], ...)` and
   keeps `input_ids`. The result is a `TokensPrompt` with `prompt_token_ids`.
4. `process_for_engine_async` (line 962) calls `_process_tokens_async`
   (line 835), which builds `tokens_input(prompt_token_ids)` and keeps the
   original text under `"prompt"`. Then it adds
   `engine_input["arrival_time"] = arrival_time`.

**Build sampling params (back in `_create_chat_completion`, `serving.py`):**

5. `get_max_tokens` caps the output by `max_model_len` minus the prompt
   length.
6. `request.to_sampling_params(...)` builds `SamplingParams`. Because
   `stream=True`, it sets `output_kind=RequestOutputKind.DELTA`
   (`protocol.py`, line 737). That means each output carries only the new
   text. This matters in section 9.
7. `self.engine_client.generate(engine_input, sampling_params, request_id, ...)`
   (line 351). `engine_client` is the `AsyncLLM`.

**Package (`vllm/v1/engine/async_llm.py`):**

8. `AsyncLLM.generate` (line 550) calls `add_request` (line 283).
9. The input is a rendered `EngineInput` (it has a `"type"` key), so
   `add_request` calls `InputProcessor.process_inputs` (line 356). That
   function (`input_processor.py`, line 255) validates and returns an
   `EngineCoreRequest` (built at line 384). `arrival_time` is copied in here.
10. `assign_request_id` (line 393) adds the internal id. A
    `RequestOutputCollector` is created as this request's mailbox (line 400).
11. `_add_request` (line 424) does two things:
    - `output_processor.add_request(...)` registers the request **in this
      process**: prompt text, detokenizer state, and the mailbox.
    - `engine_core.add_request_async(request)` (line 436) sends it to the
      **other** process.

**Send (`vllm/v1/engine/core_client.py`):**

12. `AsyncMPClient.add_request_async` (line 1148) sets `client_index` and
    calls `_send_input(EngineCoreRequestType.ADD, request)` (line 1107).
13. `self.encoder.encode(request)` (a `MsgpackEncoder`) turns the struct into
    frames. `input_socket.send_multipart((engine, b"\x00", *frames))`
    (line 1126) sends them on the ROUTER socket.

**Process boundary.**

**Receive (`vllm/v1/engine/core.py`, input IO thread):**

14. `EngineCoreProc.process_input_sockets` (line 1651) runs in its own
    thread. `recv_multipart` (line 1711) reads the frames on a DEALER socket.
15. `add_request_decoder.decode(data_frames)` (line 1722) rebuilds the
    `EngineCoreRequest`.
16. `preprocess_add_request` (line 1724, defined at line 968) turns it into
    the scheduler's `Request` object.
17. `self.input_queue.put_nowait((request_type, request))` (line 1752) hands
    it to the main thread.

**What crosses the boundary and what stays behind:**

| Crosses to the engine | Stays in the frontend |
|---|---|
| `prompt_token_ids` | the prompt text |
| `sampling_params` | the detokenizer and its state |
| `request_id`, `arrival_time`, `priority` | the `RequestOutputCollector` (mailbox) |
| | the HTTP connection |

**Why two processes:** the GIL. HTTP, templating, tokenising, detokenising and
JSON all run in the frontend. Scheduling and launching GPU work run in the
engine. In one process, each would slow the other down.

(?) Tokenising runs in a thread pool, but threads share the GIL. Does the HF
fast tokenizer (Rust) release the GIL while it encodes? If not, a long
prompt's tokenisation still blocks the event loop.

(?) On one machine, are the ZMQ sockets `ipc://` or `tcp://`? I expect IPC.

---

## 3. The engine loop

*One paragraph for now. Weeks 6 to 8 fill in sections 4 to 8.*

The engine's main thread spends its whole life in
`EngineCoreProc.run_busy_loop` (`core.py`, line 1375), which repeats two
calls. `_process_input_queue` (line 1401) blocks on `input_queue` while there
is no work. Once work arrives it drains the queue, and
`_handle_client_request` (line 1504) sends my ADD to `EngineCore.add_request`
(line 438), then to `Scheduler.add_request` (`v1/core/sched/scheduler.py`,
line 2301), where `_enqueue_waiting_request` puts it in the `waiting` queue
with status `WAITING`. Then `_process_engine_step` (line 1432) calls
`step_fn`, normally `step()` (line 583): `scheduler.schedule()` picks which
requests run this iteration and how many tokens each gets, within the
`max_num_batched_tokens` budget from the SARATHI paper. My 512-token prefill
can go in one iteration or be chunked across several, and then it decodes one
token per iteration. Next, `model_executor.execute_model(...)` runs the
forward pass, `sample_tokens` picks the next token ids, and
`scheduler.update_from_output` appends them, checks stop conditions and
builds an `EngineCoreOutput` per request (`new_token_ids`, `finish_reason`,
no text). These are grouped into `EngineCoreOutputs` per client and put on
`output_queue`. My request is one row of a batch that is rebuilt every
iteration, and the loop runs about 256 times for it, once per output token.

(?) `step_fn` is `step_with_batch_queue` instead of `step` when
`max_concurrent_batches > 1`. That is required for pipeline parallelism. Is it
also on for my single GPU because async scheduling is the default in 0.28?
Check the startup log.

---

## 9. The way back: token id → text → SSE chunk → client stamp

**One sentence:** the engine sends back only token ids; the frontend turns
them into text one token at a time, wraps each piece in an OpenAI JSON chunk,
writes it as a Server-Sent Event, and the client starts its TTFT stopwatch
on the first chunk that has `choices`.

**Engine side (`core.py`):**

1. `_process_engine_step` puts each `(client_index, EngineCoreOutputs)` on
   `output_queue` (line 1439).
2. `process_output_sockets` (line 1754) is the output IO thread. It takes
   from `output_queue`, encodes with `MsgpackEncoder.encode_into` (reusing
   buffers), and sends on a PUSH socket.

**Frontend, receive (`core_client.py`):**

3. `AsyncMPClient.process_outputs_socket` (line 1040) is an asyncio task. It
   awaits `recv_multipart` on the PULL socket, decodes with
   `MsgpackDecoder(EngineCoreOutputs)`, and puts the result on
   `outputs_queue`.

**Frontend, token id → text (`async_llm.py` and `output_processor.py`):**

4. `AsyncLLM.output_handler` (`async_llm.py`, line 686) is one background
   task for **all** requests. It awaits `engine_core.get_output_async()`,
   splits the batch into slices of `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`, and
   calls `output_processor.process_outputs` (line 705) on each slice. It
   yields to the event loop between slices.
5. `OutputProcessor.process_outputs` (`output_processor.py`, line 598) loops
   once over the batch. For my request it does four things:
   - `_update_stats_from_output`: this is where vLLM's **own** TTFT is
     recorded, as first token time minus `arrival_time`
     (`v1/metrics/stats.py`, line 393).
   - `detokenizer.update(new_token_ids, ...)` (line 669) calls `decode_next`
     once per token (`detokenizer.py`). Detokenising is incremental, and a
     token can give `""`, for example half of a multi-byte UTF-8 character.
     Stop strings are checked here too. See Thursday's Q4 notes.
   - `make_request_output(...)` (line 681) builds a `RequestOutput`. Because
     the output kind is DELTA, `.text` holds only the new text.
   - `req_state.queue.put(request_output)` (line 694) drops it in my
     request's `RequestOutputCollector`.
6. `RequestOutputCollector.put` (line 64) **merges** a new output into the one
   still waiting if the consumer has not picked it up yet. So under load
   **one SSE chunk can carry more than one token**.
7. My request's own `AsyncLLM.generate` loop (line 607) runs
   `q.get_nowait() or await q.get()` and yields the `RequestOutput`.

**Frontend, text → SSE (`chat_completion/serving.py`):**

8. `chat_completion_stream_generator` (line 431) runs
   `async for res in result_generator`.
   - **First iteration:** it yields a **role chunk** first:
     `delta={"role": "assistant", "content": ""}` (line 523, yielded at 559).
   - **Then, for every output:** `delta_text = output.text`, wrapped in
     `DeltaMessage(content=delta_text)`, then `ChatCompletionStreamResponse`,
     then `chunk.model_dump_json(exclude_unset=True)`, then
     `yield f"data: {data}\n\n"`.
   - Empty deltas are skipped only in the "chunked prefill, nothing generated
     yet" case (line 631).
   - **At the end:** a usage-only chunk (because `include_usage` is on), then
     `data: [DONE]\n\n` (line 886).
9. `StreamingResponse(content=generator, media_type="text/event-stream")`
   (`api_router.py`, line 74). Starlette pulls each string from the async
   generator and writes it to the socket as soon as it is yielded.

**Client stamp (`endpoint_request_func.py`):**

10. `async for chunk_bytes in response.content.iter_any()`, then
    `StreamedResponseHandler.add_chunk` splits the bytes into SSE messages.
    For each message, `timestamp = time.perf_counter()`.
11. The first message that has `choices` sets `ttft = timestamp - st`. Every
    later one appends `timestamp - most_recent_timestamp` to `itl`. The usage
    chunk has no `choices`, so it only updates `output_tokens`.
    `latency = most_recent_timestamp - st`.

**What this means for my Month 1 numbers:**

- My client TTFT is stamped on the **role chunk**, which has no text. That is
  still fair, because the role chunk is only sent once the first
  `RequestOutput` exists, which means the first token has been sampled.
- My client TTFT = (network + HTTP parsing + validation before
  `arrival_time`) + vLLM's TTFT + (output IPC + detokenise + JSON + network
  back). vLLM's `/metrics` TTFT sees only the middle part.

(?) The role chunk and the first content chunk come from the same `res` and
are yielded back to back. So the **first ITL sample of every request should be
about 0 ms**, which pulls mean ITL down slightly. My Month 1 JSONs don't keep
per-request ITLs. Rerun one config with `--save-detailed` and check `itls[0]`.

(?) If chunks merge under load (step 6), one ITL gap covers two or more
tokens, and some gaps are missing. Does that explain part of the ITL p99
growth from c=10 to c=20 in Week 2?

(?) My `serve.sh` sets no `--tool-call-parser` or `--reasoning-parser`, so I
think `parser_cls` is `None`. If so, a token that decodes to `""` still sends
a chunk with `content: ""`, and the client counts it as an ITL sample. Confirm
that `parser_cls` really is `None` with my flags.

---

## Saturday: open `?` marks

1. What do `@with_cancellation` and `@load_aware_call` actually do?
2. Does Qwen2.5's template add the default system prompt when none is sent?
   Check `tokenizer.apply_chat_template(..., tokenize=False)` locally.
3. Does the HF fast tokenizer release the GIL while encoding?
4. Are the frontend↔engine ZMQ sockets IPC or TCP on one node?
5. Is `step_fn` `step` or `step_with_batch_queue` on my setup?
6. Is `itls[0]` about 0 ms in a `--save-detailed` run?
7. Do merged chunks under load change what my ITL percentiles mean?
8. Is `parser_cls` `None` with my `serve.sh` flags, so empty deltas still
   become chunks?
