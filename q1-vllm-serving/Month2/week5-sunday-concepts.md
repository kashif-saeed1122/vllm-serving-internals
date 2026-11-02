# Week 5, Sunday: interview concepts (async Python, streaming APIs, tokenisation)

**What this file is for:** revising six ideas so I can explain *why* they work,
not just use them. Each section follows the same pattern:

1. **The idea** in plain words
2. **A general example** (works in any Python project)
3. **Where it lives in vLLM** (code I read this week, commit `2cf0a69`, v0.28.0)
4. **Mistakes and debugging**
5. **Interview questions**, with short answers I can say out loud

The six ideas are not separate topics. They are six stops on the same trip
that one chat request makes:

```
client ──POST──► FastAPI route (async def) ──► chat template ──► tokeniser
                   │  [2 FastAPI]                [5 templates]    [4 tokenisation]
                   │
                   │   everything in the frontend runs on ONE event loop  [1 asyncio]
                   ▼
              vLLM engine core (separate process) ──► token ids back
                   [6 vLLM V1 big picture]
                   │
                   ▼
              detokenise ──► "data: {...}\n\n" ──► client
                              [3 SSE]
```

---

## 1. asyncio: one worker who never waits around

### 1.0 Start here: the basic words in simple English

**`async` and `asyncio` are two different things.**

| Name | What it is | Job |
|---|---|---|
| `async` | A **keyword** (part of the Python language) | Marks a function that **can pause**: `async def f():` |
| `await` | A **keyword** | Means "pause me here until this is finished" |
| `asyncio` | A **library** (`import asyncio`) | **Runs** the pausable functions: the event loop, plus tools like `asyncio.run`, `sleep`, `gather`, `create_task`, `Queue`, `Event` |

`async`/`await` let you **write** pausable functions. `asyncio` is the
**machine that runs** them. (uvloop, which vLLM uses, is a faster
replacement for asyncio's loop. It does the same job.)

**Calling an `async def` function does not run it:**

```python
import asyncio

async def hello():
    print("hi")

x = hello()        # nothing is printed!
print(x)           # <coroutine object hello at 0x...>  (a paused-at-the-start function)
asyncio.run(x)     # NOW "hi" is printed
```

**There are three ways to actually run a coroutine object:**

| Way | What happens |
|---|---|
| `await x` (inside another async function) | `x` runs **as part of the current task**. No new task is created |
| `asyncio.create_task(x)` | `x` becomes its **own task** that the loop juggles separately |
| `asyncio.run(x)` | Starts the event loop with `x` as the first task |

**The loop only juggles tasks.** A chain of `await`s is one task. vLLM has
hundreds of `async def` functions, but for one HTTP request the loop sees
**one task**. Inside that task, `create_chat_completion` awaits
`_create_chat_completion`, which awaits `render_chat_request`, and so on.
When the deepest `await` pauses, the whole chain pauses.

**Which task runs next? Three rules:**

1. **No interrupting.** The running task keeps going until it reaches an
   `await` that has to wait, or finishes. The loop never stops it from outside.
2. **First come, first served.** Tasks that are ready to run wait in a line
   (the `_ready` queue). The loop takes them from the front, in order.
3. **Woken tasks go to the back of the line.** When a task's waited-for result
   arrives, it does **not** jump ahead. It joins the end of the line. With many
   results, the order is the order the results arrived (timers: the earliest
   due first). There are **no priorities** in asyncio.

**Who starts the event loop?** `async def` needs no import, but async
functions alone never run. Exactly **one** place at the top of every program
starts a loop, and everything else is awaited inside it:

| Where my code runs | How my async function gets run |
|---|---|
| A plain `.py` script | `asyncio.run(main())` once, at the bottom |
| Inside another `async def` | `await f()` |
| Colab / Jupyter | `await f()` directly in a cell. The notebook already runs a loop, so `asyncio.run(...)` fails there with "cannot be called from a running event loop" |
| A FastAPI route | Just write `async def`. Uvicorn starts the loop and FastAPI awaits the route for me |
| vLLM | `uvloop.run(run_server(args))` in `vllm/entrypoints/cli/serve.py:152`, once |

**Common bug:** forgetting `await` inside an async function.
`user = get_user()` gives `user` a coroutine object, not the data, and Python
warns `RuntimeWarning: coroutine 'get_user' was never awaited`.

**A whole event loop in 15 lines (a toy, to remove the magic):**

```python
import collections
import types

@types.coroutine
def pause():
    yield                              # hand control back to whoever called send()

async def worker(name, steps):
    for i in range(steps):
        print(f"{name} step {i}")
        await pause()                  # "I can wait; let someone else run"
    return f"{name} finished"

def tiny_event_loop(coros):
    ready = collections.deque(coros)   # the line of tasks
    while ready:
        coro = ready.popleft()         # take the first in line
        try:
            coro.send(None)            # run it until its next await
            ready.append(coro)         # it paused -> back of the line
        except StopIteration as done:  # it returned
            print(done.value)

tiny_event_loop([worker("A", 2), worker("B", 3)])
```

Output: `A step 0`, `B step 0`, `A step 1`, `B step 1`, `A finished`,
`B step 2`, `B finished`. The tasks take turns, the line is first in first
out, and nobody is interrupted. The real asyncio adds two things on top:
timers and sockets (the selector), and Futures, so a task rejoins the line
only when the thing it waits for is actually finished.

### 1.1 The idea

Most of a web server's time is spent **waiting**: for the network, for a
database, for the GPU. While one request waits, the server could be working
on another one.

**Chef analogy.** One chef (one thread) is cooking ten dishes.

- **A good chef:** puts dish 1 in the oven, then chops vegetables for dish 2
  while the oven works. When the oven beeps, they go back to dish 1.
- **A bad chef:** puts dish 1 in the oven and **stands staring at the oven**
  for 20 minutes. Nine customers wait for nothing.

asyncio is the system that makes Python the good chef.

| Word | Plain meaning |
|---|---|
| **event loop** | The chef. One loop, one thread. It keeps a list of tasks and runs whichever one is ready. |
| **coroutine** (`async def`) | A recipe that can pause in the middle. Calling it does not run it yet; it gives you a coroutine object. |
| **`await`** | "I'm waiting for something here; chef, go do other work and come back when it's ready." **This is the only place the loop can switch to another task.** |
| **task** (`asyncio.create_task`) | A coroutine handed to the loop to run in the background. |
| **`asyncio.gather`** | Run several coroutines **concurrently** (taking turns on the same one thread, so their *waiting* overlaps) and wait for all of them. Still not parallel. |
| **async generator** | An `async def` with `yield` in it. It produces values one at a time, with waiting in between. You read it with `async for`. |

**Important:** asyncio is **concurrency, not parallelism**. There is still only
one thread. It does many things *in turns*, never two at the exact same moment.
It helps when the work is **waiting** (I/O-bound). It does **not** help when the
work is **heavy computing** (CPU-bound).

### Under the hood: how asyncio actually runs (diagrams)

The table above is the "what". These diagrams are the "how": what Python
really does when you write `await`. The names (`_ready`, `__step`,
`__wakeup`) are the real internal names in CPython's `asyncio` source.

#### A. Mind map: the six pieces and how they connect

```
asyncio
├── Event loop                       the chef: ONE thread, runs forever
│   ├── _ready       queue of callbacks to run NOW
│   ├── _scheduled   heap of timers             <- asyncio.sleep, call_later
│   └── selector     asks the OS "which sockets are ready?"
│                    (epoll on Linux, IOCP on Windows, libuv in uvloop)
│
├── Coroutine        async def f(): ...
│   ├── f() does NOT run it, it only creates a coroutine object
│   ├── it runs only when a Task calls  coro.send(None)
│   └── it stops at an `await` on an unfinished Future; its frame is frozen
│
├── Future           an empty box for a result that arrives later
│   ├── fut.set_result(x)  -> puts fut's callbacks on _ready
│   └── `await fut`        -> if fut is not done, passes fut UP to the Task
│
├── Task             a Future that drives one coroutine
│   ├── create_task(coro)  -> loop.call_soon(task.__step)
│   ├── __step             -> coro.send(None) until the coroutine yields a Future
│   └── __wakeup           -> runs when that Future is done, calls __step again
│
├── gather(c1, c2, c3)
│   ├── wraps each coroutine in its own Task
│   └── returns ONE Future, done when ALL tasks are done (results in order)
│
└── Async generator  async def + yield
    ├── `async for x in agen`  -> repeatedly  await agen.__anext__()
    ├── `yield v`  inside      -> hands v to the CONSUMER, pauses
    └── `await`    inside      -> pauses back to the EVENT LOOP
```

**One sentence:** the loop runs callbacks; a Task's callback (`__step`) runs a
coroutine until it awaits an unfinished Future; when that Future gets its
result, it schedules the Task's next `__step`.

#### B. One turn of the event loop

The loop repeats this forever (`BaseEventLoop._run_once`):

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ 1. How long may I wait?                                           │
 │      _ready is not empty      -> 0 (don't wait at all)            │
 │      else a timer is pending  -> until the nearest timer          │
 │      else                     -> forever (until I/O arrives)      │
 ├──────────────────────────────────────────────────────────────────┤
 │ 2. selector.select(timeout)                                       │
 │      The ONLY place the loop waits. The OS wakes it when a        │
 │      socket has data. Ready sockets -> their callbacks to _ready  │
 ├──────────────────────────────────────────────────────────────────┤
 │ 3. Timers whose time has come -> move from _scheduled to _ready   │
 ├──────────────────────────────────────────────────────────────────┤
 │ 4. Run the callbacks in _ready, ONE AT A TIME.                    │
 │      Most callbacks are task.__step: run a coroutine until its    │
 │      next await on an unfinished Future, then return.             │
 │                                                                   │
 │      !! If a callback never returns (time.sleep, heavy CPU),      │
 │      !! steps 1-3 never happen again -> EVERYTHING freezes.       │
 └─────────────────────────────────┬────────────────────────────────┘
                                   └──> back to step 1
```

This is why "blocking freezes everything" is not a rule to memorise. It
follows from step 4: the loop can't do anything else until the current
callback returns.

#### C. What `await asyncio.sleep(2)` does, step by step

```
 YOUR COROUTINE                 TASK                        EVENT LOOP
 ──────────────                 ────                        ──────────
 await asyncio.sleep(2)
   sleep() creates Future f
   loop.call_later(2, f.set_result) ─────────────────────► timer goes in _scheduled
   await f: f not done, so yield f
   └────── f travels UP ────────► __step receives f
                                  f.add_done_callback(__wakeup)
                                  return ────────────────► loop is FREE:
                                                           runs other tasks
                                                           ... 2 s later ...
                                                           timer due: f.set_result(None)
                                                           f's callbacks go to _ready
                                  __wakeup -> __step ◄──── runs __wakeup
   ◄────── coro.send(None) ────── coroutine resumes
 await returns, the next line runs
```

Any `await` on real I/O (a socket read, an HTTP call, a ZMQ message) works the
same way. The only difference is **who** completes the Future: a timer (sleep),
the selector (socket data), or another thread (section F).

`await asyncio.sleep(0)` is a special case: it yields to the loop **without** a
Future, so the Task goes straight back on `_ready`. It means "let everyone else
who is ready run first, then continue with me." vLLM uses exactly this
(`async_llm.py:723`).

#### D. Timeline: the tea example on one thread

```
 with  await asyncio.sleep(2)            (gather: 3 tasks)

 time     0 s                                                 2 s
 thread   [A: kettle on][B: kettle on][C: kettle on]  idle   [A ready][B ready][C ready]
 loop      each task hits await -> registers a timer -> returns   timers fire
           selector.select() waits for 2 s with nothing to do        TOTAL ≈ 2 s


 with  time.sleep(2)

 time     0 s               2 s               4 s               6 s
 thread   [A: on ...sleep... ready][B: on ...sleep... ready][C: on ...sleep... ready]
 loop      never reaches select(): no other task, timer or socket is handled  TOTAL ≈ 6 s
```

#### E. Async generator: two directions of pausing

```
 CONSUMER                                    ASYNC GENERATOR  fake_llm()
 ────────                                    ───────────────────────────
 async for tok in fake_llm():
   await agen.__anext__() ─────────────────► starts / resumes
                                               await asyncio.sleep(0.3)
                                               └─► pauses to the EVENT LOOP
                                                   (the consumer waits too)
                                               ... 0.3 s later, resumed ...
   tok = "Hello" ◄──────────────────────────── yield "Hello"
                                               (pauses, value goes to CONSUMER)
   loop body runs with tok
   await agen.__anext__() ─────────────────► resumes right after the yield
   ...                                          ...
   loop ends ◄──── StopAsyncIteration ──────── function returns
```

**Remember:** inside an async generator, **`await` talks to the event loop**,
**`yield` talks to the consumer**.

#### F. `run_in_executor` / `to_thread`: sending blocking work away

```
 EVENT LOOP THREAD                             WORKER THREAD (ThreadPoolExecutor)
 ─────────────────                             ──────────────────────────────────
 ids = await loop.run_in_executor(ex, tokenize, text)
   executor.submit(tokenize, text) ──────────► tokenize(text) runs here
   wrap it in an asyncio Future f              (blocking is fine on this thread)
   await f -> the Task pauses                          │
   the loop stays FREE: other requests run             │ finished
                                               loop.call_soon_threadsafe(set f's result)
                                               + writes 1 byte to the loop's
                                                 "self-pipe" to wake select()
   select() wakes up ◄─────────────────────────────────┘
   f is done -> __wakeup -> the Task resumes
 ids is ready
```

**Limit:** this frees the **loop**, but not the **GIL**. If `tokenize` is pure
Python, it still competes with the loop thread for the GIL. That's why vLLM
puts the engine in a separate process, not a thread.

#### G. Cancellation

```
 task.cancel()
   -> the Future the task is waiting on is cancelled
   -> the task's __step runs with CancelledError
   -> coro.throw(CancelledError): raised AT the `await` line in your code
   -> your `except CancelledError:` / `finally:` blocks run (clean up here)
   -> re-raise, and the task ends as "cancelled"
```

#### H. The same machinery inside vLLM's API server

vLLM starts its server with `uvloop.run(run_server(args))`
(`vllm/entrypoints/cli/serve.py:152`). uvloop is a faster event loop written
on top of libuv (C), but it works exactly like diagram B. Uvicorn creates
**one Task per HTTP request**. While serving two streaming requests, the loop
looks like this:

```
 ONE uvloop event loop (API server process, one thread)
 │
 ├── Task: HTTP request 1
 │     StreamingResponse -> chat_completion_stream_generator -> generate()
 │     paused at:  await q.get()                       (waiting on its own mailbox)
 │
 ├── Task: HTTP request 2
 │     same chain, paused at:  await q.get()
 │
 ├── Task: process_outputs_socket          (core_client.py, AsyncMPClient)
 │     paused at:  await output_socket.recv_multipart()   (ZMQ socket I/O, step 2 in B)
 │     on data: decode, put on outputs_queue
 │
 ├── Task: output_handler                  (async_llm.py:747, create_task)
 │     paused at:  await engine_core.get_output_async()
 │     on data: process_outputs -> q.put() for each request
 │              q.put() sets an asyncio.Event -> wakes request 1's and 2's tasks
 │              await asyncio.sleep(0) between slices (diagram C, last note)
 │
 └── Thread pool (off the loop)            make_async = run_in_executor (diagram F)
       chat template, tokenising           (renderers/base.py:96, renderers/hf.py:926)

 Client disconnects -> Starlette cancels request 1's task (diagram G)
   -> generate() catches CancelledError -> self.abort(...)   (async_llm.py:618)
```

**The chain of one token, in loop terms:** ZMQ data arrives → `select` wakes
(B step 2) → `process_outputs_socket` resumes → puts on `outputs_queue` →
`output_handler` resumes → `q.put()` → the request's Task resumes at
`await q.get()` → the stream generator yields `data: ...\n\n` →
`StreamingResponse` writes it to the socket.

### 1.2 General example: waiting vs blocking

```python
import asyncio
import time

async def make_tea(name: str) -> None:
    print(f"{name}: kettle on")
    await asyncio.sleep(2)          # waiting for water -> the loop runs others
    print(f"{name}: tea ready")

async def main() -> None:
    start = time.perf_counter()
    await asyncio.gather(make_tea("A"), make_tea("B"), make_tea("C"))
    print(f"total: {time.perf_counter() - start:.1f} s")

asyncio.run(main())
```

Output: all three kettles go on, about 2 seconds pass, all three are ready.
**Total: about 2.0 s.**

Now change one line, `await asyncio.sleep(2)` to `time.sleep(2)`:

- `time.sleep` does not `await`, so the chef **stands and stares**.
- A finishes completely, then B, then C. **Total: about 6.0 s.**
- In a web server, this means **every other request freezes** for those
  seconds, not just this one.

### 1.3 Why one blocking call freezes everything

There is **one** loop. The loop can only switch tasks at an `await`. If a task
runs for 2 seconds without reaching an `await`, nothing else runs for those
2 seconds: no other request, no other stream, no health check.

**Common blocking calls hiding inside `async def`:**

- `time.sleep(...)`: use `await asyncio.sleep(...)` instead.
- `requests.get(...)`: use an async client such as `httpx.AsyncClient`.
- A sync database driver: use an async driver.
- Heavy CPU work: tokenising a huge prompt, building a huge JSON, image
  processing.

**The three fixes:**

1. **Use the async version of the library**, so you can `await` it.
2. **Move a blocking call to a thread:** the loop stays free while a thread
   waits.
   ```python
   ids = await asyncio.to_thread(slow_tokenise, text)
   # older style, same idea:
   # ids = await loop.run_in_executor(executor, slow_tokenise, text)
   ```
3. **Move heavy CPU work to another process.** Python's **GIL** (global
   interpreter lock) lets only one thread run Python code at a time, so threads
   don't help much with pure-Python heavy computing. A separate process has its
   own GIL.

### 1.4 General example: an async generator

```python
import asyncio

async def fake_llm(prompt: str):
    for token in ["Hello", " there", "!"]:
        await asyncio.sleep(0.3)    # pretend the GPU is computing
        yield token                 # hand out one piece, then pause here

async def main() -> None:
    async for token in fake_llm("hi"):
        print(repr(token))          # arrives one by one, 0.3 s apart

asyncio.run(main())
```

This is exactly the shape of token streaming: **produce a piece, hand it out,
wait for the next one.**

### 1.5 Cancellation

When a task is cancelled (for example, the client closed the browser tab),
asyncio raises `asyncio.CancelledError` **inside** that task, at the `await`
where it was paused. Good code catches it to clean up: close connections and
stop expensive work.

### 1.6 Where it lives in vLLM

| What | Where | Why it matters |
|---|---|---|
| Blocking function moved to a thread | `make_async` in `vllm/utils/async_utils.py:28` uses `loop.run_in_executor` | The general fix #2 above, written once and reused |
| Tokenising off the loop | `renderers/base.py:96`: `self._tokenize_prompt_async = make_async(self._tokenize_prompt, executor=self._executor)` | A long prompt does not freeze other requests (mostly; see the GIL note in `lifecycle.md`) |
| Chat template off the loop | `renderers/hf.py:926`: `_apply_chat_template_async = make_async(safe_apply_chat_template, ...)` | Same reason |
| One background task for all outputs | `async_llm.py:747`: `asyncio.create_task(output_handler())` | One "sorter" receives every engine output and fills each request's mailbox |
| Politely giving the loop a turn | `async_llm.py:723`: `await asyncio.sleep(0)` between output slices | `sleep(0)` means "pause for zero seconds, but let other tasks run first", so a big batch doesn't hog the loop |
| Skipping a task switch when possible | `async_llm.py:607`: `out = q.get_nowait() or await q.get()` | If the mailbox already has something, take it without waiting |
| Async generators | `AsyncLLM.generate` (`async_llm.py:550`) and `chat_completion_stream_generator` (`serving.py:431`) | The whole stream is a chain of async generators |
| Waiting for "work done" OR "client left" | `with_cancellation` in `entrypoints/serve/utils/api_utils.py:51`: `asyncio.wait([...], return_when=FIRST_COMPLETED)` | Whichever finishes first wins; the other task is cancelled |
| Cleaning up on cancel | `async_llm.py:618`: `except (asyncio.CancelledError, GeneratorExit): await self.abort(...)` | Client disconnects → request aborted → GPU stops working on it and its KV blocks are freed |
| A whole process instead of threads | The engine core runs in its own process (`core.py`, `EngineCoreProc`) | The GIL: scheduling and launching GPU work must not fight with HTTP and JSON work for one lock |

### 1.7 Interview questions

**Q: What is the event loop?**
A: A single-threaded scheduler. It runs one task until that task hits an
`await`, then switches to another task that is ready. It is how one thread
serves many requests that spend most of their time waiting.

**Q: Concurrency vs parallelism?**
A: Concurrency is handling many things by taking turns. Parallelism is doing
many things at the exact same moment on different cores. asyncio gives
concurrency. Processes (or GPUs) give parallelism.

**Q: Why does one `time.sleep` in an async route slow down the whole server?**
A: There is one event loop, and it can only switch at an `await`. `time.sleep`
never awaits, so the loop is stuck and every other request waits too.

**Q: How do you call a blocking library from async code?**
A: Use an async version if there is one. Otherwise run it in a thread with
`asyncio.to_thread` or `run_in_executor`. For heavy CPU work, use a separate
process because of the GIL. vLLM does all three: async ZMQ sockets, tokenising
in a thread pool, and the engine in its own process.

**Q: What is an async generator, and why is it useful for LLM streaming?**
A: An `async def` that uses `yield`. It hands out values one at a time and can
wait in between. That matches token streaming: wait for the next token, yield
it, repeat.

---

## 2. FastAPI async: `async def` vs `def`, and `StreamingResponse`

### 2.1 The idea

FastAPI runs on an **ASGI server** (usually **Uvicorn**), which runs **one
event loop**. FastAPI treats your route functions differently depending on how
you write them:

| You write | FastAPI runs it | Use it when | The danger |
|---|---|---|---|
| `async def route()` | **Directly on the event loop** | Everything inside is awaitable (async DB, async HTTP, `asyncio.sleep`) | One blocking call freezes **all** requests |
| `def route()` | **In a thread pool** (40 threads by default) | You must call blocking libraries (sync DB driver, `requests`) | Too many slow requests fill the pool, and the rest queue |

**Rule of thumb:** use `async def` only if you `await` everything slow inside it.
If you can't, use plain `def` and let FastAPI put it in a thread.

### 2.2 General example: three routes

```python
import asyncio
import time

from fastapi import FastAPI

app = FastAPI()

@app.get("/good")
async def good():
    await asyncio.sleep(2)      # the loop is free while waiting
    return {"ok": True}

@app.get("/bad")
async def bad():
    time.sleep(2)               # blocks the ONE event loop: everyone freezes
    return {"ok": True}

@app.get("/fine")
def fine():
    time.sleep(2)               # plain def: FastAPI runs it in a thread
    return {"ok": True}
```

Try sending 5 requests at once to each route:

```bash
for i in 1 2 3 4 5; do curl -s localhost:8000/bad > /dev/null & done; time wait
```

- `/good`: about **2 s** total, because they all wait together.
- `/bad`: about **10 s** total, because they run one after another.
- `/fine`: about **2 s** total, because each runs in its own thread.

### 2.3 `StreamingResponse`

A normal response builds the whole body, then sends it. `StreamingResponse`
takes a **generator** and sends **each piece the moment it is yielded**.

- With an **async generator**, it runs on the event loop.
- With a **normal (sync) generator**, Starlette pulls each item in a thread so
  the loop is not blocked.
- `media_type="text/event-stream"` tells the client "this is SSE" (section 3).

### 2.4 General example: a minimal LLM-style streaming endpoint

```python
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ChatIn(BaseModel):
    prompt: str

async def fake_llm(prompt: str):
    for token in ["Hello", " there", "!"]:
        await asyncio.sleep(0.3)
        yield token

def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"

@app.post("/v1/chat")
async def chat(body: ChatIn):
    async def stream():
        first = True
        async for token in fake_llm(body.prompt):
            if first:                       # role chunk only once token 1 exists
                yield sse({"delta": {"role": "assistant", "content": ""}})
                first = False
            yield sse({"delta": {"content": token}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

Run it with `uvicorn app:app --port 8000`, then:

```bash
curl -N -X POST localhost:8000/v1/chat -H "Content-Type: application/json" -d '{"prompt":"hi"}'
```

`-N` turns off curl's buffering, so you see each chunk as it arrives.

**Connect to Saturday:** if I had put the role chunk *before* the `async for`,
it would go out before any token exists. That is exactly the "Case A" worry.
vLLM puts it *inside* the loop, so it waits for token 1, just like this example.

### 2.5 Where it lives in vLLM

- **The route:** `create_chat_completion` in
  `entrypoints/openai/chat_completion/api_router.py:53` is an **`async def`**.
- **The body:** parsed into the pydantic model `ChatCompletionRequest`
  (`chat_completion/protocol.py:213`). Bad JSON is rejected before my code runs.
- **Checks before the route:** `Depends(validate_json_request)` runs first.
- **Three possible returns** (`api_router.py:63-74`):
  - an error → `JSONResponse` with the error code
  - a full answer (`stream=False`) → `JSONResponse`
  - a generator (`stream=True`) → `StreamingResponse(content=generator, media_type="text/event-stream")`
- **The server:** Uvicorn, started in `vllm/entrypoints/launcher.py:85`
  (`uvicorn.Config(app, ...)`).
- **Why `async def` is safe here:** the slow parts are either awaited (engine
  outputs over async ZMQ) or pushed to threads (templating, tokenising).

### 2.6 Debugging: "my stream arrives all at once at the end"

Check these, in order:

1. **Not actually streaming:** the route builds the full answer and returns it,
   instead of returning a `StreamingResponse` with a generator.
2. **The client isn't streaming:** Python `requests` without `stream=True`,
   or calling `.json()` / `.text` (which waits for the end). curl without `-N`.
3. **A proxy is buffering:** nginx and some load balancers collect the whole
   response before forwarding it. For nginx: `proxy_buffering off;` or send the
   header `X-Accel-Buffering: no`.
4. **Compression middleware:** gzip can hold small chunks back to compress
   them together.
5. **Missing `\n\n`:** the client's SSE parser waits for the blank line that
   ends a message, so nothing seems to arrive.

### 2.7 Interview questions

**Q: When do you use `async def` vs `def` in FastAPI?**
A: `async def` when everything slow inside is awaitable; it runs on the event
loop. Plain `def` when I must call blocking libraries; FastAPI runs it in a
thread pool. The mistake to avoid is a blocking call inside `async def`, which
freezes every request.

**Q: How do you stream tokens from a FastAPI endpoint?**
A: Return a `StreamingResponse` with an async generator and
`media_type="text/event-stream"`. The generator yields `data: <json>\n\n` per
token and ends with `data: [DONE]\n\n`.

**Q: A user closes the tab mid-stream. What should happen, and how does vLLM do it?**
A: The server should stop generating, so it doesn't waste GPU time. Starlette
notices the disconnect and cancels the generator. In vLLM,
`AsyncLLM.generate` catches `CancelledError`/`GeneratorExit` and calls `abort`,
so the engine drops the request and frees its KV blocks.

---

## 3. Server-Sent Events (SSE): a one-way stream over plain HTTP

### 3.1 The idea

SSE is a very simple standard: the server keeps an HTTP response open and
writes **text messages** into it over time. The client reads them as they
arrive.

**The format rules:**

- Each line is `field: value`. The common field is `data:`.
- **A blank line ends one message.** That's why every vLLM chunk ends in `\n\n`.
- Other fields exist: `event:` (a message type name), `id:` (for resuming),
  `retry:` (reconnect delay).
- A line starting with `:` is a **comment**, often used as a keep-alive ping.
- Response header: `Content-Type: text/event-stream`.
- `data: [DONE]` is **not** part of the SSE standard. It's an OpenAI API
  convention for "the stream is finished", and vLLM copies it.

### 3.2 Example: the raw stream from vLLM

```
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" there"}}]}

data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}}

data: [DONE]

```

### 3.3 Reading a stream on the client

**In a browser,** the built-in `EventSource` reads SSE and reconnects
automatically. But it only supports **GET** requests. LLM APIs use **POST**
(the prompt is in the body), so real clients read the stream with `fetch` or
an HTTP library and parse the lines themselves.

**In Python (general example):**

```python
import json
import time

import requests

start = time.perf_counter()
ttft = None
with requests.post("http://localhost:8000/v1/chat", json={"prompt": "hi"}, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        if not line or line.startswith(":"):
            continue                        # blank separator, or an SSE comment
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if ttft is None:
            ttft = time.perf_counter() - start
            print(f"TTFT: {ttft * 1000:.0f} ms")
        print(chunk["delta"].get("content", ""), end="", flush=True)
print()
```

This is a small version of what `vllm bench serve` does.

### 3.4 Why SSE and not WebSockets for token streaming?

| | SSE | WebSockets |
|---|---|---|
| Direction | Server → client only | Both ways |
| What we need for an LLM answer | One request in, a stream of tokens out. **One way is enough.** | More than we need |
| Protocol | Plain HTTP response | Separate protocol after an "upgrade" handshake |
| Works with proxies, load balancers, auth headers, HTTP/2 | Yes, it's just HTTP | Often needs extra setup |
| Server complexity | A generator that yields text | Long-lived stateful connections, message framing, pings |
| Reconnect | `EventSource` does it for you | You write it yourself |
| Downsides | Text only; one-way; browsers allow only ~6 open connections per domain over HTTP/1.1 | More moving parts |

**One-line answer:** an LLM response is a one-way stream that ends when the
answer ends. SSE is exactly that, over plain HTTP. WebSockets add two-way
complexity we don't use.

(WebSockets *do* make sense for things like real-time voice, where the client
keeps sending data while the server answers.)

### 3.5 Gotcha: the network doesn't respect message boundaries

TCP delivers **bytes**, not messages. One read might contain **two messages
glued together**, or **half of one**. So the client must **buffer** bytes and
cut only at the blank line.

### 3.6 Where it lives in vLLM

- **Writing SSE:** `chat_completion_stream_generator` (`serving.py:431`) does
  `yield f"data: {data}\n\n"` for every chunk, then `yield "data: [DONE]\n\n"`
  (`serving.py:886`).
- **The role chunk:** sent once, inside `if first_iteration:` (`serving.py:507`).
  See Saturday's notes for why it can't go out before token 1.
- **Reading SSE in the benchmark:** `StreamedResponseHandler.add_chunk`
  (`vllm/benchmarks/lib/endpoint_request_func.py`) keeps a buffer and splits on
  `"\n\n"`. This is section 3.5 in practice. The request function also skips
  lines starting with `:` (comments).

### 3.7 Interview questions

**Q: What does an SSE message look like?**
A: One or more `field: value` lines, usually `data: <payload>`, ended by a blank
line. The response has `Content-Type: text/event-stream`.

**Q: Why SSE instead of WebSockets for streaming LLM output?**
A: The output is one-way and ends when the answer ends. SSE is plain HTTP, so
it works with normal proxies, auth and load balancers, and the server code is
just a generator. WebSockets are two-way and stateful, which adds complexity
we don't need.

**Q: Why can't the client just split on every network read?**
A: TCP doesn't keep message boundaries. A read can hold half a message or two
messages, so the client buffers and splits on the blank line.

---

## 4. Tokenisation: text → numbers

### 4.1 The idea

A model can only work with **numbers**. The tokeniser cuts text into pieces
called **tokens** and gives each piece an id from a fixed list (the
**vocabulary**). Qwen2.5's vocabulary has about 151,000 entries.

Why pieces, and not letters or whole words?

- **Letters:** sequences become very long, so the model is slow and has to
  learn spelling from scratch.
- **Whole words:** the vocabulary would be huge, and new words ("vLLM",
  "Karachi2026", typos) would be unknown.
- **Subword pieces:** common words are one token; rare words are built from a
  few pieces. A good balance.

### 4.2 BPE (Byte Pair Encoding), step by step

BPE **learns** the pieces from a lot of text:

1. Start with the smallest units (bytes).
2. Count which **pair** of neighbouring units appears most often.
3. **Merge** that pair into a new unit and add it to the vocabulary.
4. Repeat until the vocabulary is the size you want.

**Tiny worked example** with the text `aaabdaaabac`:

```
start:                      a a a b d a a a b a c
most common pair "aa" → Z:  Z a b d Z a b a c
most common pair "ab" → Y:  Z Y d Z Y a c
most common pair "ZY" → X:  X d X a c
```

The text went from 11 units to 5. On real data, after tens of thousands of
merges, common words like `" the"` become single tokens.

### 4.3 Byte-level BPE (what Qwen, GPT and Llama use)

The starting units are the **256 possible bytes**, not letters. So:

- **Any text can be encoded**, in any language, with no "unknown" token,
  because everything is bytes in the end.
- **But one character can be several bytes**, so it can be split across tokens:

```
"a"   = 1 byte    61
"ا"   = 2 bytes   D8 A7          (Urdu/Arabic alef)
"😀"  = 4 bytes   F0 9F 98 80
```

If a character's bytes are split over two tokens, **the first token alone is
not valid text**. Decoding it gives `�` (the replacement character). This is
why vLLM's detokeniser sometimes emits `""` for a token and waits for the next
one (Thursday Q4).

### 4.4 Token count ≠ word count

```
"Hello there!"   ->  ["Hello", " there", "!"]          2 words, 3 tokens
"unbelievable"   ->  maybe ["un", "belie", "vable"]    1 word,  several tokens
" there"         ->  the SPACE is part of the token
"12345678"       ->  numbers are often split into small groups of digits
Urdu, Hindi, ... ->  usually MORE tokens per word than English
```

**Rough rule for English:** 1 token ≈ 4 characters ≈ ¾ of a word.

**Try it in Colab** (CPU is fine; it downloads only the tokeniser files):

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

for text in ["Hello there!", "unbelievable", "tokenisation", "12345678", "پاکستان", "😀🦙"]:
    ids = tok(text)["input_ids"]
    print(f"{text!r:16} words={len(text.split())}  tokens={len(ids)}  {tok.convert_ids_to_tokens(ids)}")
```

You'll see a strange `Ġ` in some pieces. That is how byte-level BPE *displays*
a space, so `Ġthere` means `" there"`.

To see a "half character" token:

```python
ids = tok("🦙")["input_ids"]
for i in range(1, len(ids) + 1):
    print(ids[:i], repr(tok.decode(ids[:i])))   # partial bytes decode to '�'
```

(If that emoji happens to be a single token, try a few other emojis or Urdu
words until one splits.)

### 4.5 Why tokens matter for an engineer

- **Cost:** APIs charge per token, not per word.
- **Context limit:** `--max-model-len 8192` is 8192 **tokens**, prompt plus
  output.
- **Latency:** prefill time grows with **input tokens** (TTFT); decode takes
  one step per **output token** (ITL).
- **My benchmarks:** "512 in / 256 out" means tokens, not words.
- **Prefix caching:** it works in **16-token blocks**, so token boundaries
  decide what is shared (Saturday).

### 4.6 Karpathy's video: the takeaways to remember

Many "LLM weirdness" problems come from tokenisation, not from the model:

- Bad at **spelling and reversing words**: the model sees `"unbelievable"` as a
  few chunks, not letters.
- Bad at **arithmetic**: digits are grouped in uneven ways.
- **Worse in non-English languages:** more tokens per word, so less fits in the
  context, and it costs more.
- **Trailing spaces** in a prompt can change the answer, because a space is
  usually glued to the *next* word's token.

### 4.7 Where it lives in vLLM

- **Tokenise:** `_tokenize_prompt` (`renderers/base.py:472`) calls
  `tokenizer(prompt["prompt"], ...)`. It runs in a thread pool (section 1).
- **Special tokens:** for chat, `default_chat_tok_params` uses
  `add_special_tokens=False` (`renderers/base.py:358-371`). The chat template
  already wrote the special tokens into the text, so adding them again would
  duplicate them (see 5.5).
- **Only ids cross to the engine:** the engine never sees text (Wednesday).
- **Detokenise incrementally:** `detokenizer.update` → `decode_next`
  (`v1/engine/detokenizer.py`), one token at a time, holding back partial
  characters (Thursday Q4).

### 4.8 Interview questions

**Q: What is a token, and why isn't it the same as a word?**
A: A piece of text from the tokeniser's vocabulary. Common words are one token;
rare words, numbers and other languages split into several. Spaces are part
of tokens. In English, a token is roughly ¾ of a word.

**Q: Explain BPE in 30 seconds.**
A: Start from bytes. Repeatedly find the most frequent neighbouring pair in
the training text and merge it into a new token. Stop at the target
vocabulary size. Frequent words end up as single tokens.

**Q: Why can one token produce no visible text?**
A: In byte-level BPE a token is a run of bytes. A character like an emoji is
several bytes, which can be split across tokens. The first part alone isn't
valid UTF-8, so the detokeniser waits for the rest.

**Q: Why does tokenisation matter for latency and cost?**
A: Cost and context limits are counted in tokens. Prefill time grows with input
tokens, and decode takes one step per output token.

---

## 5. Chat templates: a list of messages → one string

### 5.1 The idea

An API receives a **list of messages**:

```json
[{"role": "system", "content": "Be brief."},
 {"role": "user",   "content": "What is Python?"}]
```

But the model was trained on **one long string** with special marker tokens
showing who is speaking. The **chat template** converts the list into that
exact string. Each model family has its **own** format, and the model only
behaves well with the format it was trained on.

### 5.2 Example: what Qwen2.5 produces (from my Colab run)

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
```

- `<|im_start|>` and `<|im_end|>` are **special tokens**: one token each, never
  split.
- No system message was sent, so the template **added Qwen's default system
  prompt**.
- The last line, `<|im_start|>assistant\n`, comes from
  `add_generation_prompt=True`. It tells the model: "now it's your turn,
  answer as the assistant."

**Other families use different formats** (simplified):

```
Llama 3:  <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
Mistral:  <s>[INST] What is Python? [/INST]
```

### 5.3 Where the template comes from

It is a **Jinja** template (a small text-template language), stored in the
model's `tokenizer_config.json` under the key `chat_template`. A simplified
ChatML version looks like this:

```jinja
{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
```

The real Qwen template also handles tools and the default system prompt.

### 5.4 The template costs tokens

Template tokens are **real prompt tokens**. They take prefill time and count
toward the context limit.

**My own data proves it.** Week 3 recorded 54,100 prefix-cache queries for 100
prompts, which is **541 tokens per request = 512 random tokens + 29 template
tokens**. And 29 is exactly the template size I counted in Colab on Saturday.

### 5.5 Mistakes and debugging

- **Wrong template** (for example, a Llama format on a Qwen model): answers get
  worse or strange, and the model may not stop where it should.
- **A base model instead of an instruct model:** base models were never trained
  on a chat format, so they just continue the text.
- **Missing `add_generation_prompt`:** the model may continue the *user's*
  message instead of answering it.
- **Double BOS token:** the template writes a beginning-of-sequence token, and
  then tokenising with `add_special_tokens=True` adds a second one. Quality can
  drop quietly. (Qwen has no BOS token, so it is not affected, but Llama is.)
- **Unknown roles:** some templates don't accept a `developer` or `tool` role.

### 5.6 Where it lives in vLLM

- **Templating is done on text, then tokenised separately:** `preprocess_chat`
  sets `tokenize=False` for normal HF tokenisers (`renderers/online_renderer.py:379`).
- **Applying it:** `safe_apply_chat_template` in `renderers/hf.py` finds the
  template (`resolve_chat_template`), turns `developer` messages into `system`
  if the template has no developer role, and calls the HF
  `tokenizer.apply_chat_template(...)`.
- **Overriding it:** `vllm serve ... --chat-template <file or string>` replaces
  the model's built-in template.
- **The security link:** the template is identical for every request, so
  **block 0 is shared by every tenant** that sends no system prompt (Saturday's
  security line).

### 5.7 Interview questions

**Q: What is a chat template?**
A: A Jinja template, shipped with the tokeniser, that turns a list of chat
messages into the exact string format the model was trained on, with special
tokens marking each role.

**Q: The model keeps writing the user's next message instead of answering. Why?**
A: Most likely the generation prompt is missing (`add_generation_prompt`), so
the string doesn't end with "assistant's turn". Or the wrong template or a base
model is being used.

**Q: Why might the same question cost a different number of tokens on two
platforms?**
A: Different tokenisers split text differently, and different templates add
different numbers of special and system tokens.

---

## 6. The vLLM V1 big picture

### 6.1 The processes

| Process | Does | Code |
|---|---|---|
| **API server** (frontend) | HTTP, chat template, tokenise, detokenise, build SSE chunks | `entrypoints/openai/api_server.py`, `v1/engine/async_llm.py` |
| **Engine core** | Busy loop: schedule → run model → update; owns the KV cache manager | `v1/engine/core.py` |
| **GPU worker(s)** | Load weights, run the forward pass on a GPU | `v1/worker/gpu_worker.py` |

- They talk with **ZMQ** sockets carrying **msgpack** bytes (Wednesday).
- The official docs say "1 worker per GPU". **On my 1-GPU setup there are only
  2 processes:** with one GPU, `vllm/config/parallel.py:956` picks the `"uni"`
  executor, and `UniProcExecutor` (`v1/executor/uniproc_executor.py`) loads the
  worker *inside* the engine core process. Next pod session: confirm with `ps`.
- The docs' process table is in `D:\vllm\docs\design\arch_overview.md`, section
  "V1 Process Architecture".

```
┌──────────── API server process ────────────┐        ┌────────── engine core process ──────────┐
│ FastAPI/Uvicorn (one asyncio event loop)    │  ZMQ   │ run_busy_loop:                           │
│ template → tokenise → EngineCoreRequest  ───┼───────►│   schedule → execute_model → update      │
│ detokenise → SSE chunks to the client    ◄──┼────────┤   (GPU worker lives here when 1 GPU)     │
└─────────────────────────────────────────────┘ msgpack└──────────────────────────────────────────┘
```

### 6.2 The main V1 ideas (from the V1 blog post), in plain words

1. **The engine core is isolated in its own process.** CPU work (HTTP,
   tokenising, detokenising) happens in parallel with GPU work instead of
   between GPU steps. Reason: the GIL.
2. **One simple scheduler for prompt and output tokens.** There's no separate
   "prefill phase" and "decode phase". Each step, the scheduler just decides
   how many tokens each request gets: `{request_id: num_tokens}`. A prompt is
   simply a request with many tokens still to compute (`num_tokens -
   num_computed_tokens`, Thursday Q3).
3. **Chunked prefill is built in.** A long prompt can be split across steps
   within the token budget, so decodes don't stall (the SARATHI paper,
   Monday).
4. **Prefix caching is on by default** and costs almost nothing even when
   nothing hits (Saturday).
5. **Persistent batch:** input tensors stay on the GPU between steps, and only
   the changes are applied each step, so preparing a batch is cheap.
6. **torch.compile and CUDA graphs:** the model's GPU work is compiled and
   recorded once, then replayed with low launch overhead.

### 6.3 Where each day of Week 5 fits

| Day | What I studied | Where it sits in the picture |
|---|---|---|
| Mon | SARATHI: chunked prefill, token budget | Scheduler inside the engine core |
| Tue | Router, `async def`, `StreamingResponse` | API server (sections 2 and 3 here) |
| Wed | `EngineCoreRequest`, ZMQ, msgpack, `arrival_time` | The arrow between the two processes |
| Thu | `step()`, request states, `num_computed_tokens`, detokeniser | Engine core loop, and the way back |
| Fri | `lifecycle.md` sections 1, 2, 3, 9 | The whole trip |
| Sat | Role chunk timing, shared template block | SSE output (section 3), prefix cache (section 5) |

### 6.4 Interview question: the big one

**Q: What happens when I send a chat request to vLLM? (aim for about 60 seconds)**

A: "The request hits a FastAPI `async` route in the API server process.
pydantic validates the body. The chat template turns the messages into one
string, and the tokeniser turns that into token ids. The template and the
tokeniser run in a thread pool so the event loop stays free. The ids and
sampling settings are packed into an `EngineCoreRequest`, serialised with
msgpack, and sent over ZMQ to the engine core process. That's a separate
process because of the GIL. The engine core runs a loop: the scheduler picks
which requests run this step and how many tokens each gets, within a token
budget, so long prompts are chunked. The model runs one forward pass for the
whole batch, and new token ids go back over ZMQ. In the API server, the
output processor turns token ids into text incrementally and puts each update
in that request's queue. The streaming generator turns each update into an
SSE `data:` chunk, and FastAPI's `StreamingResponse` sends it to the client
straight away. If the client disconnects, the generator is cancelled and the
request is aborted, so its KV cache blocks are freed."

---

## 7. Interview drill

### 7.1 Say these out loud (60 seconds each, no notes)

1. What is the event loop, and why does a blocking call freeze it?
2. `async def` vs `def` in FastAPI: what runs where?
3. Why SSE and not WebSockets for token streaming?
4. Why isn't token count the same as word count?
5. What does a chat template do, and what goes wrong with the wrong one?
6. The full request trip through vLLM (6.4).

### 7.2 Debugging scenarios: what would I check?

| Symptom | First things to check |
|---|---|
| The stream arrives all at once at the end | Not returning a `StreamingResponse`; client not streaming (`stream=True`, `curl -N`); proxy buffering; gzip middleware; missing `\n\n` (2.6) |
| The whole server freezes when one user sends a huge prompt | CPU-heavy or blocking work inside `async def` (tokenising, JSON). Move it to a thread or another process (1.3) |
| The GPU stays busy after users close the tab | No cancellation handling. The request must be aborted on `CancelledError`/`GeneratorExit` (1.6, `async_llm.py:618`) |
| TTFT is high but ITL is fine | Queueing (`num_requests_waiting`), long prompts (prefill), the token budget, prefix-cache misses, tokenisation time |
| The model rambles or talks as the user | Wrong template, missing `add_generation_prompt`, or a base model (5.5) |
| Our token counts don't match the provider's | Different tokeniser; template and system tokens counted; special tokens (4.4, 5.4) |

### 7.3 Build drill (15 minutes, from memory)

Write a FastAPI `POST /v1/chat` that streams fake tokens as SSE, plus a Python
client that prints them as they arrive and measures TTFT. Then check:

- [ ] Route is `async def` and uses `await asyncio.sleep`, not `time.sleep`
- [ ] Returns `StreamingResponse(..., media_type="text/event-stream")`
- [ ] Every chunk is `data: <json>\n\n`, and the stream ends with `data: [DONE]\n\n`
- [ ] The role chunk is sent only after token 1 exists
- [ ] Client uses `stream=True`, skips blank and `:` lines, stops at `[DONE]`
- [ ] TTFT is measured with `time.perf_counter()`

---

## 8. Resources (in the order I'd read them today)

| Topic | Resource | What to focus on |
|---|---|---|
| asyncio | Real Python, "Async IO in Python: A Complete Walkthrough" (realpython.com/async-io-python) | Event loop, `await`, `gather`, async generators |
| asyncio | Python docs, `asyncio` (docs.python.org/3/library/asyncio.html) | `create_task`, `to_thread`, cancellation |
| FastAPI | FastAPI docs, "Concurrency and async / await" (fastapi.tiangolo.com/async) | When to use `async def` vs `def` |
| SSE | MDN, "Using server-sent events" | The message format; `EventSource` |
| Tokenisation | Andrej Karpathy, "Let's build the GPT Tokenizer" (YouTube), **first hour only** | BPE, byte-level tokens, why tokenisation causes weirdness |
| Chat templates | Hugging Face Transformers docs, "Chat templates" | `apply_chat_template`, `add_generation_prompt` |
| vLLM | vLLM blog, "vLLM V1: A Major Upgrade to vLLM's Core Architecture" (blog.vllm.ai, Jan 2025) | The main V1 ideas (6.2) |
| vLLM | vLLM docs, "Architecture Overview"; local copy at `D:\vllm\docs\design\arch_overview.md` | The process table (6.1) |

### A light 2-hour plan

| Time | Do |
|---|---|
| 0:00–0:25 | Real Python asyncio. Run the tea example with `asyncio.sleep`, then with `time.sleep` |
| 0:25–0:40 | FastAPI concurrency page. Run the three routes in 2.2 |
| 0:40–0:50 | MDN SSE page. Run the streaming endpoint in 2.4 and `curl -N` it |
| 0:50–1:30 | Karpathy's first hour at 1.5× speed. Run the Colab snippet in 4.4 |
| 1:30–1:40 | HF chat templates page. Compare it with my Colab output |
| 1:40–1:50 | vLLM V1 blog. Match each idea to my Week 5 notes |
| 1:50–2:00 | Drill 7.1 out loud. Mark the questions I stumbled on |

---

## 9. Self-test (cover everything above)

1. What is the only place the event loop can switch to another task?
2. `time.sleep(2)` inside an `async def` route: what happens to other requests?
3. Name two ways vLLM keeps slow work off the event loop.
4. What does `StreamingResponse` do with each value the generator yields?
5. What ends one SSE message?
6. Why do LLM clients parse SSE by hand instead of using the browser's `EventSource`?
7. Walk through BPE on `aaabdaaabac`.
8. Why can't byte-level BPE ever hit an "unknown word"?
9. Where does `<|im_start|>assistant\n` at the end of the prompt come from, and why does it matter?
10. My Week 3 data shows 541 tokens per request for a 512-token prompt. Where do the other 29 come from?
11. How many processes does vLLM run on my 1-GPU setup, and why not 3?
12. Name three of the V1 ideas and say which Week 5 day each one connects to.
