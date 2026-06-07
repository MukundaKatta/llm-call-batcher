# llm-call-batcher

[![CI](https://github.com/MukundaKatta/llm-call-batcher/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-call-batcher/actions/workflows/ci.yml)

Accumulate individual requests into a batch and dispatch them together,
flushing whenever the batch fills up **or** a timeout elapses — whichever
comes first.

Many LLM providers (and plenty of other APIs) are far more efficient when you
send several inputs in one call: you amortise the network round-trip, stay
under per-request rate limits, and reuse a shared prompt prefix. But your
application code usually produces requests one at a time. `llm-call-batcher`
sits in between: callers submit single requests and get single results back,
while under the hood the requests are coalesced into batches and handed to a
function you provide.

- **Zero dependencies** — pure standard library, thread-safe.
- **Size *and* time triggers** — flush on `max_size` or after `flush_interval`.
- **Blocking and fire-and-forget APIs** — `submit` waits for a result;
  `submit_nowait` enqueues and optionally calls you back.
- **Robust error handling** — exceptions in your batch function are surfaced
  per request; missing results are reported instead of silently dropped.

## Install

```
pip install llm-call-batcher
```

Or from source:

```
git clone https://github.com/MukundaKatta/llm-call-batcher
cd llm-call-batcher
pip install -e .
```

## Usage

```python
from llm_call_batcher import CallBatcher, BatchResult

# Your batch function receives a list[BatchRequest] and must return a
# list[BatchResult] (one per request). This is where the real LLM call goes —
# here we just double the payload to keep the example self-contained.
def my_batch_fn(requests):
    return [BatchResult(r.request_id, result=r.payload * 2) for r in requests]

batcher = CallBatcher(batch_fn=my_batch_fn, max_size=10, flush_interval=0.5)
try:
    # Blocking submit — waits until the batch flushes, then returns the result.
    result = batcher.submit("req-1", payload=21)
    print(result.ok, result.result)        # True 42

    # Fire-and-forget with a callback invoked when the batch flushes.
    batcher.submit_nowait("req-2", payload=5, callback=lambda r: print(r.result))

    # Force any pending requests to flush right now.
    batcher.flush()
finally:
    batcher.shutdown()                      # stops the timer and flushes
```

`CallBatcher` is also a context manager, so the `try/finally` above can be
written as:

```python
with CallBatcher(batch_fn=my_batch_fn, max_size=10, flush_interval=0.5) as batcher:
    print(batcher.submit("req-1", payload=21).result)
```

### Wiring it to a real LLM

`batch_fn` is the only integration point. A sketch against a chat API:

```python
def my_batch_fn(requests):
    results = []
    for r in requests:
        try:
            response = client.chat(messages=r.payload["messages"])
            results.append(BatchResult(r.request_id, result=response))
        except Exception as exc:
            results.append(BatchResult(r.request_id, error=exc))
    return results

batcher = CallBatcher(batch_fn=my_batch_fn, max_size=20, flush_interval=0.2)
result = batcher.submit("chat-1", payload={"messages": [...]})
```

If a provider exposes a true batch endpoint, send the whole `requests` list in
one call and split the response back into one `BatchResult` per `request_id`.

## API

### `CallBatcher(batch_fn, max_size=10, flush_interval=1.0)`

Coordinates batching. Thread-safe.

- `batch_fn: Callable[[list[BatchRequest]], list[BatchResult]]` — called with a
  non-empty list of requests; must return a result per request. If it raises,
  every request in that batch receives the exception. Any `request_id` it omits
  is reported as a `RuntimeError`.
- `max_size: int` — flush as soon as this many requests are pending (`>= 1`).
- `flush_interval: float` — seconds between periodic timer-driven flushes
  (`> 0`).

Methods and properties:

- `submit(request_id, payload, timeout=None) -> BatchResult` — enqueue and block
  until the batch flushes. On timeout, returns a `BatchResult` carrying a
  `TimeoutError`.
- `submit_nowait(request_id, payload, callback=None) -> None` — enqueue without
  blocking; if `callback` is given it is invoked with the `BatchResult` once the
  batch flushes.
- `flush() -> None` — flush all currently pending requests immediately.
- `shutdown() -> None` — stop the background timer and flush any remainder;
  idempotent. No further submissions are accepted afterwards.
- `pending_count: int` — number of queued, not-yet-flushed requests.
- `max_size: int`, `flush_interval: float` — the configured values.

### `BatchRequest(request_id, payload, submitted_at=...)`

One queued request. `submitted_at` defaults to a monotonic timestamp.

### `BatchResult(request_id, result=None, error=None)`

The outcome for one request. `result.ok` is `True` when `error is None`.

## Development

Run the test suite (standard-library `unittest`, no third-party deps):

```
python -m unittest discover -s tests
```

## License

MIT
