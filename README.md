# llm-call-batcher

Accumulate individual LLM requests into a batch, then flush on size or timeout.

## Install

```
pip install llm-call-batcher
```

## Usage

```python
from llm_call_batcher import CallBatcher, BatchRequest, BatchResult

def my_batch_fn(requests):
    return [BatchResult(r.request_id, result=f"processed:{r.payload}") for r in requests]

batcher = CallBatcher(batch_fn=my_batch_fn, max_size=10, flush_interval=0.5)

# Blocking submit — waits until batch flushes
result = batcher.submit("req-1", payload={"messages": [...]})
print(result.result)

# Non-blocking enqueue
batcher.submit_nowait("req-2", payload={"messages": [...]})

# Manual flush
batcher.flush()
batcher.shutdown()
```
