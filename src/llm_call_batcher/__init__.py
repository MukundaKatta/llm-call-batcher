"""
llm-call-batcher: Accumulate individual LLM requests into a batch, then flush on size or timeout.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BatchRequest:
    request_id: str
    payload: Any
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class BatchResult:
    request_id: str
    result: Any = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class CallBatcher:
    """
    Collect individual requests and flush them as a batch when either
    the batch is full or a timeout elapses.

    Usage::

        def my_batch_fn(requests):
            # process a list[BatchRequest] -> list[BatchResult]
            return [BatchResult(r.request_id, result=r.payload*2) for r in requests]

        batcher = CallBatcher(batch_fn=my_batch_fn, max_size=10, flush_interval=0.5)
        result = batcher.submit("req-1", payload=42)   # blocks until batch flushes
        result.result  # 84
        batcher.shutdown()
    """

    def __init__(
        self,
        batch_fn: Callable[[list[BatchRequest]], list[BatchResult]],
        max_size: int = 10,
        flush_interval: float = 1.0,
    ) -> None:
        self._batch_fn = batch_fn
        self._max_size = max_size
        self._flush_interval = flush_interval

        self._lock = threading.Lock()
        self._pending: list[BatchRequest] = []
        self._futures: dict[str, threading.Event] = {}
        self._results: dict[str, BatchResult] = {}
        self._counter = 0
        self._shutdown = False

        self._timer: Optional[threading.Timer] = None
        self._start_timer()

    # -- internal ----------------------------------------------------------

    def _start_timer(self) -> None:
        if self._shutdown:
            return
        self._timer = threading.Timer(self._flush_interval, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        self._flush()
        if not self._shutdown:
            self._start_timer()

    def _flush(self) -> None:
        with self._lock:
            if not self._pending:
                return
            batch = self._pending[:]
            self._pending.clear()

        try:
            results = self._batch_fn(batch)
        except Exception as exc:
            results = [BatchResult(r.request_id, error=exc) for r in batch]

        # index results by request_id
        result_map = {r.request_id: r for r in results}
        # fill in missing (if batch_fn didn't return all)
        for req in batch:
            if req.request_id not in result_map:
                result_map[req.request_id] = BatchResult(
                    req.request_id, error=RuntimeError("No result returned by batch_fn")
                )

        with self._lock:
            for req_id, result in result_map.items():
                # Only retain results that a blocking submit() is waiting for.
                # submit_nowait() and timed-out submit() calls register no live
                # future, so storing their results would leak memory unbounded.
                evt = self._futures.get(req_id)
                if evt is not None:
                    self._results[req_id] = result
                    evt.set()

    # -- public API -------------------------------------------------------

    def submit(
        self, request_id: str, payload: Any, timeout: Optional[float] = None
    ) -> BatchResult:
        """Add a request and block until the batch is flushed."""
        evt = threading.Event()
        req = BatchRequest(request_id=request_id, payload=payload)

        with self._lock:
            self._pending.append(req)
            self._futures[request_id] = evt
            should_flush = len(self._pending) >= self._max_size

        if should_flush:
            self._flush()
        else:
            evt.wait(timeout=timeout)

        with self._lock:
            result = self._results.pop(request_id, None)
            self._futures.pop(request_id, None)

        if result is None:
            return BatchResult(
                request_id, error=TimeoutError("Batch did not flush in time")
            )
        return result

    def submit_nowait(self, request_id: str, payload: Any) -> None:
        """Enqueue a request without waiting for the result."""
        req = BatchRequest(request_id=request_id, payload=payload)
        with self._lock:
            self._pending.append(req)
            should_flush = len(self._pending) >= self._max_size
        if should_flush:
            threading.Thread(target=self._flush, daemon=True).start()

    def flush(self) -> None:
        """Manually flush all pending requests."""
        self._flush()

    def shutdown(self) -> None:
        """Stop the background timer and flush remaining requests."""
        self._shutdown = True
        if self._timer:
            self._timer.cancel()
        self._flush()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def flush_interval(self) -> float:
        return self._flush_interval


__all__ = ["CallBatcher", "BatchRequest", "BatchResult"]
