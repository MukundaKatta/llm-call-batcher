"""llm-call-batcher: coalesce individual requests into batches.

Accumulate individual LLM (or any) requests and dispatch them to a
user-supplied ``batch_fn`` as a single list, flushing whenever the batch
reaches ``max_size`` or ``flush_interval`` seconds elapse -- whichever comes
first. This amortises per-call overhead (network round-trips, provider rate
limits, fixed prompt prefixes) across many logical requests.

The public surface is intentionally small:

* :class:`BatchRequest`  -- one queued request (id + payload).
* :class:`BatchResult`   -- the outcome for one request (result or error).
* :class:`CallBatcher`   -- the coordinator you interact with.

Example::

    from llm_call_batcher import CallBatcher, BatchResult

    def my_batch_fn(requests):
        # requests: list[BatchRequest]  ->  list[BatchResult]
        return [BatchResult(r.request_id, result=r.payload * 2) for r in requests]

    batcher = CallBatcher(batch_fn=my_batch_fn, max_size=10, flush_interval=0.5)
    try:
        result = batcher.submit("req-1", payload=21)  # blocks until flushed
        assert result.result == 42
    finally:
        batcher.shutdown()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__version__ = "0.1.0"

# A function that takes a batch of requests and returns a result per request.
BatchFn = Callable[[List["BatchRequest"]], List["BatchResult"]]
# Optional callback invoked with the result of a fire-and-forget request.
ResultCallback = Callable[["BatchResult"], None]


@dataclass
class BatchRequest:
    """A single queued request.

    Attributes:
        request_id: Caller-supplied identifier used to correlate the request
            with its :class:`BatchResult`. Should be unique among in-flight
            requests.
        payload: Arbitrary user data handed to ``batch_fn`` unchanged.
        submitted_at: Monotonic timestamp (seconds) recorded at construction.
    """

    request_id: str
    payload: Any
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class BatchResult:
    """The outcome of processing a single request.

    Exactly one of ``result`` / ``error`` is meaningful: when ``error`` is
    ``None`` the call succeeded and ``result`` holds the value, otherwise
    ``error`` holds the raised exception.
    """

    request_id: str
    result: Any = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        """``True`` when the request completed without an error."""
        return self.error is None


class CallBatcher:
    """Collect individual requests and flush them as a batch.

    A batch is flushed when either it reaches ``max_size`` pending requests or
    ``flush_interval`` seconds elapse since the periodic timer last fired,
    whichever happens first. ``batch_fn`` is always called with a non-empty
    ``list[BatchRequest]`` and is expected to return a ``list[BatchResult]``;
    any request id missing from the returned list is reported as an error, and
    if ``batch_fn`` raises, every request in that batch receives the raised
    exception. The class is thread-safe.

    Args:
        batch_fn: Callable mapping a list of requests to a list of results.
        max_size: Flush as soon as this many requests are pending. Must be >= 1.
        flush_interval: Seconds between periodic timer-driven flushes. Must be
            > 0.

    Usage::

        def my_batch_fn(requests):
            return [BatchResult(r.request_id, result=r.payload * 2)
                    for r in requests]

        batcher = CallBatcher(batch_fn=my_batch_fn, max_size=10,
                              flush_interval=0.5)
        result = batcher.submit("req-1", payload=21)  # blocks until flushed
        result.result  # 42
        batcher.shutdown()
    """

    def __init__(
        self,
        batch_fn: BatchFn,
        max_size: int = 10,
        flush_interval: float = 1.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")

        self._batch_fn = batch_fn
        self._max_size = max_size
        self._flush_interval = flush_interval

        self._lock = threading.Lock()
        self._pending: List[BatchRequest] = []
        # request_id -> Event signalled when the result is ready.
        self._futures: Dict[str, threading.Event] = {}
        # request_id -> result, only retained while a waiter exists.
        self._results: Dict[str, BatchResult] = {}
        # request_id -> callback for fire-and-forget requests.
        self._callbacks: Dict[str, ResultCallback] = {}
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
        except Exception as exc:  # noqa: BLE001 - surface to every request
            results = [BatchResult(r.request_id, error=exc) for r in batch]

        # Index results by request_id.
        result_map = {r.request_id: r for r in results}
        # Fill in any request the batch_fn did not return a result for.
        for req in batch:
            if req.request_id not in result_map:
                result_map[req.request_id] = BatchResult(
                    req.request_id,
                    error=RuntimeError("No result returned by batch_fn"),
                )

        callbacks: List[tuple] = []
        with self._lock:
            for req_id, result in result_map.items():
                evt = self._futures.get(req_id)
                if evt is not None:
                    # A waiter exists: stash the result for it to pick up.
                    self._results[req_id] = result
                    evt.set()
                cb = self._callbacks.pop(req_id, None)
                if cb is not None:
                    callbacks.append((cb, result))
                # If there is neither a waiter nor a callback the result is
                # intentionally dropped so ``_results`` cannot grow unbounded.

        # Run callbacks outside the lock so user code cannot deadlock us.
        for cb, result in callbacks:
            try:
                cb(result)
            except Exception:  # noqa: BLE001 - never let a callback break flush
                pass

    # -- public API -------------------------------------------------------

    def submit(
        self,
        request_id: str,
        payload: Any,
        timeout: Optional[float] = None,
    ) -> BatchResult:
        """Add a request and block until its batch is flushed.

        Args:
            request_id: Unique identifier for this request.
            payload: Data passed through to ``batch_fn``.
            timeout: Maximum seconds to wait for the result. ``None`` waits
                indefinitely. On timeout a :class:`BatchResult` carrying a
                :class:`TimeoutError` is returned.

        Returns:
            The :class:`BatchResult` for ``request_id``.
        """
        evt = threading.Event()
        req = BatchRequest(request_id=request_id, payload=payload)

        with self._lock:
            if self._shutdown:
                raise RuntimeError("CallBatcher is shut down")
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

    def submit_nowait(
        self,
        request_id: str,
        payload: Any,
        callback: Optional[ResultCallback] = None,
    ) -> None:
        """Enqueue a request without blocking for its result.

        Args:
            request_id: Unique identifier for this request.
            payload: Data passed through to ``batch_fn``.
            callback: Optional one-argument callable invoked with the
                :class:`BatchResult` once the batch is flushed. Exceptions
                raised by the callback are swallowed.
        """
        req = BatchRequest(request_id=request_id, payload=payload)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("CallBatcher is shut down")
            self._pending.append(req)
            if callback is not None:
                self._callbacks[request_id] = callback
            should_flush = len(self._pending) >= self._max_size
        if should_flush:
            threading.Thread(target=self._flush, daemon=True).start()

    def flush(self) -> None:
        """Flush all currently pending requests immediately."""
        self._flush()

    def shutdown(self) -> None:
        """Stop the background timer and flush any remaining requests.

        Idempotent and safe to call from a ``finally`` block. After shutdown
        no further requests may be submitted.
        """
        with self._lock:
            self._shutdown = True
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
        self._flush()

    def __enter__(self) -> "CallBatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    @property
    def pending_count(self) -> int:
        """Number of requests currently queued and not yet flushed."""
        with self._lock:
            return len(self._pending)

    @property
    def max_size(self) -> int:
        """Maximum batch size before a size-triggered flush."""
        return self._max_size

    @property
    def flush_interval(self) -> float:
        """Seconds between periodic timer-driven flushes."""
        return self._flush_interval


__all__ = ["CallBatcher", "BatchRequest", "BatchResult", "__version__"]
