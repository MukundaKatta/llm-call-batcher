import threading
import time
from llm_call_batcher import CallBatcher, BatchRequest, BatchResult


def simple_batch_fn(requests):
    return [BatchResult(r.request_id, result=r.payload * 2) for r in requests]


def make_batcher(**kwargs):
    return CallBatcher(
        batch_fn=simple_batch_fn, max_size=5, flush_interval=0.1, **kwargs
    )


def test_basic_submit():
    batcher = make_batcher()
    try:
        result = batcher.submit("r1", payload=21)
        assert result.ok
        assert result.result == 42
    finally:
        batcher.shutdown()


def test_result_id_matches():
    batcher = make_batcher()
    try:
        result = batcher.submit("my-id", payload=1)
        assert result.request_id == "my-id"
    finally:
        batcher.shutdown()


def test_flush_on_max_size():
    flushed = []

    def tracking_fn(requests):
        flushed.append(len(requests))
        return [BatchResult(r.request_id, result=r.payload) for r in requests]

    batcher = CallBatcher(batch_fn=tracking_fn, max_size=3, flush_interval=10.0)
    try:
        results = []
        for i in range(3):
            t = threading.Thread(
                target=lambda i=i: results.append(batcher.submit(f"r{i}", payload=i))
            )
            t.start()
        time.sleep(0.3)
        assert 3 in flushed or len(results) == 3
    finally:
        batcher.shutdown()


def test_flush_on_timeout():
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=100, flush_interval=0.05)
    try:
        result = batcher.submit("t1", payload=5)
        assert result.result == 10
    finally:
        batcher.shutdown()


def test_error_in_batch_fn():
    def failing_fn(requests):
        raise RuntimeError("batch failed")

    batcher = CallBatcher(batch_fn=failing_fn, max_size=1, flush_interval=10.0)
    try:
        result = batcher.submit("e1", payload=1)
        assert not result.ok
        assert isinstance(result.error, RuntimeError)
    finally:
        batcher.shutdown()


def test_submit_nowait_enqueues():
    batcher = make_batcher()
    try:
        batcher.submit_nowait("n1", payload=1)
        assert batcher.pending_count <= 1
    finally:
        batcher.shutdown()


def test_manual_flush():
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=100, flush_interval=10.0)
    try:
        batcher.submit_nowait("m1", payload=1)
        batcher.flush()
        # After flush, pending should be 0
        assert batcher.pending_count == 0
    finally:
        batcher.shutdown()


def test_pending_count():
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=100, flush_interval=100.0)
    try:
        batcher.submit_nowait("p1", payload=1)
        assert batcher.pending_count == 1
    finally:
        batcher.shutdown()


def test_properties():
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=7, flush_interval=2.5)
    try:
        assert batcher.max_size == 7
        assert batcher.flush_interval == 2.5
    finally:
        batcher.shutdown()


def test_batch_request_dataclass():
    req = BatchRequest(request_id="x", payload=99)
    assert req.request_id == "x"
    assert req.payload == 99
    assert req.submitted_at > 0


def test_batch_result_ok():
    r = BatchResult("id", result=42)
    assert r.ok
    assert r.result == 42


def test_batch_result_error():
    r = BatchResult("id", error=ValueError("oops"))
    assert not r.ok


def test_shutdown_flushes():
    results_holder = []

    def collecting_fn(requests):
        results_holder.extend(requests)
        return [BatchResult(r.request_id, result=r.payload) for r in requests]

    batcher = CallBatcher(batch_fn=collecting_fn, max_size=100, flush_interval=10.0)
    batcher.submit_nowait("s1", payload=1)
    batcher.shutdown()
    assert any(r.request_id == "s1" for r in results_holder)


def test_concurrent_submits():
    results = {}
    lock = threading.Lock()

    def collecting_fn(requests):
        return [BatchResult(r.request_id, result=r.payload + 100) for r in requests]

    batcher = CallBatcher(batch_fn=collecting_fn, max_size=5, flush_interval=0.05)
    try:
        threads = []
        for i in range(5):

            def submit(i=i):
                r = batcher.submit(f"c{i}", payload=i)
                with lock:
                    results[r.request_id] = r.result

            t = threading.Thread(target=submit)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        assert len(results) == 5
    finally:
        batcher.shutdown()


def test_submit_nowait_does_not_leak_results():
    # Fire-and-forget requests have no waiter to pop their result, so the
    # batcher must not retain them after flushing.
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=1000, flush_interval=100.0)
    try:
        for i in range(50):
            batcher.submit_nowait(f"n{i}", payload=i)
        batcher.flush()
        assert batcher.pending_count == 0
        assert len(batcher._results) == 0
    finally:
        batcher.shutdown()


def test_submit_timeout_does_not_leak_results():
    # A submit() that times out must return a TimeoutError and must not leave a
    # stale result/future behind even when the batch flushes afterwards.
    batcher = CallBatcher(batch_fn=simple_batch_fn, max_size=1000, flush_interval=5.0)
    try:
        result = batcher.submit("late", payload=1, timeout=0.05)
        assert not result.ok
        assert isinstance(result.error, TimeoutError)
        batcher.flush()  # would have stored a stale result before the fix
        time.sleep(0.05)
        assert len(batcher._results) == 0
        assert len(batcher._futures) == 0
    finally:
        batcher.shutdown()
