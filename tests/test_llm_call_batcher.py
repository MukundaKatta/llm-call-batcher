"""Standard-library unittest suite for llm_call_batcher.

Run with::

    python3 -m unittest discover -s tests
"""
import os
import sys
import threading
import time
import unittest

# Make the ``src`` layout package importable without installing the project,
# so the suite runs via ``python3 -m unittest discover -s tests`` as-is.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_call_batcher import BatchRequest, BatchResult, CallBatcher, __version__


def double_batch_fn(requests):
    """A trivial batch_fn that doubles each integer payload."""
    return [BatchResult(r.request_id, result=r.payload * 2) for r in requests]


def make_batcher(**kwargs):
    params = dict(batch_fn=double_batch_fn, max_size=5, flush_interval=0.1)
    params.update(kwargs)
    return CallBatcher(**params)


class BatchResultTests(unittest.TestCase):
    def test_ok_when_no_error(self):
        r = BatchResult("id", result=42)
        self.assertTrue(r.ok)
        self.assertEqual(r.result, 42)

    def test_not_ok_when_error(self):
        r = BatchResult("id", error=ValueError("oops"))
        self.assertFalse(r.ok)
        self.assertIsInstance(r.error, ValueError)


class BatchRequestTests(unittest.TestCase):
    def test_fields(self):
        req = BatchRequest(request_id="x", payload=99)
        self.assertEqual(req.request_id, "x")
        self.assertEqual(req.payload, 99)
        self.assertGreater(req.submitted_at, 0)


class ConstructorValidationTests(unittest.TestCase):
    def test_max_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            CallBatcher(batch_fn=double_batch_fn, max_size=0)

    def test_flush_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            CallBatcher(batch_fn=double_batch_fn, flush_interval=0)


class SubmitTests(unittest.TestCase):
    def test_basic_submit_returns_result(self):
        batcher = make_batcher()
        try:
            result = batcher.submit("r1", payload=21)
            self.assertTrue(result.ok)
            self.assertEqual(result.result, 42)
        finally:
            batcher.shutdown()

    def test_result_id_matches_request(self):
        batcher = make_batcher()
        try:
            result = batcher.submit("my-id", payload=1)
            self.assertEqual(result.request_id, "my-id")
        finally:
            batcher.shutdown()

    def test_flush_on_timeout(self):
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=100, flush_interval=0.05
        )
        try:
            result = batcher.submit("t1", payload=5)
            self.assertEqual(result.result, 10)
        finally:
            batcher.shutdown()

    def test_submit_timeout_returns_timeout_error(self):
        # max_size large and flush_interval long: the result will not arrive
        # within the per-call timeout, so a TimeoutError result is returned.
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=100, flush_interval=100.0
        )
        try:
            result = batcher.submit("late", payload=1, timeout=0.05)
            self.assertFalse(result.ok)
            self.assertIsInstance(result.error, TimeoutError)
        finally:
            batcher.shutdown()

    def test_submit_after_shutdown_raises(self):
        batcher = make_batcher()
        batcher.shutdown()
        with self.assertRaises(RuntimeError):
            batcher.submit("x", payload=1)


class FlushTriggerTests(unittest.TestCase):
    def test_flush_on_max_size(self):
        flushed_sizes = []

        def tracking_fn(requests):
            flushed_sizes.append(len(requests))
            return [BatchResult(r.request_id, result=r.payload) for r in requests]

        batcher = CallBatcher(
            batch_fn=tracking_fn, max_size=3, flush_interval=10.0
        )
        try:
            results = []

            def worker(i):
                results.append(batcher.submit(f"r{i}", payload=i))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=2.0)
            self.assertEqual(len(results), 3)
            self.assertIn(3, flushed_sizes)
        finally:
            batcher.shutdown()

    def test_manual_flush_clears_pending(self):
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=100, flush_interval=10.0
        )
        try:
            batcher.submit_nowait("m1", payload=1)
            self.assertEqual(batcher.pending_count, 1)
            batcher.flush()
            self.assertEqual(batcher.pending_count, 0)
        finally:
            batcher.shutdown()


class ErrorHandlingTests(unittest.TestCase):
    def test_exception_in_batch_fn_propagates_to_result(self):
        def failing_fn(requests):
            raise RuntimeError("batch failed")

        batcher = CallBatcher(
            batch_fn=failing_fn, max_size=1, flush_interval=10.0
        )
        try:
            result = batcher.submit("e1", payload=1)
            self.assertFalse(result.ok)
            self.assertIsInstance(result.error, RuntimeError)
        finally:
            batcher.shutdown()

    def test_missing_result_reported_as_error(self):
        def partial_fn(requests):
            # Return nothing -> every request should be marked as an error.
            return []

        batcher = CallBatcher(
            batch_fn=partial_fn, max_size=1, flush_interval=10.0
        )
        try:
            result = batcher.submit("p1", payload=1)
            self.assertFalse(result.ok)
            self.assertIsInstance(result.error, RuntimeError)
        finally:
            batcher.shutdown()


class SubmitNowaitTests(unittest.TestCase):
    def test_enqueues_request(self):
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=100, flush_interval=100.0
        )
        try:
            batcher.submit_nowait("n1", payload=1)
            self.assertEqual(batcher.pending_count, 1)
        finally:
            batcher.shutdown()

    def test_callback_invoked_with_result(self):
        received = []
        done = threading.Event()

        def cb(result):
            received.append(result)
            done.set()

        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=100, flush_interval=10.0
        )
        try:
            batcher.submit_nowait("cb1", payload=4, callback=cb)
            batcher.flush()
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].result, 8)
            self.assertEqual(received[0].request_id, "cb1")
        finally:
            batcher.shutdown()

    def test_no_result_leak_for_fire_and_forget(self):
        # Regression test: results for requests with no waiter and no callback
        # must not accumulate in the internal results map.
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=2, flush_interval=100.0
        )
        try:
            batcher.submit_nowait("a", payload=1)
            batcher.submit_nowait("b", payload=2)  # triggers a size flush
            # Give the background flush thread time to run.
            deadline = time.monotonic() + 2.0
            while batcher.pending_count != 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(batcher.pending_count, 0)
            # The private results map should be empty: nothing was waiting.
            self.assertEqual(len(batcher._results), 0)
        finally:
            batcher.shutdown()


class LifecycleTests(unittest.TestCase):
    def test_properties_reflect_construction(self):
        batcher = CallBatcher(
            batch_fn=double_batch_fn, max_size=7, flush_interval=2.5
        )
        try:
            self.assertEqual(batcher.max_size, 7)
            self.assertEqual(batcher.flush_interval, 2.5)
        finally:
            batcher.shutdown()

    def test_shutdown_flushes_remaining(self):
        seen = []

        def collecting_fn(requests):
            seen.extend(requests)
            return [BatchResult(r.request_id, result=r.payload) for r in requests]

        batcher = CallBatcher(
            batch_fn=collecting_fn, max_size=100, flush_interval=10.0
        )
        batcher.submit_nowait("s1", payload=1)
        batcher.shutdown()
        self.assertTrue(any(r.request_id == "s1" for r in seen))

    def test_shutdown_is_idempotent(self):
        batcher = make_batcher()
        batcher.shutdown()
        # A second shutdown must not raise.
        batcher.shutdown()

    def test_context_manager_shuts_down(self):
        seen = []

        def collecting_fn(requests):
            seen.extend(requests)
            return [BatchResult(r.request_id, result=r.payload) for r in requests]

        with CallBatcher(
            batch_fn=collecting_fn, max_size=100, flush_interval=10.0
        ) as batcher:
            batcher.submit_nowait("ctx", payload=1)
        # Exiting the context flushes the pending request.
        self.assertTrue(any(r.request_id == "ctx" for r in seen))


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_submits_all_resolve(self):
        results = {}
        lock = threading.Lock()

        def add_100(requests):
            return [
                BatchResult(r.request_id, result=r.payload + 100)
                for r in requests
            ]

        batcher = CallBatcher(
            batch_fn=add_100, max_size=5, flush_interval=0.05
        )
        try:
            def worker(i):
                r = batcher.submit(f"c{i}", payload=i)
                with lock:
                    results[r.request_id] = r.result

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=2.0)
            self.assertEqual(len(results), 5)
            self.assertEqual(results["c0"], 100)
            self.assertEqual(results["c4"], 104)
        finally:
            batcher.shutdown()


class MetadataTests(unittest.TestCase):
    def test_version_is_string(self):
        self.assertIsInstance(__version__, str)


if __name__ == "__main__":
    unittest.main()
