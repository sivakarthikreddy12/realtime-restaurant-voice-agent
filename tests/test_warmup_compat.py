"""Tests for robust consumption of the speech-to-speech warmup streamer."""

from __future__ import annotations

import queue
import threading
import time
import unittest

from warmup_compat import drain_streamer_while_worker_runs


class QueueStreamer:
    def __init__(self) -> None:
        self.items = queue.Queue()
        self.stop = object()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.items.get(timeout=0.01)
        if value is self.stop:
            raise StopIteration
        return value


class WarmupCompatTests(unittest.TestCase):
    def test_empty_polls_are_allowed_while_worker_is_alive(self) -> None:
        streamer = QueueStreamer()

        def delayed_completion() -> None:
            time.sleep(0.06)
            streamer.items.put(streamer.stop)

        worker = threading.Thread(target=delayed_completion)
        worker.start()
        drain_streamer_while_worker_runs(streamer, worker, [], timeout_seconds=1)
        self.assertFalse(worker.is_alive())

    def test_generation_thread_error_is_preserved(self) -> None:
        streamer = QueueStreamer()
        worker_error = ValueError("unsupported CPU operation")
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join()
        with self.assertRaisesRegex(RuntimeError, "ValueError: unsupported CPU operation") as raised:
            drain_streamer_while_worker_runs(streamer, worker, [worker_error], timeout_seconds=1)
        self.assertIs(raised.exception.__cause__, worker_error)


if __name__ == "__main__":
    unittest.main()
