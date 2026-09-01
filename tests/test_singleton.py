from __future__ import annotations

import os
import unittest
import uuid

from anime_trivia_automation.singleton import (
    WorkerAlreadyRunningError,
    WorkerMutex,
)


@unittest.skipUnless(os.name == "nt", "Windows named mutex test")
class WorkerMutexTests(unittest.TestCase):
    def test_second_worker_is_rejected_and_release_allows_next(self) -> None:
        name = f"Local\\AnimeTriviaTest.{uuid.uuid4().hex}"
        first = WorkerMutex(name)
        try:
            with self.assertRaises(WorkerAlreadyRunningError):
                WorkerMutex(name)
        finally:
            first.close()
        third = WorkerMutex(name)
        third.close()


if __name__ == "__main__":
    unittest.main()
