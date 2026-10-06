from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "runner/openhands_call_budget"))

from logical_call_budget import (  # noqa: E402
    LogicalLLMBudgetExceeded,
    logical_call_budget,
    reserve,
    record_response,
)


class LogicalCallBudgetTests(unittest.TestCase):
    def test_progress_is_preserved_on_interrupt(self):
        receipts = []
        with self.assertRaises(KeyboardInterrupt):
            with logical_call_budget(3, receipt_callback=receipts.append):
                reserve("solver")
                record_response({"usage": {"prompt_tokens": 10}})
                raise KeyboardInterrupt
        self.assertEqual(receipts[-1]["total"], 1)
        self.assertEqual(receipts[-1]["provider_usage"]["prompt_tokens"], 10)


    def test_limit_and_kind_accounting(self) -> None:
        with logical_call_budget(3) as state:
            reserve("solver")
            reserve("condenser")
            reserve("solver")
            with self.assertRaises(LogicalLLMBudgetExceeded):
                reserve("solver")
        self.assertEqual(state.receipt()["total"], 3)
        self.assertEqual(state.receipt()["by_kind"], {"condenser": 1, "solver": 2})
        self.assertTrue(state.receipt()["exhausted"])

    def test_contexts_are_isolated_between_worker_threads(self) -> None:
        def worker() -> int:
            with logical_call_budget(2) as state:
                reserve("solver")
                reserve("solver")
            return state.total

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(lambda _: worker(), range(2))), [2, 2])

    def test_provider_usage_is_preserved_in_budget_receipt(self) -> None:
        response = {
            "model": "deepseek-v4-flash-260425",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "choices": [{"message": {"reasoning_content": None}}],
        }
        with logical_call_budget(2) as state:
            reserve("solver")
            record_response(response)
        usage = state.receipt()["provider_usage"]
        self.assertEqual(usage["successful_responses"], 1)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 20)
        self.assertEqual(usage["cache_read_tokens"], 40)
        self.assertEqual(usage["effective_models"], ["deepseek-v4-flash-260425"])



if __name__ == "__main__":
    unittest.main()
