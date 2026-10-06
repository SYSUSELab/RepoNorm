from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))

from task_telemetry import (  # noqa: E402
    LogicalLLMBudgetExceeded,
    ModelIdentityViolation,
    install_hooks,
    task_telemetry,
)


MODEL = "deepseek-v4-flash-260425"
WIRE = {"model": "deepseek/" + MODEL, "api_base": "https://provider.invalid/v1",
        "temperature": 0, "extra_body": {"thinking": {"type": "disabled"}}, "drop_params": False}


def response(**overrides):
    return {
        "model": MODEL,
        "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                  "prompt_tokens_details": {"cached_tokens": 0},
                  "completion_tokens_details": {"reasoning_tokens": 0}},
        "choices": [{"message": {"content": "private content must not persist"}}],
    } | overrides


def fixture():
    module = SimpleNamespace(completion=lambda **kwargs: kwargs.get("response", response()))

    class Model:
        abort_exceptions = []

        def query(self, *, fail_after=False, retry_once=False, **kwargs):
            if retry_once:
                try:
                    module.completion(fail=True)
                except ValueError:
                    pass
            result = module.completion(**kwargs)
            if fail_after:
                raise ValueError("sensitive parse error")
            return result

        def summarize_context(self, **kwargs):
            return module.completion(**kwargs)

    return Model, module


def prepare_fixture():
    module = SimpleNamespace(completion=lambda **kwargs: response())

    class Model:
        abort_exceptions = []

        def _prepare_messages_for_api(self, messages):
            return [dict(message) for message in messages]

        def query(self, messages, **kwargs):
            return self._prepare_messages_for_api(messages)

        def summarize_context(self, messages, **kwargs):
            return self._prepare_messages_for_api(messages)

    return Model, module


class TaskTelemetryTests(unittest.TestCase):
    def test_agicto_empty_tool_reasoning_replay_compat_is_scoped_and_counted(self):
        model, module = prepare_fixture()
        install_hooks(model, module)
        messages = [
            {"role": "assistant", "content": "summary without provider reasoning"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}],
             "reasoning_content": ""},
        ]
        with task_telemetry(reasoning_content_replay_compat="agicto_space_for_empty_tool_reasoning") as state:
            prepared = model().query(messages)
        self.assertEqual(prepared[0]["reasoning_content"], " ")
        self.assertEqual(prepared[1]["reasoning_content"], " ")
        self.assertEqual(state.receipt()["reasoning_content_replay_injections"], 2)
        with task_telemetry() as state:
            untouched = model().query(messages)
        self.assertNotIn("reasoning_content", untouched[0])
        self.assertEqual(untouched[1]["reasoning_content"], "")
        self.assertEqual(state.receipt()["reasoning_content_replay_injections"], 0)

    def test_solver_and_compaction_budget_and_receipt(self):
        model, module = fixture()
        install_hooks(model, module)
        install_hooks(model, module)
        with task_telemetry(limit=2) as state:
            model().query()
            model().summarize_context()
            with self.assertRaises(LogicalLLMBudgetExceeded):
                model().query()
        receipt = state.receipt()
        self.assertEqual(receipt["total"], 2)
        self.assertEqual(receipt["by_kind"], {"solver": 1, "compaction": 1})
        self.assertTrue(receipt["exhausted"])
        self.assertEqual(receipt["tokens"]["input_tokens"]["value"], 20)
        self.assertEqual(receipt["tokens"]["cached_input_tokens"]["value"], 0)
        self.assertEqual(receipt["tokens"]["cached_input_tokens"]["status"], "complete")
        self.assertEqual(receipt["actual_http_requests"]["status"], "unavailable")
        self.assertIsNotNone(receipt["ended_at"])
        self.assertGreaterEqual(receipt["elapsed_seconds"], 0)
        self.assertNotIn("private content", repr(receipt))

    def test_response_usage_survives_parse_failure(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as state:
            with self.assertRaises(ValueError):
                model().query(fail_after=True)
        receipt = state.receipt()
        self.assertEqual(receipt["failed"], 1)
        self.assertEqual(receipt["provider_responses"], 1)
        self.assertEqual(receipt["tokens"]["input_tokens"]["value"], 10)
        self.assertNotIn("sensitive", repr(receipt))

    def test_retry_is_not_another_logical_call_and_missing_usage_is_partial(self):
        model, module = fixture()

        def completion(**kwargs):
            if kwargs.get("fail"):
                raise ValueError("private provider error")
            return response()

        module.completion = completion
        install_hooks(model, module)
        with task_telemetry(limit=1) as state:
            model().query(retry_once=True)
        receipt = state.receipt()
        self.assertEqual(receipt["total"], 1)
        self.assertEqual(receipt["sdk_completion_attempts"], 2)
        self.assertEqual(receipt["sdk_completion_retries"], 1)
        self.assertEqual(receipt["tokens"]["input_tokens"], {
            "value": 10, "status": "partial", "responses_with_value": 1,
            "responses_total": 1, "completion_attempts_total": 2,
        })

    def test_unavailable_usage_is_not_zero(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as state:
            model().query(response=response(usage=None))
        for metric in state.receipt()["tokens"].values():
            self.assertIsNone(metric["value"])
            self.assertEqual(metric["status"], "unavailable")
            self.assertEqual(metric["responses_total"], 1)

    def test_mixed_usage_has_per_metric_denominators(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as state:
            model().query()
            model().summarize_context(response=response(usage={"prompt_tokens": 8}))
        receipt = state.receipt()
        self.assertEqual(receipt["tokens"]["input_tokens"]["status"], "complete")
        self.assertEqual(receipt["tokens"]["input_tokens"]["value"], 18)
        self.assertEqual(receipt["tokens"]["output_tokens"]["status"], "partial")
        self.assertEqual(receipt["by_component"]["compaction"]["tokens"]["output_tokens"]["status"], "unavailable")

    def test_identity_violation_preserves_usage_and_aborts(self):
        model, module = fixture()
        install_hooks(model, module)
        self.assertIn(ModelIdentityViolation, model.abort_exceptions)
        with task_telemetry() as state:
            with self.assertRaises(ModelIdentityViolation):
                model().query(response=response(model="deepseek-v4-flash-ga-260731"))
        receipt = state.receipt()
        self.assertEqual(receipt["identity_status"], "violation")
        self.assertEqual(receipt["tokens"]["input_tokens"]["value"], 10)
        self.assertEqual(receipt["violations"][0]["reason"], "effective_model_mismatch")

    def test_nonthinking_reasoning_and_missing_model_fail_closed(self):
        cases = [
            (response(model=None), "effective_model_missing"),
            (response(usage={"completion_tokens_details": {"reasoning_tokens": 2}}), "nonthinking_reasoning_tokens"),
            (response(choices=[{"message": {"reasoning_content": "secret reasoning"}}]), "nonthinking_reasoning_content"),
            (response(choices=[{"message": {"provider_specific_fields": {"thinking_blocks": ["secret"]}}}]), "nonthinking_reasoning_content"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                model, module = fixture()
                install_hooks(model, module)
                with task_telemetry() as state:
                    with self.assertRaises(ModelIdentityViolation):
                        model().query(response=payload)
                self.assertEqual(state.receipt()["violations"][0]["reason"], expected)
                self.assertNotIn("secret", repr(state.receipt()))

    def test_empty_run_is_unavailable_and_no_context_does_not_count(self):
        model, module = fixture()
        install_hooks(model, module)
        model().query(response=response(model="other"))
        with task_telemetry() as state:
            pass
        self.assertEqual(state.receipt()["total"], 0)
        self.assertEqual(state.receipt()["identity_status"], "unavailable")
        self.assertIsNone(state.receipt()["tokens"]["input_tokens"]["value"])

    def test_threads_have_independent_contexts(self):
        model, module = fixture()
        install_hooks(model, module)

        def work(index):
            with task_telemetry(limit=5) as state:
                for _ in range(index):
                    model().query()
            return state.receipt()["total"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            self.assertEqual(list(executor.map(work, range(1, 6))), [1, 2, 3, 4, 5])

    def test_receipt_is_a_detached_snapshot(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as state:
            model().query()
        receipt = state.receipt()
        receipt["calls"][0]["status"] = "corrupted"
        self.assertEqual(state.receipt()["calls"][0]["status"], "succeeded")

    def test_nested_context_restores_parent(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as outer:
            model().query()
            with task_telemetry() as inner:
                model().summarize_context()
            model().query()
        self.assertEqual(outer.receipt()["total"], 2)
        self.assertEqual(inner.receipt()["total"], 1)

    def test_legacy_hook_is_rejected(self):
        model, module = fixture()
        model._n4c_logical_call_budget_installed = True
        with self.assertRaisesRegex(RuntimeError, "legacy"):
            install_hooks(model, module)

    def test_invalid_token_values_are_unavailable(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry() as state:
            model().query(response=response(usage={"prompt_tokens": True, "completion_tokens": -1}))
        self.assertIsNone(state.receipt()["tokens"]["input_tokens"]["value"])
        self.assertIsNone(state.receipt()["tokens"]["output_tokens"]["value"])

    def test_invalid_limits(self):
        for limit in (0, -1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                with task_telemetry(limit=limit):
                    pass

    def test_sdk_request_attestation_checks_both_components_without_secrets(self):
        model, module = fixture()
        install_hooks(model, module)
        with task_telemetry(expected_wire=WIRE) as state:
            model().query(**WIRE, api_key="secret-key")
            model().summarize_context(**WIRE, api_key="secret-key")
        receipt = state.receipt()
        self.assertEqual(receipt["configured_sdk_request_attestation"]["status"], "matched")
        self.assertEqual(receipt["configured_sdk_request_attestation"]["checks"], 2)
        check = receipt["calls"][0]["sdk_request_checks"][0]
        self.assertEqual(check["thinking_type"], "disabled")
        self.assertEqual(len(check["endpoint_digest"]), 64)
        self.assertNotIn("secret-key", repr(receipt))
        self.assertNotIn("provider.invalid", repr(receipt))

    def test_sdk_request_violation_blocks_provider_and_records_reason(self):
        cases = [
            {"model": "deepseek/other"}, {"api_base": "https://other.invalid"},
            {"temperature": 1}, {"extra_body": {"thinking": {"type": "enabled"}}},
            {"drop_params": True}, {"reasoning_effort": None},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                model, module = fixture()
                requests = []
                module.completion = lambda **kwargs: requests.append(kwargs) or response()
                install_hooks(model, module)
                with task_telemetry(expected_wire=WIRE) as state:
                    with self.assertRaises(ModelIdentityViolation):
                        model().query(**(WIRE | overrides))
                receipt = state.receipt()
                self.assertEqual(requests, [])
                self.assertEqual(receipt["sdk_completion_attempts"], 0)
                self.assertEqual(receipt["identity_status"], "violation")
                self.assertTrue(receipt["violations"])

    def test_progress_receipts_preserve_inflight_and_terminal_usage(self):
        model, module = fixture()
        snapshots = []
        install_hooks(model, module)
        with task_telemetry(receipt_callback=snapshots.append):
            with self.assertRaises(ValueError):
                model().query(fail_after=True)
        self.assertEqual(snapshots[0]["total"], 0)
        reserved = next(item for item in snapshots if item["total"] == 1)
        self.assertEqual(reserved["calls"][0]["status"], "running")
        self.assertEqual(reserved["sdk_completion_attempts"], 0)
        received = next(item for item in snapshots if item["provider_responses"] == 1)
        self.assertEqual(received["tokens"]["input_tokens"]["value"], 10)
        self.assertEqual(received["calls"][0]["status"], "running")
        self.assertEqual(snapshots[-1]["failed"], 1)
        self.assertIsNotNone(snapshots[-1]["ended_at"])

    def test_failed_progress_persistence_stops_before_provider(self):
        model, module = fixture()
        requests = []
        module.completion = lambda **kwargs: requests.append(kwargs) or response()
        install_hooks(model, module)

        def write(receipt):
            if receipt["total"]:
                raise OSError("disk full")

        with self.assertRaises(OSError):
            with task_telemetry(receipt_callback=write):
                model().query()
        self.assertEqual(requests, [])
        # The failed callback cannot leak task context into a later query.
        model().query()
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
