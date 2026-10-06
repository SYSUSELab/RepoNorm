"""Offline tests for the downstream Beijing cache protocol."""
import argparse
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runner'))
from dashscope_profile import (
    BOUNDED_RECOVERY_PROFILE, RECOVERY_PROFILE, CACHE_POLICY, PROFILE,
    BoundedRecoveryDashScopeCodingProfile, DashScopeCodingProfile,
    RecoveryDashScopeCodingProfile, cache_messages, frozen_config,
    recovery_config,
)
from four_arm_common import model_kwargs, read_json
from task_telemetry import install_hooks, task_telemetry


class DashScopeTests(unittest.TestCase):



    def test_tool_block_cache_preserves_messages(self):
        messages = [{'role': 'system', 'content': 'rules'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call1'}]},
                    {'role': 'tool', 'tool_call_id': 'call1', 'content': 'tool result'}]
        original = copy.deepcopy(messages)
        result = cache_messages(messages)
        self.assertEqual(messages, original)
        self.assertEqual(result[-1]['tool_call_id'], 'call1')
        self.assertNotIn('cache_control', result[-1])
        self.assertEqual(result[-1]['content'][0]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(cache_messages(result), result)

    def test_solver_and_compaction_use_cache_and_receipts(self):
        sent = []
        def completion(**kwargs):
            sent.append(copy.deepcopy(kwargs))
            return {'model': 'qwen3.5-flash', 'choices': [], 'usage': {
                'prompt_tokens': 1200, 'completion_tokens': 30,
                'completion_tokens_details': {'reasoning_tokens': 20},
                'prompt_tokens_details': {'cached_tokens': 1024, 'cache_creation_input_tokens': 176}}}
        module = SimpleNamespace(completion=completion)
        config = frozen_config()
        wire = {'model': config['wire_model'], **model_kwargs(config)}
        class Model:
            abort_exceptions = []
            def query(self, messages):
                return module.completion(messages=messages, **wire)
            def summarize_context(self, messages, summary_prompt):
                return module.completion(messages=messages, **wire)
        install_hooks(Model, module)
        messages = [{'role': 'user', 'content': 'unchanged prompt'}]
        with task_telemetry(expected_model='qwen3.5-flash', expected_wire=wire,
                            expected_reasoning_mode='enabled', expected_reasoning_max_tokens=8192,
                            prompt_cache_policy=CACHE_POLICY) as state:
            Model().query(messages)
            Model().summarize_context(messages, 'summary')
        self.assertEqual(messages[0]['content'], 'unchanged prompt')
        self.assertEqual(len(sent), 2)
        for request in sent:
            self.assertEqual(request['messages'][0]['content'][0]['cache_control'], {'type': 'ephemeral'})
        receipt = state.receipt()
        self.assertEqual(receipt['tokens']['cached_input_tokens']['value'], 2048)
        self.assertEqual(receipt['identity_status'], 'observed')
        self.assertEqual(receipt['prompt_cache_policy'], CACHE_POLICY)

    def test_real_litellm_transform_retains_only_selected_cache_markers(self):
        from dashscope_profile import install_litellm_cache_passthrough
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        install_litellm_cache_passthrough()
        config = frozen_config()
        wire = {'model': config['wire_model'], **model_kwargs(config)}
        marked = cache_messages([{'role': 'tool', 'tool_call_id': 'call1', 'content': 'result'}])
        adapter = OpenAIGPTConfig()
        def transform():
            return adapter.transform_request(model='qwen3.5-flash', messages=copy.deepcopy(marked),
                                             optional_params={}, litellm_params={}, headers={})
        self.assertNotIn('cache_control', transform()['messages'][0]['content'][0])
        with task_telemetry(expected_model='qwen3.5-flash', expected_wire=wire,
                            expected_reasoning_mode='enabled', prompt_cache_policy=CACHE_POLICY):
            body = transform()
            self.assertEqual(body['messages'][0]['content'][0]['cache_control'], {'type': 'ephemeral'})
        self.assertNotIn('cache_control', transform()['messages'][0]['content'][0])


if __name__ == '__main__':
    unittest.main()
