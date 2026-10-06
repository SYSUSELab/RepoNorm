"""Downstream-only Beijing profile; independent of extraction model profiles."""
from copy import deepcopy
from functools import wraps
from typing import Literal

from pydantic import BaseModel, ConfigDict

from four_arm_common import digest

PROFILE = 'dashscope-beijing-qwen3.5-flash-thinking-8192-cache'
PROTOCOL = 'cooperbench-four-arm-1.13-r5-dashscope-qwen35-cache'
PROTOCOL_77 = 'cooperbench-four-arm-77-1.7-dashscope-qwen35-cache'
ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
BOUNDED_PROFILE = PROFILE + '-bounded'
BOUNDED_PROTOCOL = PROTOCOL + '-bounded-1'
BOUNDED_PROTOCOL_77 = PROTOCOL_77 + '-bounded-1'
RECOVERY_PROFILE = PROFILE + '-workspace-recovery'
RECOVERY_PROTOCOL = PROTOCOL + '-workspace-recovery-1'
RECOVERY_PROTOCOL_77 = PROTOCOL_77 + '-workspace-recovery-1'
BOUNDED_RECOVERY_PROFILE = BOUNDED_PROFILE + '-workspace-recovery'
BOUNDED_RECOVERY_PROTOCOL = BOUNDED_PROTOCOL + '-workspace-recovery-1'
BOUNDED_RECOVERY_PROTOCOL_77 = BOUNDED_PROTOCOL_77 + '-workspace-recovery-1'
CACHE_POLICY = 'dashscope-last-text-block-1.0'


class DashScopeCodingProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['downstream-dashscope-profile-1.0'] = 'downstream-dashscope-profile-1.0'
    profile_id: Literal['dashscope-beijing-qwen3.5-flash-thinking-8192-cache'] = PROFILE
    model: Literal['qwen3.5-flash'] = 'qwen3.5-flash'
    wire_model: Literal['openai/qwen3.5-flash'] = 'openai/qwen3.5-flash'
    endpoint: Literal['https://dashscope.aliyuncs.com/compatible-mode/v1'] = ENDPOINT
    credential_env: Literal['DASHSCOPE_API_KEY'] = 'DASHSCOPE_API_KEY'
    temperature: Literal[0] = 0
    enable_thinking: Literal[True] = True
    thinking_budget: Literal[8192] = 8192
    max_tokens: Literal[65536] = 65536
    prompt_cache_policy: Literal['dashscope-last-text-block-1.0'] = CACHE_POLICY
    weight_identity_status: Literal['floating alias; backend version unavailable'] = 'floating alias; backend version unavailable'

    @property
    def profile_digest(self) -> str:
        return digest(self.model_dump(mode='json'))


class BoundedDashScopeCodingProfile(DashScopeCodingProfile):
    profile_id: Literal['dashscope-beijing-qwen3.5-flash-thinking-8192-cache-bounded'] = BOUNDED_PROFILE
    api_timeout_seconds: Literal[120] = 120
    api_attempt_deadline_seconds: Literal[150] = 150
    api_max_attempts: Literal[2] = 2
    api_sdk_max_retries: Literal[0] = 0


class RecoveryDashScopeCodingProfile(DashScopeCodingProfile):
    profile_id: Literal['dashscope-beijing-qwen3.5-flash-thinking-8192-cache-workspace-recovery'] = RECOVERY_PROFILE


class BoundedRecoveryDashScopeCodingProfile(BoundedDashScopeCodingProfile):
    profile_id: Literal['dashscope-beijing-qwen3.5-flash-thinking-8192-cache-bounded-workspace-recovery'] = BOUNDED_RECOVERY_PROFILE


def bounded_config(dataset: str = 'engineering-v5') -> dict:
    config = frozen_config(dataset)
    config.update(protocol=BOUNDED_PROTOCOL if dataset == 'engineering-v5' else BOUNDED_PROTOCOL_77,
                  model_profile_id=BOUNDED_PROFILE, global_concurrency=6,
                  api_timeout_seconds=120, api_attempt_deadline_seconds=150,
                  api_max_attempts=2, api_sdk_max_retries=0)
    return config


def recovery_config(dataset: str = 'engineering-v5', *, bounded: bool = False) -> dict:
    config = bounded_config(dataset) if bounded else frozen_config(dataset)
    if bounded:
        config.update(
            protocol=(BOUNDED_RECOVERY_PROTOCOL if dataset == 'engineering-v5'
                      else BOUNDED_RECOVERY_PROTOCOL_77),
            model_profile_id=BOUNDED_RECOVERY_PROFILE,
        )
    else:
        config.update(
            protocol=RECOVERY_PROTOCOL if dataset == 'engineering-v5' else RECOVERY_PROTOCOL_77,
            model_profile_id=RECOVERY_PROFILE,
        )
    config.update(
        budget_exhaustion_workspace_recovery=True,
        empty_submission_workspace_recovery=True,
    )
    return config


def frozen_config(dataset: str = 'engineering-v5') -> dict:
    if dataset not in {'engineering-v5', 'engineering-77-v2'}:
        raise ValueError('Unsupported DashScope dataset')
    return {
        'protocol': PROTOCOL if dataset == 'engineering-v5' else PROTOCOL_77,
        'dataset': dataset, 'revision': 5 if dataset == 'engineering-v5' else 1,
        'agent': 'mini_swe_agent_v2', 'model_profile_id': PROFILE,
        'model': 'qwen3.5-flash', 'wire_model': 'openai/qwen3.5-flash',
        'endpoint': ENDPOINT, 'credential_env': 'DASHSCOPE_API_KEY',
        'response_model': 'qwen3.5-flash', 'expected_provider': None,
        'temperature': 0, 'thinking': {'type': 'enabled'},
        'reasoning_effort': None, 'reasoning_max_tokens': 8192,
        'extra_body': {'enable_thinking': True, 'thinking_budget': 8192},
        'max_tokens': 65536, 'prompt_cache_policy': CACHE_POLICY,
        'max_logical_llm_calls': 300, 'safety_step_limit': 330,
        'compaction_token_trigger': 28000, 'formal_attempts_per_task': 1,
        'global_concurrency': 20, 'solver_container_timeout': 'infinity',
        'command_timeout_seconds': 180, 'command_cleanup_grace_seconds': 10,
        'knowledge_generation_model_matches_coding_model': False,
        'evaluation_network': 'none', 'evaluation_timeout_seconds': 900,
        'weight_identity_status': 'floating alias; backend version unavailable',
    }


def cache_messages(messages: list[dict]) -> list[dict]:
    """One content-block breakpoint on the latest text, including tool results."""
    result = deepcopy(messages)
    for message in result:
        message.pop('cache_control', None)
        content = message.get('content')
        if isinstance(content, list):
            for block in content:
                block.pop('cache_control', None)
    for message in reversed(result):
        content = message.get('content')
        if isinstance(content, str) and content:
            message['content'] = [{'type': 'text', 'text': content,
                                   'cache_control': {'type': 'ephemeral'}}]
            return result
        if isinstance(content, list):
            for block in reversed(content):
                if block.get('type') == 'text' and block.get('text'):
                    block['cache_control'] = {'type': 'ephemeral'}
                    return result
    raise ValueError('DashScope cache policy requires a text content block')


def install_litellm_cache_passthrough() -> None:
    """LiteLLM's OpenAI adapter strips cache_control; retain it only in this scope."""
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
    if getattr(OpenAIGPTConfig, '_n4c_dashscope_cache_installed', False):
        return
    original = OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools

    @wraps(original)
    def preserve(self, model, messages, tools=None):
        from task_telemetry import _CURRENT
        state = _CURRENT.get()
        if state is not None and state.prompt_cache_policy == CACHE_POLICY:
            if model != 'qwen3.5-flash':
                raise ValueError('DashScope transformed model mismatch')
            return messages, tools
        return original(self, model=model, messages=messages, tools=tools)

    OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools = preserve
    OpenAIGPTConfig._n4c_dashscope_cache_installed = True
