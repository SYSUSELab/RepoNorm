from dsp_rng_contracts import check_injected_rng

import importlib

import dspy
import pytest
from dspy.propose.grounded_proposer import GroundedProposer, generate_instruction_class
from dspy.utils.dummies import DummyLM


def test_DSP_8f773dc75e34_N001():
    full = generate_instruction_class().signature.fields
    reduced = generate_instruction_class(
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=False,
    ).signature.fields

    assert {"dataset_description", "program_description", "task_demos", "previous_instructions", "tip"} <= set(full)
    assert not {"dataset_description", "program_description", "previous_instructions", "tip"} & set(reduced)


def test_DSP_8f773dc75e34_N002(monkeypatch):
    check_injected_rng(monkeypatch)


def test_DSP_8f773dc75e34_N003(monkeypatch):
    module = importlib.import_module("dspy.propose.grounded_proposer")
    program = dspy.Predict("question -> answer")
    proposer = GroundedProposer(
        prompt_model=DummyLM([{"proposed_instruction": "unused"}]),
        program=program,
        trainset=[],
        program_aware=False,
        use_dataset_summary=False,
    )

    class FailingGenerator:
        def __init__(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            raise RuntimeError("instruction generation failed")

    monkeypatch.setattr(module, "GenerateModuleInstruction", FailingGenerator)
    with pytest.raises(RuntimeError, match="instruction generation failed"):
        proposer.propose_instruction_for_predictor(program, None, 0, 0.5, None, 0, {})
