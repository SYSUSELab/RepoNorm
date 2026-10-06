import pytest

import importlib
from types import SimpleNamespace

import dspy
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize("enabled", [False, True])
def test_DSP_8f773dc75e34_N001(enabled):
    """Preserve existing context switches in observable final LM requests."""
    model = DummyLM([{"proposed_instruction": "instruction"}])
    program = dspy.Predict("question -> answer")
    proposer = GroundedProposer(
        prompt_model=model,
        program=program,
        trainset=[],
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=enabled,
        verbose=False,
    )
    # Avoid constructing a dataset summary; observe only its existing gate.
    proposer.use_dataset_summary = enabled
    proposer.data_summary = "SENTINEL_DATASET_NORM"
    proposer.propose_instruction_for_predictor(
        program, None, 0, 0.5, None, 0, {}, tip="SENTINEL_TIP_NORM"
    )
    assert model.history, "proposal never reached the final LM"
    request = "\n".join(message["content"] for message in model.history[-1]["messages"])
    assert ("SENTINEL_DATASET_NORM" in request) is enabled
    assert ("SENTINEL_TIP_NORM" in request) is enabled


def test_DSP_8f773dc75e34_N003():
    module = importlib.import_module("dspy.propose.grounded_proposer")
    generator = module.GenerateModuleInstruction(
        program_code_string="program",
        use_dataset_summary=False,
        program_aware=True,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=False,
        max_description_chars=20,
    )
    generator.describe_program = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("description failed"))
    generator.generate_module_instruction = lambda **kwargs: SimpleNamespace(proposed_instruction="instruction")

    result = generator.forward(None, 0, 0, dspy.Predict("question -> answer"), "", None)

    assert result.proposed_instruction == "instruction"
    assert generator.program_aware is False
