import importlib
from types import SimpleNamespace

import dspy
import pytest


def make_generator():
    module = importlib.import_module("dspy.propose.grounded_proposer")
    generator = module.GenerateModuleInstruction(
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=False,
    )
    return generator


def test_DSP_8f773dc75e34_N003_strips_the_generated_prefix():
    generator = make_generator()
    generator.generate_module_instruction = lambda **kwargs: SimpleNamespace(
        proposed_instruction="PROPOSED INSTRUCTION: clean instruction"
    )

    result = generator.forward(None, 0, 0, dspy.Predict("question -> answer"), "", None)

    assert result.proposed_instruction == "clean instruction"


def test_DSP_8f773dc75e34_N003_propagates_final_generation_errors():
    generator = make_generator()
    generator.generate_module_instruction = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("generation failed"))

    with pytest.raises(RuntimeError, match="generation failed"):
        generator.forward(None, 0, 0, dspy.Predict("question -> answer"), "", None)
