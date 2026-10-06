from types import SimpleNamespace

import dspy
from dspy.propose import utils
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

    assert {"dataset_description", "program_description", "previous_instructions", "tip"} <= set(full)
    assert not {"dataset_description", "program_description", "previous_instructions", "tip"} & set(reduced)


def test_DSP_8f773dc75e34_N004(monkeypatch):
    class Program:
        def __init__(self, instruction="base"):
            self.instruction = instruction

        def deepcopy(self):
            return Program(self.instruction)

        def load(self, path):
            self.instruction = path

        def predictors(self):
            return [self]

    monkeypatch.setattr(utils, "get_signature", lambda predictor: SimpleNamespace(instructions=predictor.instruction))
    history = utils.create_predictor_level_history_string(
        Program(),
        0,
        {
            0: {"program_path": "low", "score": 1},
            1: {"program_path": "middle", "score": 4},
            2: {"program_path": "high", "score": 9},
            3: {"program_path": "high", "score": 7},
        },
        2,
    )

    assert history == "middle | Score: 4.0\n\nhigh | Score: 8.0\n\n"


def test_DSP_8f773dc75e34_N004_configured_limit_reaches_history_builder(monkeypatch):
    observed = []
    monkeypatch.setattr(
        "dspy.propose.grounded_proposer.create_predictor_level_history_string",
        lambda program, pred_i, logs, top_n: observed.append(top_n) or "",
    )
    program = dspy.Predict("question -> answer")
    proposer = GroundedProposer(
        prompt_model=DummyLM([{"proposed_instruction": "instruction"}]),
        program=program,
        trainset=[],
        program_aware=False,
        use_dataset_summary=False,
        set_tip_randomly=False,
        set_history_randomly=False,
        max_instruct_history=0,
    )

    proposer.propose_instruction_for_predictor(program, None, 0, 0.5, None, 0, {})

    assert observed == [0]
