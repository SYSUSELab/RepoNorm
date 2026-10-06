import random
from dsp_rng_contracts import check_injected_rng

import importlib
from types import MethodType

import dspy
import pytest
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


class RecordingRNG:
    def __init__(self):
        self.calls = []

    def random(self):
        self.calls.append("random")
        return 0.0

    def choice(self, values):
        self.calls.append("choice")
        return values[-1]

    def uniform(self, start, end):
        self.calls.append("uniform")
        return start


def make_proposer(rng):
    program = dspy.Predict("question -> answer")
    proposer = GroundedProposer(
        prompt_model=DummyLM([{"proposed_instruction": "instruction"}]),
        program=program,
        trainset=[],
        program_aware=False,
        use_dataset_summary=False,
        set_history_randomly=False,
        rng=rng,
        tip_weights={"none": 0, "creative": 7, "simple": 0, "description": 0, "high_stakes": 0, "persona": 0},
    )
    return proposer, program


def test_DSP_8f773dc75e34_N001():
    proposer, program = make_proposer(random.Random(8129))
    observed = []

    def capture(self, **kwargs):
        observed.append(kwargs["tip"])
        return "instruction"

    proposer.propose_instruction_for_predictor = MethodType(capture, proposer)
    proposer.propose_instructions_for_program([], program, None, {}, N=1, T=0.5)

    assert observed == ["Don't be afraid to be creative when creating the new instruction!"]


def test_DSP_8f773dc75e34_N002(monkeypatch):
    check_injected_rng(monkeypatch, weighted=True)
